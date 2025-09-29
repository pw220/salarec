"""
Model definition for SALARec.

This module contains the implementation of the SALARec model, which
combines a self‐attention based encoder with dual contrastive
objectives. The core ideas are drawn from the paper *SALARec: Dual
Alignment Contrastive Learning with Preference-Aware Adversarial
Augmentation for Sequential Recommendation* (ESWA 2025). The model
consists of three main components:

1. A Transformer encoder stack that maps a sequence of item
   embeddings into contextualised representations. Each layer
   comprises multi‐head self attention followed by a gated feed
   forward network and residual connections.
2. A recommendation head that predicts the next item given the
   encoded sequence. It uses a softmax over all items and trains
   using cross‐entropy loss.
3. A dual contrastive learning objective composed of layer‐wise
   cross‐view alignment (LCA) and layer‐wise intra‐view self
   alignment (LISA). LCA aligns representations at the same layer
   across two preference‐aware adversarially augmented views. LISA
   aligns adjacent layers within the same view to encourage smooth
   transitions of user intent.

The `SALARec` class exposes methods for computing the
recommendation and contrastive losses separately, as well as a
combined forward pass. Prediction for evaluation is provided via the
`predict` method, which returns scores for candidate items given a
historical sequence.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import PreferenceAwareAugmentation

__all__ = ["SALARec"]


class GatedFeedForward(nn.Module):
    """Gated feed forward network used within the Transformer encoder.

    This layer doubles the dimensionality of the input via a 1×1
    convolution and splits it into two halves: one is gated by a
    sigmoid activation and multiplied elementwise with the other. The
    result is then projected back to the original dimensionality. A
    residual connection adds the input to the output. Dropout is
    applied to the intermediate representation.
    """

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(hidden_units, hidden_units * 2, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x_t = x.transpose(-1, -2)  # (B, D, L)
        x_proj = self.fc1(x_t)  # (B, 2D, L)
        x1, x2 = x_proj.chunk(2, dim=1)
        gated = x1 * torch.sigmoid(x2)
        out = self.fc2(self.dropout(gated))  # (B, D, L)
        out = out.transpose(-1, -2)  # (B, L, D)
        return out + x  # residual


class SALARec(nn.Module):
    """Self Alignment and Layer Aware Recommendation model."""

    def __init__(self, user_num: int, item_num: int, args) -> None:
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = args.hidden_units
        self.num_layers = args.num_blocks
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate
        self.max_len = args.max_len
        self.temperature = getattr(args, "temperature", 0.2)
        self.contrastive_weight = getattr(args, "contrastive_weight", 1.0)
        self.device = args.device

        # Embedding layers
        self.item_emb = nn.Embedding(item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        # Transformer layers
        self.attn_norms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_norms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.attn_layers.append(
                nn.MultiheadAttention(self.hidden_units, self.num_heads, dropout=self.dropout_rate)
            )
            self.ffn_norms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.ffn_layers.append(GatedFeedForward(self.hidden_units, self.dropout_rate))
        self.final_norm = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Adversarial augmentation module
        self.pan = PreferenceAwareAugmentation(
            hidden_units=self.hidden_units,
            epsilon=getattr(args, "epsilon", 0.01),
            beta=getattr(args, "beta", 0.001),
            dropout_p=getattr(args, "pan_dropout", 0.1),
        )

        # Initialise embeddings: set padding token weights to zero
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight.data[0, :] = 0.0
            self.pos_emb.weight.data[0, :] = 0.0

    def _add_position_embedding(self, seq_emb: torch.Tensor, log_seqs: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to item embeddings.

        The input ``seq_emb`` is expected to be already scaled by
        ``sqrt(hidden_units)``. Positions corresponding to padding
        indices (value zero) receive no positional embedding. Sequence
        positions are 1‐indexed.
        """
        batch_size, seq_len, _ = seq_emb.size()
        positions = torch.arange(1, seq_len + 1, device=seq_emb.device).unsqueeze(0).repeat(batch_size, 1)
        # Zero out positions where the item id is 0 (padding)
        mask = (log_seqs == 0)
        positions = positions * (~mask)
        pos_emb = self.pos_emb(positions)
        return seq_emb + pos_emb

    def encode(self, seqs: torch.Tensor) -> List[torch.Tensor]:
        """Pass a batch of sequences through the Transformer encoder.

        Args:
            seqs: Tensor of shape (B, T, D) with item+pos embeddings.

        Returns:
            A list of layer outputs; each element has shape (B, T, D).
        """
        layer_outputs: List[torch.Tensor] = []
        # Causal mask to prevent attending to future positions
        # attn_mask shape expected by PyTorch MultiheadAttention is (L, L)
        seq_len = seqs.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=seqs.device, dtype=torch.bool), diagonal=1)
        x = seqs
        for i in range(self.num_layers):
            # MultiheadAttention expects (L, B, D)
            x_t = x.transpose(0, 1)
            q = self.attn_norms[i](x_t)
            attn_output, _ = self.attn_layers[i](q, x_t, x_t, attn_mask=attn_mask)
            x_t = q + attn_output
            x = x_t.transpose(0, 1)
            x = self.ffn_norms[i](x)
            x = self.ffn_layers[i](x)
            layer_outputs.append(x)
        return layer_outputs

    def _infonce_loss(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute the InfoNCE loss between query and key embeddings.

        Both tensors must be of shape (B, D). The loss encourages each
        query to align with its corresponding key while pushing away
        other keys in the mini‐batch. A temperature hyperparameter
        controls the sharpness of the distribution.
        """
        # Normalise to unit length
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(key, dim=-1)
        # Similarity matrix (B, B)
        logits = torch.matmul(query_norm, key_norm.t())
        labels = torch.arange(query.size(0), device=query.device)
        logits /= self.temperature
        return F.cross_entropy(logits, labels)

    def _lca_loss(self, view1_layers: List[torch.Tensor], view2_layers: List[torch.Tensor]) -> torch.Tensor:
        """Layer‐wise cross‐view alignment (LCA) loss.

        For each layer, we take the representation of the last time step
        from two augmented views and compute an InfoNCE loss. The
        average over all layers constitutes the LCA loss.
        """
        losses = []
        for v1, v2 in zip(view1_layers, view2_layers):
            q = v1[:, -1, :]
            k = v2[:, -1, :]
            losses.append(self._infonce_loss(q, k))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _lisa_loss(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """Layer‐wise intra‐view self alignment (LISA) loss.

        Adjacent layers within the same view are encouraged to produce
        similar representations at the last time step. We exclude the
        input embedding layer (layer 0) as suggested in the paper to
        avoid unstable shallow representations.
        """
        losses = []
        # Exclude the very first layer if desired; here we align all
        # adjacent pairs starting from the first encoder layer
        for i in range(len(layers) - 1):
            q = layers[i][:, -1, :]
            k = layers[i + 1][:, -1, :]
            losses.append(self._infonce_loss(q, k))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def recommendation_loss(
        self,
        log_seqs: torch.Tensor,
        pos_seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Compute the next‐item prediction loss.

        Args:
            log_seqs: Input sequences of shape (B, T) with item IDs.
            pos_seqs: Target sequences of shape (B, T) with item IDs.

        Returns:
            A tuple ``(loss, raw_embeddings, layer_outputs)`` where
            ``raw_embeddings`` are the unscaled item embeddings (B, T, D)
            before positional encodings, and ``layer_outputs`` is the
            list of encoder layer outputs for the current batch.
        """
        # Raw item embeddings (before scaling and position)
        seq_emb_raw = self.item_emb(log_seqs)  # (B, T, D)
        # Scale embeddings
        seq_emb = seq_emb_raw * math.sqrt(self.hidden_units)
        # Add positional encodings
        seq_emb = self._add_position_embedding(seq_emb, log_seqs)
        seq_emb = self.emb_dropout(seq_emb)
        # Encode through transformer
        layers = self.encode(seq_emb)
        # Final layer normalisation
        last_layer = self.final_norm(layers[-1])
        # Compute logits for each position over all items
        # last_layer: (B, T, D), item_emb.weight: (N+1, D)
        logits = torch.matmul(last_layer, self.item_emb.weight.t())  # (B, T, N+1)
        # Do not predict padding token
        logits[..., 0] = -float('inf')
        # Flatten for cross entropy: ignore padding positions in targets
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(logits.view(-1, self.item_num + 1), pos_seqs.view(-1))
        return loss, seq_emb_raw, layers

    def forward(
        self,
        log_seqs: torch.Tensor,
        pos_seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the recommendation and contrastive losses.

        This method wraps the entire forward computation including
        recommendation loss, generation of adversarial views, and both
        LCA and LISA losses. The returned losses can be weighted and
        combined externally.
        """
        rec_loss, seq_emb_raw, original_layers = self.recommendation_loss(log_seqs, pos_seqs)
        # Generate two adversarial views using PAN. We only pass the raw
        # embeddings and the recommendation loss; the encode function is
        # bound with positional encoding and dropout for context.
        def context_encode(x: torch.Tensor) -> List[torch.Tensor]:
            # x: raw embeddings (B, T, D)
            x_scaled = x * math.sqrt(self.hidden_units)
            x_pos = self._add_position_embedding(x_scaled, log_seqs)
            x_pos = self.emb_dropout(x_pos)
            return self.encode(x_pos)

        view1_raw, view2_raw = self.pan(seq_emb_raw, rec_loss, context_encode)
        # Prepare the perturbed embeddings for the encoder (scale and add pos)
        view1 = view1_raw * math.sqrt(self.hidden_units)
        view2 = view2_raw * math.sqrt(self.hidden_units)
        view1 = self._add_position_embedding(view1, log_seqs)
        view2 = self._add_position_embedding(view2, log_seqs)
        view1 = self.emb_dropout(view1)
        view2 = self.emb_dropout(view2)
        # Encode augmented views
        view1_layers = self.encode(view1)
        view2_layers = self.encode(view2)
        # Contrastive losses
        lca = self._lca_loss(view1_layers, view2_layers)
        # LISA on both augmented views; average them
        lisa1 = self._lisa_loss(view1_layers)
        lisa2 = self._lisa_loss(view2_layers)
        lisa = (lisa1 + lisa2) * 0.5
        contrastive = lca + lisa
        return rec_loss, contrastive

    @torch.no_grad()
    def predict(
        self, log_seqs: np.ndarray, item_indices: np.ndarray
    ) -> torch.Tensor:
        """Score candidate items given a batch of sequences.

        Args:
            log_seqs: Array of shape (B, T) containing item IDs.
            item_indices: Array of shape (B, K) containing candidate item IDs.

        Returns:
            A tensor of shape (B, K) with predicted scores for each
            candidate item.
        """
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs).to(self.device)
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.LongTensor(item_indices).to(self.device)
        # Raw embeddings
        seq_emb = self.item_emb(log_seqs) * math.sqrt(self.hidden_units)
        seq_emb = self._add_position_embedding(seq_emb, log_seqs)
        seq_emb = self.emb_dropout(seq_emb)
        layers = self.encode(seq_emb)
        last_layer = self.final_norm(layers[-1])
        final_feat = last_layer[:, -1, :]  # (B, D)
        item_embs = self.item_emb(item_indices)  # (B, K, D)
        logits = (item_embs * final_feat.unsqueeze(1)).sum(-1)
        return logits