"""
Augmentation strategies for sequential recommendation.

This module implements a collection of data augmentation operators for
constructing multiple views of a user interaction sequence. Random
perturbations such as cropping, masking and reordering are provided
alongside a preference‐aware adversarial augmentation (PAN) inspired
by the SALARec paper. PAN adaptively perturbs item embeddings based
on the gradient of the recommendation loss and a learnable preference
probe. The resulting views are used to build robust contrastive
learning objectives.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RandomCrop",
    "RandomMask",
    "RandomReorder",
    "RandomSubSplit",
    "RandomAugmentSub",
    "PreferenceAwareAugmentation",
]


class RandomCrop:
    """Randomly crop a contiguous subsequence from a sequence.

    The cropping ratio ``tau`` determines the length of the subsequence
    relative to the original. Padding or truncation back to the
    original maximum length is handled outside of this operator.
    """

    def __init__(self, tau: float = 0.2) -> None:
        self.tau = tau

    def __call__(self, sequence: List[int]) -> List[int]:
        copied = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        sub_len = max(1, int(self.tau * len(copied)))
        if len(copied) <= sub_len:
            return copied
        start = random.randint(0, len(copied) - sub_len)
        return copied[start : start + sub_len]


class RandomMask:
    """Randomly mask elements in a sequence.

    Each element is independently masked with probability ``gamma``. The
    masked token is replaced by zero (reserved for padding).
    """

    def __init__(self, gamma: float = 0.7) -> None:
        self.gamma = gamma

    def __call__(self, sequence: List[int]) -> List[int]:
        copied = sequence.clone().tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        mask_count = int(self.gamma * len(copied))
        if mask_count == 0:
            return copied
        mask_idx = random.sample(range(len(copied)), k=mask_count)
        for idx in mask_idx:
            copied[idx] = 0
        return copied


class RandomReorder:
    """Randomly reorder a contiguous subsequence of a sequence.

    A subsequence of length ``beta`` times the original length is
    selected and shuffled in place.
    """

    def __init__(self, beta: float = 0.7) -> None:
        self.beta = beta

    def __call__(self, sequence: List[int]) -> List[int]:
        copied = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        sub_len = max(1, int(self.beta * len(copied)))
        if len(copied) <= sub_len:
            return copied
        start = random.randint(0, len(copied) - sub_len)
        sub_seq = copied[start : start + sub_len]
        random.shuffle(sub_seq)
        return copied[:start] + sub_seq + copied[start + sub_len :]


class RandomSubSplit:
    """Randomly drop items from a sequence (subsequence splitting)."""

    def __init__(self, theta: float = 0.3) -> None:
        self.theta = theta

    def __call__(self, sequence: List[int]) -> List[int]:
        copied = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        augmented = [val for val in copied if random.random() > self.theta]
        if not augmented:
            augmented.append(random.choice(copied))
        return augmented


class RandomAugmentSub:
    """Apply a random subsequence splitting augmentation to a batch of sequences.

    This operator picks one of the provided augmentation strategies for
    each sequence in the batch and pads or truncates the result to a
    maximum length.
    """

    def __init__(self, maxlen: int = 50, methods: Optional[List] = None) -> None:
        # By default only use sub‐sequence splitting. Masking and cropping
        # can be added by the caller if desired.
        if methods is None:
            methods = [RandomSubSplit()]
        self.methods = methods
        self.maxlen = maxlen

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        is_tensor = isinstance(sequences, torch.Tensor)
        if is_tensor:
            original_dtype = sequences.dtype
            original_device = sequences.device
            sequences = sequences.tolist()
        augmented_sequences: List[List[int]] = []
        for seq in sequences:
            method = random.choice(self.methods)
            aug = method(seq)
            # left pad to maxlen
            pad_len = self.maxlen - len(aug)
            if pad_len > 0:
                aug = [0] * pad_len + aug
            else:
                aug = aug[-self.maxlen :]
            augmented_sequences.append(aug)
        if is_tensor:
            return torch.tensor(augmented_sequences, dtype=original_dtype, device=original_device)
        return augmented_sequences


class PreferenceAwareAugmentation(nn.Module):
    """Preference‐Aware Adversarial Augmentation (PAN).

    PAN constructs adversarial views of a sequence by adding several
    perturbations to the item embeddings. The perturbation is
    comprised of three terms:

    * A sign‐normalised adversarial gradient scaled by ``epsilon``.
    * A learnable weight ``alpha`` derived from a preference probe on
      contextualised item representations.
    * Feature‐level dropout with probability ``dropout_p`` and small
      Gaussian noise scaled by ``beta``.

    When called, two independent views are generated using freshly
    sampled dropout masks and noise, but the gradient and preference
    weights are shared. The returned views are detached from the
    computation graph so that gradients do not propagate through the
    augmentation pipeline.

    Args:
        hidden_units: Dimensionality of item embeddings.
        epsilon: Global scaling factor controlling adversarial strength.
        beta: Standard deviation of Gaussian noise.
        dropout_p: Dropout probability for feature masking.
        temperature: Temperature used in the contrastive loss (for
            consistency with SALARec but unused here).
    """

    def __init__(
        self,
        hidden_units: int,
        epsilon: float = 0.01,
        beta: float = 0.001,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta
        self.dropout_p = dropout_p
        # Preference probe: maps contextual representation to a scalar weight
        self.pref_probe = nn.Linear(hidden_units, 1)

    def _generate_view(
        self,
        seq_emb: torch.Tensor,
        rec_loss: torch.Tensor,
        encode_func,
    ) -> torch.Tensor:
        """Generate a single adversarial view of the sequence embeddings.

        Args:
            seq_emb: Raw item embeddings of shape (B, T, D).
            rec_loss: Scalar recommendation loss used to compute gradients.
            encode_func: Function mapping embeddings to contextualised
                representations (list of layers). Should accept a batch
                of embeddings and return a list of tensors of shape
                (B, T, D).

        Returns:
            Adversarially perturbed embeddings of shape (B, T, D).
        """
        # Compute gradient of the recommendation loss with respect to the
        # raw embeddings. The gradient carries information about which
        # dimensions of each embedding contribute most strongly to the
        # loss. We detach the loss from the graph before backward to
        # avoid accumulating gradients in higher layers.
        grad = torch.autograd.grad(
            outputs=rec_loss,
            inputs=seq_emb,
            retain_graph=True,
            create_graph=False,
        )[0]
        # Normalise the gradient to its sign and scale by epsilon
        r = self.epsilon * grad.sign()
        # Obtain contextualised representations (without updating the
        # gradient) to compute the preference weights alpha. We detach
        # here to avoid backpropagating through the encoder during
        # augmentation.
        with torch.no_grad():
            context = encode_func(seq_emb)[-1]  # use the final layer
        alpha = torch.sigmoid(self.pref_probe(context))  # (B, T, 1)
        # Broadcast alpha to match embedding dimensions
        alpha = alpha.expand_as(seq_emb)
        # Feature‐level dropout mask: 1 means keep, 0 means drop
        mask = torch.bernoulli(
            torch.full_like(seq_emb, 1.0 - self.dropout_p, device=seq_emb.device)
        )
        noise = torch.randn_like(seq_emb) * self.beta
        perturbed = seq_emb * mask + alpha * r + noise
        return perturbed.detach()

    def forward(
        self,
        seq_emb: torch.Tensor,
        rec_loss: torch.Tensor,
        encode_func,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two independent adversarial views.

        Args:
            seq_emb: Raw item embeddings (B, T, D).
            rec_loss: Recommendation loss from which gradients are
                computed.
            encode_func: Function mapping embeddings to contextualised
                representations.

        Returns:
            A tuple of two adversarially perturbed embeddings.
        """
        view1 = self._generate_view(seq_emb, rec_loss, encode_func)
        view2 = self._generate_view(seq_emb, rec_loss, encode_func)
        return view1, view2