#!/usr/bin/env python
"""
Training script for the SALARec recommender model.

This script assembles the various components of SALARec into a
complete training pipeline. It supports configurable hyperparameters
via command line arguments and reports progress, memory usage and
evaluation metrics. The pipeline is intentionally modular: the
dataset loader, sampler, model and evaluation logic reside in
separate modules so that they can be easily swapped or tested
individually.

Example usage::

    python train.py --data /path/to/dataset.txt --batch-size 256 \
                    --max-len 50 --epochs 500

See ``python train.py --help`` for a complete list of options.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import numpy as np
import torch

from salarec.model import SALARec
from salarec.utils import (
    seed_everything,
    data_partition,
    WarpSampler,
    full_evaluate,
    count_parameters,
    format_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SALARec for sequential recommendation")
    parser.add_argument("--data", type=str, required=True, help="Path to the interaction file (user item rating)")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini‐batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--max-len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--hidden-units", type=int, default=64, help="Dimension of item embeddings")
    parser.add_argument("--num-blocks", type=int, default=2, help="Number of Transformer blocks")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate for embeddings and attention")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam optimiser")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Strength of adversarial perturbations (PAN)")
    parser.add_argument("--beta", type=float, default=0.001, help="Scale of Gaussian noise in PAN")
    parser.add_argument("--pan-dropout", type=float, default=0.1, help="Feature dropout probability in PAN")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for contrastive losses")
    parser.add_argument("--contrastive-weight", type=float, default=1.0, help="Weight of the contrastive loss term")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cpu or cuda)"
    )
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate on the test set every N epochs")
    parser.add_argument("--max-users", type=int, default=10000, help="Maximum users to sample during evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Load and split data
    print(f"[INFO] Loading data from {args.data}")
    user_train, user_valid, user_test, usernum, itemnum = data_partition(args.data)
    print(f"[INFO] Number of users: {usernum}, number of items: {itemnum}")
    dataset = (user_train, user_valid, user_test, usernum, itemnum)

    # Compute average sequence length for reporting
    avg_len = float(sum(len(seq) for seq in user_train.values())) / max(1, len(user_train))
    print(f"[INFO] Average training sequence length: {avg_len:.2f}")

    # Update args with derived parameters
    args.max_len = args.max_len  # alias for position embedding
    args.hidden_units = args.hidden_units
    args.num_blocks = args.num_blocks
    args.num_heads = args.num_heads
    args.dropout_rate = args.dropout_rate
    args.epsilon = args.epsilon
    args.beta = args.beta
    args.pan_dropout = args.pan_dropout
    args.temperature = args.temperature
    args.contrastive_weight = args.contrastive_weight
    args.device = device

    # Build model
    model = SALARec(usernum, itemnum, args).to(device)
    total_params, trainable_params, bpp = count_parameters(model)
    model_size = trainable_params * bpp
    print("=" * 80)
    print(
        f"[INFO] Model parameters: total={total_params:,}, trainable={trainable_params:,}, "
        f"approx size={format_bytes(model_size)}"
    )
    if torch.cuda.is_available():
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU")
    print("=" * 80)

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # Sampler for training data
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.max_len,
        n_workers=4,
    )

    num_batches = args.batch_size  # number of iterations per epoch (approximate)
    best_metrics: Dict[str, float] = {}
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_contrastive = 0.0
            t0 = time.time()
            for _ in range(num_batches):
                u, seq, pos, neg = sampler.next_batch()
                # Convert numpy arrays to tensors
                log_seqs = torch.LongTensor(seq).to(device)
                pos_seqs = torch.LongTensor(pos).to(device)
                optimizer.zero_grad()
                rec_loss, con_loss = model(log_seqs, pos_seqs)
                loss = rec_loss + args.contrastive_weight * con_loss
                if torch.isnan(loss):
                    raise RuntimeError("Loss became NaN")
                loss.backward()
                optimizer.step()
                epoch_loss += rec_loss.item()
                epoch_contrastive += con_loss.item()
            # Report training statistics
            elapsed = time.time() - t0
            avg_loss = epoch_loss / num_batches
            avg_con = epoch_contrastive / num_batches
            if torch.cuda.is_available():
                cur_mem = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                mem_str = f" | GPU Mem: {format_bytes(cur_mem)} (peak {format_bytes(peak_mem)})"
                torch.cuda.reset_peak_memory_stats()
            else:
                mem_str = ""
            print(
                f"[Epoch {epoch:04d}] Train Loss: {avg_loss:.4f} | Contrastive: {avg_con:.4f} | "
                f"Time: {elapsed:.2f}s{mem_str}"
            )
            # Periodic evaluation
            if epoch % args.eval_every == 0:
                model.eval()
                print("[INFO] Evaluating on validation and test sets...")
                # Validation metrics (use held out validation items)
                val_metrics = full_evaluate(
                    model,
                    dataset,
                    {"maxlen": args.max_len},
                    device,
                    mode="valid",
                    max_users=args.max_users,
                )
                test_metrics = full_evaluate(
                    model,
                    dataset,
                    {"maxlen": args.max_len},
                    device,
                    mode="test",
                    max_users=args.max_users,
                )
                print("[Validation]", ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
                print("[Test]", ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
                # Track the best test metric (e.g. NDCG@10)
                key = "NDCG@10"
                if key in test_metrics:
                    if not best_metrics or test_metrics[key] > best_metrics.get(key, 0.0):
                        best_metrics = {k: v for k, v in test_metrics.items()}
                        best_metrics["epoch"] = epoch
                        print(f"[INFO] New best {key}: {test_metrics[key]:.4f} at epoch {epoch}")
            # Optional early stopping could be added here based on validation loss
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        sampler.close()
        if best_metrics:
            print("=" * 80)
            print("[INFO] Best test metrics:")
            for k, v in best_metrics.items():
                if k != "epoch":
                    print(f"  {k}: {v:.4f}")
            print(f"  epoch: {best_metrics['epoch']}")
        print("[INFO] Training completed.")


if __name__ == "__main__":
    main()