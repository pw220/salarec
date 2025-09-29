"""
Utility functions and classes for SALARec.

This module consolidates common helpers used across the SALARec
implementation, including data loading, evaluation, batching and
reproducibility utilities. The goal of this file is to decouple
infrastructure code from the core model logic so that each component
remains testable and easy to maintain.

The functions defined here are adapted from our prototype training
script and extended to support more general use cases. In particular,
the data partitioning routine assumes that the raw interaction file
contains triples of ``user_id item_id rating`` where user and item
indices start from zero. Users with fewer than three interactions are
assigned all of their history to the training set and left without
validation or test items. Otherwise the last two interactions are
reserved for validation and test.

The :class:`WarpSampler` class wraps a multiprocessing queue to
generate mini‐batches of training samples on the fly. This decouples
data sampling from the main training loop and helps saturate GPU
utilisation. The sampler outputs four numpy arrays: users, histories,
positive items and negative items. Negative items are sampled
uniformly at random from the unobserved item set for the given user.

Evaluation metrics follow standard definitions for sequential
recommendation. We implement hit rate (HR) and normalised discounted
cumulative gain (NDCG) at various cutoffs. The ``full_evaluate``
function avoids materialising the entire item set by sampling
unobserved items for each user on the fly.

Finally, helper functions are provided to fix random seeds across
Python, NumPy and PyTorch, and to introspect model parameter counts
and estimated memory usage.
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "data_partition",
    "WarpSampler",
    "full_evaluate",
    "count_parameters",
    "format_bytes",
]


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch.

    Deterministic behaviour is critical when comparing algorithms or
    debugging. This helper fixes the global random state across
    multiple libraries. PyTorch's CUDA backend is also initialised
    deterministically when available.

    Args:
        seed: An integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # It is generally undesirable to use deterministic cuDNN kernels in
    # modern PyTorch versions since they may impact performance. We
    # therefore leave the cuDNN flags unset here. Should full
    # reproducibility be required, callers can override these flags.


def data_partition(file_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int]:
    """Partition user–item interaction data into train/valid/test splits.

    The input file is expected to contain whitespace‐separated triples
    ``user item rating``. Users and items are assumed to be indexed
    from zero; indices are incremented by one internally so that
    ``0`` can be reserved for padding. For each user with fewer than
    three interactions, all interactions are assigned to the training
    set. Otherwise the last two interactions are held out for
    validation and test respectively.

    Args:
        file_path: Path to the interaction file.

    Returns:
        A tuple ``(user_train, user_valid, user_test, num_users, num_items)``.
    """
    usernum = 0
    itemnum = 0
    # Map of user ID to the ordered list of interacted items
    User: Dict[int, List[int]] = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            u_str, i_str, _ = line.strip().split()
            u = int(u_str) + 1  # reserve index 0
            i = int(i_str) + 1
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, List[int]] = {}
    user_test: Dict[int, List[int]] = {}

    for user, items in User.items():
        n_feedback = len(items)
        if n_feedback < 3:
            # Not enough interactions to form validation/test sets
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            # Last two interactions are held out
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return user_train, user_valid, user_test, usernum, itemnum


def _random_neq(l: int, r: int, exclude: Iterable[int]) -> int:
    """Sample a random integer from [l, r) excluding a set of values."""
    t = np.random.randint(l, r)
    while t in exclude:
        t = np.random.randint(l, r)
    return t


def _sample_user_batch(
    user_train: Dict[int, List[int]],
    usernum: int,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    result_queue: Queue,
    seed: int,
) -> None:
    """Worker function for `WarpSampler` producing mini‐batches.

    Each batch contains tuples of (user, history sequence, positive
    target, negative target). The sequences are left padded with zeros
    to length ``maxlen``. Positive targets correspond to the next item
    in the sequence; negatives are sampled uniformly from items the
    user has not interacted with.
    """

    def sample() -> Tuple[int, List[int], List[int], List[int]]:
        # Draw a random user with at least two interactions
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)

        nxt = user_train[user][-1]
        idx = maxlen - 1
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = _random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return user, seq, pos, neg

    np.random.seed(seed)
    while True:
        batch = [sample() for _ in range(batch_size)]
        users, seqs, poss, negs = zip(*batch)
        result_queue.put((np.array(users), np.array(seqs), np.array(poss), np.array(negs)))


class WarpSampler:
    """Data loader producing training mini‐batches in parallel.

    The sampler spawns a number of worker processes that repeatedly
    generate random mini‐batches. Batches are placed on a shared
    multiprocessing queue and consumed by the main training loop via
    the :meth:`next_batch` method. Worker processes are terminated
    cleanly when :meth:`close` is invoked.
    """

    def __init__(
        self,
        user_train: Dict[int, List[int]],
        usernum: int,
        itemnum: int,
        batch_size: int = 64,
        maxlen: int = 10,
        n_workers: int = 1,
    ) -> None:
        self.result_queue: Queue = Queue(maxsize=n_workers * 10)
        self.processors: List[Process] = []
        for _ in range(n_workers):
            p = Process(
                target=_sample_user_batch,
                args=(user_train, usernum, itemnum, batch_size, maxlen, self.result_queue, np.random.randint(2**31)),
            )
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the next available mini‐batch."""
        return self.result_queue.get()

    def close(self) -> None:
        """Terminate all worker processes."""
        for p in self.processors:
            p.terminate()
            p.join()


@torch.no_grad()
def full_evaluate(
    model: torch.nn.Module,
    dataset: Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], int, int],
    cfg: Dict[str, any],
    device: torch.device,
    mode: str = "test",
    topk: Tuple[int, ...] = (5, 10, 20),
    max_users: int = 10000,
) -> Dict[str, float]:
    """Evaluate a model on either the validation or test split.

    This routine iterates over a subset of users (up to ``max_users``)
    and, for each user, constructs a candidate item set consisting of
    one held‐out positive item and a number of randomly sampled
    negatives. The model's ``predict`` method is called to score the
    sequence and candidate items. Rankings are used to compute hit
    rate and NDCG at the specified cutoffs.

    Args:
        model: The recommender model with a ``predict`` method.
        dataset: Tuple returned by :func:`data_partition`.
        cfg: Dictionary of configuration parameters (must contain
            ``maxlen``).
        device: Device on which the model lives.
        mode: Either ``"test"`` or ``"valid"`` to choose the held‐out
            split.
        topk: Cutoff values for which metrics are computed.
        max_users: Maximum number of users to sample for evaluation.

    Returns:
        A dictionary mapping metric names to their mean values across
        evaluated users.
    """
    assert mode in {"test", "valid"}, "mode must be either 'test' or 'valid'"
    train, valid, test, usernum, itemnum = dataset
    metrics = {f"NDCG@{k}": 0.0 for k in topk}
    metrics.update({f"HR@{k}": 0.0 for k in topk})
    valid_user = 0.0

    # Items and users are 1‐indexed; 0 is reserved for padding
    all_items: set = set(range(1, itemnum + 1))
    users = list(range(1, usernum + 1))
    if len(users) > max_users:
        users = random.sample(users, max_users)

    for u in users:
        # Skip users without the necessary held‐out items
        if len(train.get(u, [])) < 1:
            continue
        if mode == "valid" and not valid.get(u):
            continue
        if mode == "test" and not test.get(u):
            continue

        # Build the input sequence padded on the left
        seq = [0] * cfg["maxlen"]
        idx = cfg["maxlen"] - 1
        # For test mode, include the validation item as part of the input
        if mode == "test" and valid.get(u):
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        # Determine the ground truth item
        gt_item = valid[u][0] if mode == "valid" else test[u][0]

        # Sample negatives by excluding all items seen by the user
        rated = set(train[u] + valid.get(u, []) + test.get(u, []))
        rated.add(0)
        candidates = list(all_items - rated)
        # Append the ground truth at the end
        item_idx = candidates + [gt_item]

        scores = model.predict(np.array([seq]), np.array(item_idx))
        scores = scores[0].cpu().numpy()
        gt_score = scores[-1]
        rank = (scores >= gt_score).sum()

        valid_user += 1
        for k in topk:
            if rank <= k:
                metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)
                metrics[f"HR@{k}"] += 1.0

    for k in topk:
        if valid_user > 0:
            metrics[f"NDCG@{k}"] /= valid_user
            metrics[f"HR@{k}"] /= valid_user
    return metrics


def count_parameters(model: torch.nn.Module) -> Tuple[int, int, int]:
    """Count the total and trainable parameters of a model.

    Args:
        model: An instance of :class:`torch.nn.Module`.

    Returns:
        A tuple ``(total_params, trainable_params, bytes_per_param)`` where
        ``bytes_per_param`` corresponds to the size in bytes of a single
        parameter in the model (e.g. 4 for float32).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        bytes_per_param = next(model.parameters()).element_size()
    except StopIteration:
        bytes_per_param = 4
    return total, trainable, bytes_per_param


def format_bytes(n_bytes: int) -> str:
    """Format a byte count into a human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.2f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f}PB"