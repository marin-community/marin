# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure checkpoint weights and updates without full-size float64 copies."""

import torch


def weight_update_gram(anchor: torch.Tensor, donors: list[torch.Tensor], *, chunk_elements: int) -> torch.Tensor:
    """Return FP64 dot products of [anchor, donors..., donor-anchor updates...].

    Zero vectors retain zero dot products; their cosine similarities are undefined.
    """
    if any(tensor.shape != anchor.shape for tensor in donors):
        raise ValueError("Checkpoint tensor shapes differ")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    flattened = [tensor.reshape(-1) for tensor in [anchor, *donors]]
    count = 1 + 2 * len(donors)
    gram = torch.zeros((count, count), dtype=torch.float64, device=anchor.device)
    for start in range(0, anchor.numel(), chunk_elements):
        weights = [tensor[start : start + chunk_elements].double() for tensor in flattened]
        # Explicit deltas preserve small differences between nearly identical weights.
        vectors = torch.stack([*weights, *(weight - weights[0] for weight in weights[1:])])
        gram += vectors @ vectors.T
    if not torch.isfinite(gram).all():
        raise ValueError("Non-finite checkpoint geometry")
    return gram
