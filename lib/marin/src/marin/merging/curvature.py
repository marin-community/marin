# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OTA aggregation and Fisher grafting for aligned checkpoint tensors."""

import math

import torch


def fisher_graft_mask(delta: torch.Tensor, second_moment: torch.Tensor, density: float) -> torch.Tensor:
    """Select the highest curvature-weighted edit saliencies within one tensor.

    Tensor-local selection is an explicit adaptation of global FFG selection.
    Zero-saliency coordinates are never selected, including at full density.
    """
    if not 0 < density <= 1:
        raise ValueError("Density must be in (0, 1]")
    if delta.shape != second_moment.shape:
        raise ValueError("Task vector and curvature must have identical shapes")
    if not torch.isfinite(second_moment).all() or torch.any(second_moment < 0):
        raise ValueError("Second moments must be finite and nonnegative")
    saliency = delta.float().square() * second_moment.float()
    if not torch.isfinite(saliency).all():
        raise ValueError("Nonfinite graft saliency")
    if density == 1:
        return saliency > 0
    count = max(1, math.ceil(saliency.numel() * density))
    indices = torch.topk(saliency.flatten(), count, sorted=False).indices
    mask = torch.zeros(saliency.numel(), device=saliency.device, dtype=torch.bool)
    mask[indices] = True
    return mask.reshape(saliency.shape) & (saliency > 0)


def ota_merge_tensor(
    anchor: torch.Tensor,
    donors: list[torch.Tensor],
    second_moments: list[torch.Tensor],
    *,
    density: float,
    epsilon: float,
) -> torch.Tensor:
    """Merge FFG-selected updates with sqrt(second moment) preconditioners.

    All donors contribute to the denominator even when their update is grafted
    back to the anchor. Inputs may be optimizer moments or calibrated squared
    gradients; the caller must record which estimator produced them.
    """
    if not donors or len(donors) != len(second_moments):
        raise ValueError("Each donor requires a second-moment tensor")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Preconditioner epsilon must be finite and positive")
    base = anchor.float()
    numerator = torch.zeros_like(base)
    denominator = torch.zeros_like(base)
    for donor, moment in zip(donors, second_moments, strict=True):
        if donor.shape != anchor.shape or donor.dtype != anchor.dtype or moment.shape != anchor.shape:
            raise ValueError("Donor weights and moments must align with the anchor")
        delta = donor.float() - base
        mask = fisher_graft_mask(delta, moment, density)
        preconditioner = moment.float().sqrt().add_(epsilon)
        numerator.add_(preconditioner * delta * mask)
        denominator.add_(preconditioner)
    result = (base + numerator / denominator).to(anchor.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite curvature merge")
    return result
