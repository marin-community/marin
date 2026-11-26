# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-spectrum composition of aligned matrices (ISO, arXiv:2607.19331)."""

import torch

RIDGE = 1e-12
MAX_RETENTION = 1.5


def tangent(frame: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Project a displacement onto the frame's Stiefel tangent space."""
    product = frame.T @ displacement
    return displacement - frame @ ((product + product.T) / 2)


def retract(frame: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(frame, full_matrices=False)
    return left @ right


def isospectral_merge(
    base: torch.Tensor,
    donors: list[torch.Tensor],
    keep: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """Combine frame displacements with unit-retention coefficients and Base singular values.

    The caller selects complete matrices; masking reconstructed
    scalar coordinates would generally invalidate the spectrum constraint.
    """
    if base.ndim != 2 or not donors or not 0 < keep <= 1:
        raise ValueError("ISO requires a matrix, donors and mode retention in (0, 1]")
    if any(x.shape != base.shape or x.dtype != base.dtype for x in donors):
        raise ValueError("ISO inputs must have matching shapes and dtypes")
    original = base.to(device=device, dtype=torch.float64)
    u, spectrum, vt = torch.linalg.svd(original, full_matrices=False)
    v = vt.T
    modes = max(1, round(keep * len(spectrum)))
    changes = []
    effects = []
    for donor in donors:
        du, _, dvt = torch.linalg.svd(donor.to(device=device, dtype=torch.float64), full_matrices=False)
        signs = torch.where((du * u).sum(dim=0) >= 0, 1.0, -1.0)
        left = tangent(u, du * signs - u)
        right = tangent(v, dvt.T * signs - v)
        left[:, modes:] = 0
        right[:, modes:] = 0
        changes.append((left, right))
        effects.append(((left * spectrum) @ vt + (u * spectrum) @ right.T).reshape(-1))
    effects = torch.stack(effects)
    gram = effects @ effects.T
    coefficients, info = torch.linalg.solve_ex(
        gram + RIDGE * torch.eye(len(donors), device=device, dtype=torch.float64), gram.diag()
    )
    if info.item() != 0:
        # Match ISO's unit-strength fallback for a numerically singular Gram solve.
        coefficients = torch.ones(len(donors), device=device, dtype=torch.float64)
    coefficients = coefficients.clamp(0, MAX_RETENTION)
    left_update = torch.zeros_like(u)
    right_update = torch.zeros_like(v)
    for coefficient, (left, right) in zip(coefficients, changes, strict=True):
        left_update.add_(coefficient * left)
        right_update.add_(coefficient * right)
    result = (retract(u + tangent(u, left_update)) * spectrum) @ retract(v + tangent(v, right_update)).T
    result = result.to(device=base.device, dtype=base.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite ISO reconstruction")
    return result, coefficients.tolist()
