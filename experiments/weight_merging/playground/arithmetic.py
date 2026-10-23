# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable CPU reference for five merges; no dependency on Marin arithmetic.

Coefficients weight donor-minus-anchor vectors. TIES elects a magnitude-weighted
sign, drops disagreeing updates, and divides by agreeing coefficient mass.
DARE uses Bernoulli retention with inverse-density scaling. Its masks reproduce
Marin's seed/name/donor-position contract; top-k trimming is tensor-local.
"""

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

import torch


class Method(StrEnum):
    AVERAGE = "average"
    TASK_ARITHMETIC = "task_arithmetic"
    TIES = "ties"
    DARE_LINEAR = "dare_linear"
    DARE_TIES = "dare_ties"


@dataclass(frozen=True)
class Recipe:
    method: Method
    coefficients: tuple[float, ...]
    density: float = 1.0
    scale: float = 1.0
    seed: int = 42
    ties_density: float = 1.0

    def __post_init__(self) -> None:
        if not self.coefficients or any(c < 0 or not math.isfinite(c) for c in self.coefficients):
            raise ValueError("Coefficients must be finite and nonnegative")
        if not 0 < self.density <= 1 or not 0 < self.ties_density <= 1 or not math.isfinite(self.scale):
            raise ValueError("Densities must be in (0, 1]; scale must be finite")
        if self.method == Method.AVERAGE:
            if sum(self.coefficients) > 1 or self.scale != 1 or self.density != 1:
                raise ValueError("Average requires a convex combination and unit scale/density")
        if self.method == Method.TASK_ARITHMETIC and self.density != 1:
            raise ValueError("Task arithmetic requires density=1")


def keep_largest(update: torch.Tensor, density: float) -> torch.Tensor:
    """Retain ceil(density * size) entries, with Torch's top-k tie behavior."""
    if density == 1:
        return update
    count = max(1, math.ceil(density * update.numel()))
    indices = update.flatten().abs().topk(count, sorted=False).indices
    mask = torch.zeros(update.numel(), dtype=torch.bool)
    mask[indices] = True
    return update * mask.reshape(update.shape)


def merge(anchor: torch.Tensor, donors: list[torch.Tensor], recipe: Recipe, name: str) -> torch.Tensor:
    """Merge aligned CPU tensors in FP32, then cast once to anchor dtype."""
    if len(donors) != len(recipe.coefficients):
        raise ValueError("Each donor needs one coefficient")
    for tensor in [anchor, *donors]:
        if tensor.device.type != "cpu" or tensor.shape != anchor.shape or tensor.dtype != anchor.dtype:
            raise ValueError(f"Unaligned CPU tensor: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite input: {name}")
    if not anchor.is_floating_point():
        if any(not torch.equal(anchor, donor) for donor in donors):
            raise ValueError(f"Non-floating tensor differs: {name}")
        return anchor.clone()
    base = anchor.float()
    if recipe.method == Method.AVERAGE:
        result = base * (1 - sum(recipe.coefficients))
        for coefficient, donor in zip(recipe.coefficients, donors, strict=True):
            result.add_(donor.float(), alpha=coefficient)
    else:
        updates = []
        for position, (coefficient, donor) in enumerate(zip(recipe.coefficients, donors, strict=True)):
            update = donor.float() - base
            if recipe.method in (Method.DARE_LINEAR, Method.DARE_TIES):
                digest = hashlib.sha256(f"{recipe.seed}:{name}:{position}".encode()).digest()
                rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
                retained = torch.rand(update.shape, generator=rng) < recipe.density
                update = update * retained / recipe.density
            density = recipe.density if recipe.method == Method.TIES else recipe.ties_density
            if recipe.method in (Method.TIES, Method.DARE_TIES):
                update = keep_largest(update, density)
            updates.append(update * coefficient)
        total = torch.zeros_like(base)
        for update in updates:
            total += update
        if recipe.method in (Method.TIES, Method.DARE_TIES):
            elected = total.sign()
            total = torch.zeros_like(base)
            mass = torch.zeros_like(base)
            for coefficient, update in zip(recipe.coefficients, updates, strict=True):
                agrees = (update.sign() == elected) & (update != 0)
                total += update * agrees
                mass.add_(agrees, alpha=coefficient)
            total /= mass.masked_fill(mass == 0, 1)
        result = total * recipe.scale + base
    result = result.to(anchor.dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"Merge overflow: {name}")
    return result
