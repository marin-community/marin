# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP32 checkpoint merging with explicit task-vector and sparsification semantics."""

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

import torch


class MergeMethod(StrEnum):
    AVERAGE = "average"
    TASK_ARITHMETIC = "task_arithmetic"
    TIES = "ties"
    DARE_LINEAR = "dare_linear"
    DARE_TIES = "dare_ties"
    RAM = "ram"
    RAM_PLUS_TL = "ramplus_tl"


@dataclass(frozen=True)
class RamParameters:
    """Activity threshold and tensor-local unique-update amplification."""

    epsilon: float
    rescale: float
    ratio_cap: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value >= 0 for value in (self.epsilon, self.rescale, self.ratio_cap)):
            raise ValueError("RAM parameters must be finite and nonnegative")
        if self.epsilon == 0:
            raise ValueError("RAM requires a positive activity threshold")


@dataclass(frozen=True)
class MergeParameters:
    """Coefficients multiply donor weights or donor-minus-anchor task vectors.

    Average includes the anchor with weight ``1 - sum(coefficients)``.
    Task arithmetic and DARE-linear add an unnormalized sum of task vectors.
    TIES elects signs from weighted updates, averages the sign-consistent
    updates by their retained coefficient mass, then applies ``scale``.
    Density is applied separately to each tensor, never across the model.
    """

    method: MergeMethod
    coefficients: tuple[float, ...]
    density: float
    scale: float
    seed: int
    ram: RamParameters | None = None

    def __post_init__(self) -> None:
        if self.method in (MergeMethod.RAM, MergeMethod.RAM_PLUS_TL):
            if self.ram is None or self.density != 1 or self.scale != 1 or any(c != 1 for c in self.coefficients):
                raise ValueError("RAM requires explicit RAM parameters, unit coefficients, density=1, scale=1")
            if self.method == MergeMethod.RAM and self.ram.rescale != 0:
                raise ValueError("Use ramplus_tl for unique-update amplification")
        elif self.ram is not None:
            raise ValueError("RAM parameters only apply to RAM methods")
        if not self.coefficients or any(not math.isfinite(c) or c < 0 for c in self.coefficients):
            raise ValueError("Provide finite nonnegative donor coefficients")
        if not 0 < self.density <= 1 or not math.isfinite(self.scale):
            raise ValueError("Density must be in (0, 1] and scale must be finite")
        if self.method == MergeMethod.AVERAGE:
            if sum(self.coefficients) > 1 or self.density != 1 or self.scale != 1:
                raise ValueError("Averages require sum(coefficients) <= 1, density=1, scale=1")
        if self.method == MergeMethod.TASK_ARITHMETIC and self.density != 1:
            raise ValueError("Task arithmetic does not sparsify; use density=1")


def merge_tensor(
    anchor: torch.Tensor,
    donors: list[torch.Tensor],
    parameters: MergeParameters,
    *,
    tensor_name: str,
) -> torch.Tensor:
    """Merge aligned floating-point CPU tensors and preserve the anchor dtype.

    DARE masks depend on the seed, tensor name and donor position, so processing
    order and output sharding do not change the result. Exact zero sign votes
    in TIES produce no update. Non-floating tensors must agree exactly.
    """
    if len(donors) != len(parameters.coefficients):
        raise ValueError("Each donor needs one coefficient")
    for donor in donors:
        if donor.shape != anchor.shape or donor.dtype != anchor.dtype:
            raise ValueError(f"Incompatible tensor {tensor_name}: shape or dtype differs")
        if donor.device.type != "cpu":
            raise ValueError("Checkpoint merging expects CPU tensors")
    if anchor.device.type != "cpu":
        raise ValueError("Checkpoint merging expects CPU tensors")
    if not anchor.is_floating_point():
        if any(not torch.equal(anchor, donor) for donor in donors):
            raise ValueError(f"Non-floating tensor differs: {tensor_name}")
        return anchor.clone()
    if not torch.isfinite(anchor).all() or any(not torch.isfinite(d).all() for d in donors):
        raise ValueError(f"Nonfinite input: {tensor_name}")

    base = anchor.float()
    if parameters.method == MergeMethod.AVERAGE:
        result = base * (1 - sum(parameters.coefficients))
        for coefficient, donor in zip(parameters.coefficients, donors, strict=True):
            result.add_(donor.float(), alpha=coefficient)
        return _cast_result(result, anchor.dtype, tensor_name)

    if parameters.ram is not None:
        return _cast_result(_ram_merge(base, donors, parameters.ram), anchor.dtype, tensor_name)

    updates = []
    for index, (coefficient, donor) in enumerate(zip(parameters.coefficients, donors, strict=True)):
        update = donor.float() - base
        if parameters.method in (MergeMethod.DARE_LINEAR, MergeMethod.DARE_TIES):
            digest = hashlib.sha256(f"{parameters.seed}:{tensor_name}:{index}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
            mask = torch.rand(update.shape, generator=generator) < parameters.density
            update.mul_(mask).div_(parameters.density)
        elif parameters.method == MergeMethod.TIES and parameters.density < 1:
            count = max(1, math.ceil(update.numel() * parameters.density))
            indices = torch.topk(update.flatten().abs(), count, sorted=False).indices
            mask = torch.zeros(update.numel(), dtype=torch.bool)
            mask[indices] = True
            update.mul_(mask.reshape(update.shape))
        updates.append(update.mul_(coefficient))

    result = torch.zeros_like(base)
    for update in updates:
        result.add_(update)
    if parameters.method in (MergeMethod.TIES, MergeMethod.DARE_TIES):
        elected = result.sign()
        result.zero_()
        mass = torch.zeros_like(base)
        for coefficient, update in zip(parameters.coefficients, updates, strict=True):
            agree = (update.sign() == elected) & (update != 0)
            result.add_(update * agree)
            mass.add_(agree, alpha=coefficient)
        result.div_(mass.masked_fill(mass == 0, 1))
    result.mul_(parameters.scale).add_(base)
    return _cast_result(result, anchor.dtype, tensor_name)


def _cast_result(result: torch.Tensor, dtype: torch.dtype, tensor_name: str) -> torch.Tensor:
    result = result.to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"Merge overflow: {tensor_name}")
    return result


def _ram_merge(base: torch.Tensor, donors: list[torch.Tensor], parameters: RamParameters) -> torch.Tensor:
    # Count active donors without stacking all full-sized expert-bank tensors.
    counts = torch.zeros_like(base, dtype=torch.int32)
    for donor in donors:
        counts.add_((donor.float() - base).abs() > parameters.epsilon)
    shared = counts > 1
    unique = counts == 1
    denominator = counts.clamp_min(1)
    result = base.clone()
    for donor in donors:
        delta = donor.float() - base
        active = delta.abs() > parameters.epsilon
        unique_count = (active & unique).sum().item()
        shared_count = (active & shared).sum().item()
        ratio = shared_count / max(unique_count, parameters.epsilon)
        amplification = 1 + parameters.rescale * min(ratio, parameters.ratio_cap)
        delta.mul_(active).div_(denominator)
        delta.mul_(1 + unique * (amplification - 1))
        result.add_(delta)
    return result
