# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Differentiable mixing of frozen CPU weights with bounded GPU temporaries."""

import torch
from torch.autograd.function import FunctionCtx


class _BlendContext(FunctionCtx):
    anchor: torch.Tensor
    donors: tuple[torch.Tensor, ...]
    coefficient_shape: torch.Size
    block_elements: int


class _FrozenWeightBlend(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _BlendContext,
        coefficients: torch.Tensor,
        anchor: torch.Tensor,
        block_elements: int,
        *donors: torch.Tensor,
    ) -> torch.Tensor:
        ctx.anchor = anchor
        ctx.donors = donors
        ctx.coefficient_shape = coefficients.shape
        ctx.block_elements = block_elements
        output = torch.empty(anchor.numel(), dtype=anchor.dtype, device=coefficients.device)
        for chunk in range(coefficients.shape[0]):
            chunk_start = anchor.numel() * chunk // coefficients.shape[0]
            chunk_end = anchor.numel() * (chunk + 1) // coefficients.shape[0]
            for start in range(chunk_start, chunk_end, block_elements):
                end = min(chunk_end, start + block_elements)
                base = anchor.flatten()[start:end].to(device=coefficients.device, dtype=torch.float32)
                value = base.clone()
                for donor_index, donor in enumerate(donors):
                    delta = donor.flatten()[start:end].to(device=coefficients.device, dtype=torch.float32) - base
                    value.add_(coefficients[chunk, donor_index] * delta)
                output[start:end] = value
        return output.reshape(anchor.shape)

    @staticmethod
    def backward(ctx: _BlendContext, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (gradient,) = grad_outputs
        gradient_flat = gradient.reshape(-1)
        coefficient_gradient = torch.zeros(ctx.coefficient_shape, device=gradient.device, dtype=torch.float32)
        for chunk in range(ctx.coefficient_shape[0]):
            chunk_start = gradient.numel() * chunk // ctx.coefficient_shape[0]
            chunk_end = gradient.numel() * (chunk + 1) // ctx.coefficient_shape[0]
            for start in range(chunk_start, chunk_end, ctx.block_elements):
                end = min(chunk_end, start + ctx.block_elements)
                base = ctx.anchor.flatten()[start:end].to(device=gradient.device, dtype=torch.float32)
                grad = gradient_flat[start:end].float()
                for donor_index, donor in enumerate(ctx.donors):
                    delta = donor.flatten()[start:end].to(device=gradient.device, dtype=torch.float32) - base
                    coefficient_gradient[chunk, donor_index].add_(torch.dot(grad, delta))
        return (coefficient_gradient, None, None, *(None for _ in ctx.donors))


def differentiable_weight_blend(
    anchor: torch.Tensor,
    donors: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
    *,
    block_elements: int,
) -> torch.Tensor:
    """Materialize anchor + sum(alpha * donor_delta) with coefficient gradients.

    Coefficient rows partition flattened weights into contiguous chunks. Source
    weights remain frozen on CPU; only the output and bounded FP32 blocks use
    the coefficient device.
    """
    if coefficients.ndim != 2 or coefficients.shape[1] != len(donors) or not donors:
        raise ValueError("Coefficients must have shape (chunks, donors)")
    if not 0 < coefficients.shape[0] <= anchor.numel() or block_elements <= 0:
        raise ValueError("Provide nonempty chunks and a positive block size")
    for tensor in (anchor, *donors):
        if tensor.requires_grad or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Source weights must be frozen contiguous CPU tensors")
        if tensor.shape != anchor.shape or tensor.dtype != anchor.dtype:
            raise ValueError("Source weights must have matching shapes and dtypes")
    return _FrozenWeightBlend.apply(coefficients, anchor, block_elements, *donors)
