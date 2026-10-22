# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grug MuonH and AdamH updates for Megatron's FP32 master parameters.

The BF16 Newton--Schulz transform is adapted from NVIDIA NeMo
Emerging-Optimizers at 6ef41445b246d2c64c2e6f82cc56fbc9c9c07937.
"""

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

type MegatronGrugRoute = Literal["grug_muonh", "grug_muonh_qkv", "grug_muonh_gate_up", "grug_adamh", "adam"]
_DIRECTION_SCRATCH_BYTES = 16 * 1024 * 1024

_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_HYPERBALL_EPS = 1e-10


def _muon_direction(matrix: Tensor, *, steps: int, eps: float) -> Tensor:
    """Return Marin's BF16 quintic direction with its matrix shape scale."""
    original_dtype = matrix.dtype
    x = matrix.to(torch.bfloat16)
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for index in range(steps):
        a, b, c = _QUINTIC_COEFFICIENTS[index % len(_QUINTIC_COEFFICIENTS)]
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    # PyTorch Linear matrices are (fan_out, fan_in); Marin stores the transpose.
    rows, columns = matrix.shape[-2:]
    return x.to(original_dtype).mul_(max(1.0, rows / columns) ** 0.5)


def _matrix_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True, dtype=torch.float32).square_().sqrt_()


def _matrix_step_(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    ns_steps: int | None = None,
    muon_eps: float = 1e-8,
    clamp_final_norm: bool,
) -> None:
    """Apply HyperBall to each complete matrix, consuming direction as scratch."""
    if ns_steps is not None:
        direction = _muon_direction(direction, steps=ns_steps, eps=muon_eps)
    parameter_norm = _matrix_norm(parameter)
    direction_norm = _matrix_norm(direction).clamp_min_(_HYPERBALL_EPS)
    direction.mul_(parameter_norm / direction_norm).mul_(-lr).add_(parameter)
    candidate_norm = _matrix_norm(direction)
    if clamp_final_norm:
        candidate_norm.clamp_min_(_HYPERBALL_EPS)
    direction.mul_(parameter_norm / candidate_norm)
    parameter.copy_(direction)


def megatron_grug_route(name: str, parameter: Tensor) -> MegatronGrugRoute:
    """Classify Megatron's fused parameter names using the Grug recipe."""
    lower = name.lower()
    if "gated_norm" in lower:
        return "grug_muonh"
    if lower.endswith((".down_proj.weight", ".up_proj.weight")) and any(
        f"{norm}." in lower for norm in ("embed_norm", "final_layernorm", "input_layernorm", "pre_mlp_layernorm")
    ):
        return "grug_muonh"
    if "output_layer.weight" in lower or "output_proj" in lower or "lm_head" in lower:
        return "grug_adamh"
    if (
        "embed" in lower
        or "sconv" in lower
        or "attn_gate" in lower
        or ".router." in lower
        or lower.startswith("router.")
    ):
        return "adam"
    if lower.endswith("linear_qkv.weight"):
        return "grug_muonh_qkv"
    if "linear_fc1.weight" in lower and (".experts." in lower or ".shared_experts." in lower):
        return "grug_muonh_gate_up"
    return "grug_muonh" if parameter.ndim in (2, 3) else "adam"


def _muon_update_(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    ns_steps: int,
    eps: float,
    route: MegatronGrugRoute,
    qkv_split_shapes: tuple[int, int, int],
) -> None:
    """Update complete logical matrices, gathering Q/K/V across all KV groups.

    A fused QKV tensor holds interleaved groups, but each Q, K or V projection
    is one matrix for Newton--Schulz and HyperBall. Gate/up halves are separate
    matrices. Parameters are updated in place; direction may be consumed.
    """
    if route == "grug_muonh_qkv":
        if parameter.ndim != 2 or parameter.shape[0] % sum(qkv_split_shapes):
            raise ValueError(f"Unexpected fused QKV shape: {tuple(parameter.shape)}")
        groups = parameter.shape[0] // sum(qkv_split_shapes)
        parameter_view = parameter.view(groups, sum(qkv_split_shapes), parameter.shape[1])
        direction_view = direction.view_as(parameter_view)
        for parameter_part, direction_part in zip(
            parameter_view.split(qkv_split_shapes, dim=1), direction_view.split(qkv_split_shapes, dim=1)
        ):
            # A single QKV group can leave the split contiguous, so contiguous()
            # would alias parameter_part and make the write-back overlap.
            logical_parameter = parameter_part.reshape(-1, parameter.shape[1]).clone()
            logical_direction = direction_part.reshape_as(logical_parameter).contiguous()
            _matrix_step_(
                logical_parameter, logical_direction, lr=lr, ns_steps=ns_steps, muon_eps=eps, clamp_final_norm=True
            )
            parameter_part.copy_(logical_parameter.view_as(parameter_part))
        return

    if route == "grug_muonh_gate_up":
        if parameter.ndim not in (2, 3) or parameter.shape[-2] % 2:
            raise ValueError(f"Unexpected fused gate/up shape: {tuple(parameter.shape)}")
        for parameter_part, direction_part in zip(parameter.chunk(2, dim=-2), direction.chunk(2, dim=-2)):
            _matrix_step_(parameter_part, direction_part, lr=lr, ns_steps=ns_steps, muon_eps=eps, clamp_final_norm=True)
        return

    _matrix_step_(parameter, direction, lr=lr, ns_steps=ns_steps, muon_eps=eps, clamp_final_norm=True)


def _adamh_direction_in_grad_(
    gradient: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor, *, step: int, betas: tuple[float, float], eps: float
) -> None:
    """Reuse the consumed FP32 gradient for AdamH's direction, with bounded scratch."""
    beta1, beta2 = betas
    bias1 = 1 - beta1**step
    bias2 = 1 - beta2**step
    row_bytes = gradient[0].numel() * gradient.element_size()
    rows_per_chunk = max(1, _DIRECTION_SCRATCH_BYTES // row_bytes)
    for start in range(0, gradient.shape[0], rows_per_chunk):
        end = start + rows_per_chunk
        direction_chunk = gradient[start:end]
        direction_chunk.copy_(exp_avg[start:end]).div_(bias1)
        denominator = exp_avg_sq[start:end].clone().div_(bias2).sqrt_().add_(eps)
        direction_chunk.div_(denominator)
        del denominator


def _offloaded_muon_direction_in_grad_(gradient: Tensor, momentum: Tensor, *, beta: float, nesterov: bool) -> None:
    """Update CPU momentum in bounded GPU chunks, leaving the direction in the consumed gradient."""
    if not gradient.is_contiguous():
        raise ValueError("Offloaded MuonH requires contiguous FP32 master gradients")
    gradient_flat = gradient.view(-1)
    momentum_flat = momentum.view(-1)
    elements_per_chunk = max(1, _DIRECTION_SCRATCH_BYTES // gradient.element_size())
    for start in range(0, gradient_flat.numel(), elements_per_chunk):
        gradient_chunk = gradient_flat[start : start + elements_per_chunk]
        momentum_chunk = momentum_flat[start : start + elements_per_chunk]
        scratch = torch.empty_like(gradient_chunk)
        scratch.copy_(momentum_chunk)
        scratch.mul_(beta).add_(gradient_chunk)
        momentum_chunk.copy_(scratch)
        if nesterov:
            gradient_chunk.add_(scratch, alpha=beta)
        else:
            gradient_chunk.copy_(scratch)
        del scratch


class MegatronGrugMuonH(Optimizer):
    """Flat optimizer state for Megatron's mixed-precision and sharded checkpoint wrappers.

    Megatron creates a separate instance per optimizer route and expert-parallel
    group. Its standard Adam path owns the remaining parameters.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        eps: float,
        muon_eps: float,
        qkv_split_shapes: tuple[int, int, int],
        offload_momentum: bool = False,
    ) -> None:
        self.qkv_split_shapes = qkv_split_shapes
        self.offload_momentum = offload_momentum
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "eps": eps,
                "muon_eps": muon_eps,
                "weight_decay": 0.0,
            },
        )

    def initialize_state(self) -> None:
        """Materialize all moments before Megatron builds a checkpoint load template."""
        for group in self.param_groups:
            route = group.get("optimizer", "grug_muonh")
            for parameter in group["params"]:
                state = self.state[parameter]
                if route == "grug_adamh":
                    if "step" not in state:
                        state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                else:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = (
                            torch.zeros(
                                parameter.shape, dtype=parameter.dtype, device="cpu", pin_memory=parameter.is_cuda
                            )
                            if self.offload_momentum
                            else torch.zeros_like(parameter)
                        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.offload_momentum:
            super().load_state_dict(state_dict)
            return
        # PyTorch moves per-parameter state to the parameter's GPU on load.
        # Keep Muon moments on CPU without ever materializing that GPU copy.
        saved_momenta = {}
        stripped_state = {}
        for parameter_id, values in state_dict["state"].items():
            if "momentum_buffer" in values:
                saved_momenta[parameter_id] = values["momentum_buffer"]
                stripped_state[parameter_id] = {key: value for key, value in values.items() if key != "momentum_buffer"}
            else:
                stripped_state[parameter_id] = values
        existing = {parameter: values.get("momentum_buffer") for parameter, values in self.state.items()}
        super().load_state_dict({**state_dict, "state": stripped_state})
        for saved_group, current_group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for parameter_id, parameter in zip(saved_group["params"], current_group["params"], strict=True):
                if parameter_id not in saved_momenta:
                    continue
                saved = saved_momenta[parameter_id]
                buffer = existing.get(parameter)
                if buffer is None or buffer.shape != saved.shape:
                    buffer = torch.empty(saved.shape, dtype=parameter.dtype, device="cpu", pin_memory=parameter.is_cuda)
                buffer.copy_(saved)
                self.state[parameter]["momentum_buffer"] = buffer

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            route = group.get("optimizer", "grug_muonh")
            if route not in ("grug_muonh", "grug_muonh_qkv", "grug_muonh_gate_up", "grug_adamh"):
                raise ValueError(f"Unsupported Grug Megatron optimizer route: {route}")
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Grug MuonH/AdamH does not support sparse gradients")
                if parameter.ndim not in (2, 3):
                    raise RuntimeError(f"Grug MuonH/AdamH received rank-{parameter.ndim} parameter")
                state = self.state[parameter]
                if not state:
                    self.initialize_state()

                if route == "grug_adamh":
                    beta1, beta2 = group["betas"]
                    state["step"].add_(1)
                    state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                    step = int(state["step"].item())
                    _adamh_direction_in_grad_(
                        gradient,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step=step,
                        betas=(beta1, beta2),
                        eps=group["eps"],
                    )
                    _matrix_step_(parameter, gradient, lr=group["lr"], clamp_final_norm=False)
                    continue

                momentum_buffer = state["momentum_buffer"]
                if momentum_buffer.device.type == "cpu" and gradient.is_cuda:
                    _offloaded_muon_direction_in_grad_(
                        gradient, momentum_buffer, beta=group["momentum"], nesterov=group["nesterov"]
                    )
                    direction = gradient
                else:
                    momentum_buffer.mul_(group["momentum"]).add_(gradient)
                    direction = (
                        gradient.add_(momentum_buffer, alpha=group["momentum"])
                        if group["nesterov"]
                        else momentum_buffer
                    )
                _muon_update_(
                    parameter,
                    direction,
                    lr=group["lr"],
                    ns_steps=group["ns_steps"],
                    eps=group["muon_eps"],
                    route=route,
                    qkv_split_shapes=self.qkv_split_shapes,
                )
        return loss
