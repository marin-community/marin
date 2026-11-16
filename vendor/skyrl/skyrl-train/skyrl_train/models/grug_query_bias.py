# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Query-bias observations and optimizer-window reduction for Grug MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class GrugQueryBiasConfig(Protocol):
    num_experts_per_tok: int
    num_local_experts: int
    num_hidden_layers: int


class GrugQueryBiasCaptureModel(Protocol):
    config: GrugQueryBiasConfig

    def begin_query_bias_capture(self, candidate_count: int, token_mask: torch.Tensor) -> None: ...

    def take_query_bias_observation(self, *, candidate_count: int) -> "GrugQueryBiasObservation": ...

    def begin_loss_free_bias_capture(self, token_mask: torch.Tensor) -> None: ...

    def take_loss_free_bias_observation(self) -> "GrugQueryBiasObservation": ...

    def get_query_bias(self) -> torch.Tensor: ...

    def set_query_bias(self, bias: torch.Tensor) -> None: ...


def query_bias_candidate_count(valid_tokens: int, experts_per_token: int, num_experts: int) -> int:
    """Return Grug's per-expert candidate count for one optimizer window."""

    if valid_tokens < 1:
        raise ValueError(f"valid_tokens must be positive, got {valid_tokens}")
    return max(1, valid_tokens * experts_per_token // num_experts)


@dataclass(frozen=True)
class GrugQueryBiasLayerObservation:
    """Compact routing data from one layer and one microbatch.

    ``candidates`` contains the largest ``q`` values of ``s - alpha`` for each
    expert during Quantile Balancing. Keeping only these values is sufficient
    to recover the exact q-th value across a gradient-accumulation window. It
    is empty when only selected-expert loads are captured.
    """

    candidates: torch.Tensor
    selected_experts: torch.Tensor
    combine_weights: torch.Tensor


@dataclass(frozen=True)
class GrugQueryBiasObservation:
    """Routing observations; candidate_count is zero for load-only capture."""

    layers: tuple[GrugQueryBiasLayerObservation, ...]
    candidate_count: int


@dataclass(frozen=True)
class GrugQueryBiasShardLayout:
    """Virtual EP ownership within one optimizer window."""

    micro_batch_size: int
    accumulation_steps: int
    ep_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        if self.micro_batch_size < 1 or self.accumulation_steps < 1:
            raise ValueError(
                "micro_batch_size and accumulation_steps must be positive, "
                f"got {self.micro_batch_size}, {self.accumulation_steps}"
            )
        if self.ep_size < 1 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"invalid EP coordinate ep_rank={self.ep_rank}, ep_size={self.ep_size}")
        rows_per_window = self.micro_batch_size * self.accumulation_steps
        if rows_per_window % self.ep_size:
            raise ValueError(
                "Grug query-bias virtual shards require "
                f"micro_batch_size*accumulation_steps={rows_per_window} divisible by ep_size={self.ep_size}"
            )

    def mask_for(self, attention_mask: torch.Tensor, local_step: int) -> torch.Tensor:
        """Select this EP coordinate's rows from one microbatch."""

        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be 2-D, got shape={tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != self.micro_batch_size:
            raise RuntimeError(
                "the final policy optimizer window is incomplete: "
                f"expected microbatch rows={self.micro_batch_size}, got {attention_mask.shape[0]}"
            )

        rows_per_shard = self.micro_batch_size * self.accumulation_steps // self.ep_size
        shard_start = self.ep_rank * rows_per_shard
        shard_end = shard_start + rows_per_shard
        microbatch_start = (local_step % self.accumulation_steps) * self.micro_batch_size
        row_offsets = torch.arange(self.micro_batch_size, device=attention_mask.device) + microbatch_start
        owned_rows = (row_offsets >= shard_start) & (row_offsets < shard_end)
        return attention_mask.bool() & owned_rows.unsqueeze(1)


@dataclass(frozen=True)
class GrugQueryBiasCapturePlan:
    """Valid-token counts and virtual ownership for one policy batch."""

    valid_token_counts: tuple[int, ...]
    shard_layout: GrugQueryBiasShardLayout

    @classmethod
    def build(
        cls,
        attention_mask: torch.Tensor,
        shard_layout: GrugQueryBiasShardLayout,
    ) -> GrugQueryBiasCapturePlan:
        valid_token_counts = tuple(
            int(shard_layout.mask_for(mask, local_step).sum().item())
            for local_step, mask in enumerate(attention_mask.split(shard_layout.micro_batch_size))
        )
        remainder = len(valid_token_counts) % shard_layout.accumulation_steps
        if remainder:
            raise RuntimeError(
                "the final policy optimizer window is incomplete: "
                f"expected {shard_layout.accumulation_steps} microbatches, got {remainder}"
            )
        return cls(valid_token_counts, shard_layout)


class GrugQueryBiasAccumulator:
    """Retain exact per-expert top-q candidates across one optimizer window."""

    def __init__(self, *, candidate_count: int, num_layers: int, num_experts: int) -> None:
        if candidate_count < 1:
            raise ValueError(f"candidate_count must be positive, got {candidate_count}")
        self.candidate_count = candidate_count
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._candidates: list[torch.Tensor | None] = [None] * num_layers

    def observe(self, observation: GrugQueryBiasObservation) -> None:
        if observation.candidate_count != self.candidate_count:
            raise ValueError(
                f"observation q={observation.candidate_count} does not match window q={self.candidate_count}"
            )
        if len(observation.layers) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer observations, got {len(observation.layers)}")

        for layer_idx, layer in enumerate(observation.layers):
            candidates = layer.candidates.float()
            if candidates.ndim != 2 or candidates.shape[0] != self.num_experts:
                raise ValueError(
                    f"layer {layer_idx} candidates must have shape [{self.num_experts}, q], "
                    f"got {tuple(candidates.shape)}"
                )
            previous = self._candidates[layer_idx]
            combined = candidates if previous is None else torch.cat((previous, candidates), dim=-1)
            keep = min(self.candidate_count, combined.shape[-1])
            self._candidates[layer_idx] = torch.topk(combined, k=keep, dim=-1, sorted=True).values

    @torch.no_grad()
    def finalize_betas(self) -> torch.Tensor:
        """Return ``[layers, experts]`` local q-th values, averaged across ranks."""

        if any(candidates is None for candidates in self._candidates):
            raise RuntimeError("cannot finalize query bias before observing every layer")

        betas = []
        for layer_idx, candidates in enumerate(self._candidates):
            assert candidates is not None
            if candidates.shape[-1] < self.candidate_count:
                raise RuntimeError(
                    f"layer {layer_idx} saw only {candidates.shape[-1]} candidates for q={self.candidate_count}"
                )
            betas.append(candidates[:, self.candidate_count - 1])
        result = torch.stack(betas, dim=0).float()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            result.div_(world_size)
        return result


class GrugLossFreeBiasAccumulator:
    """Accumulate per-layer expert assignments for a policy optimizer window."""

    def __init__(self, *, num_layers: int, num_experts: int) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._loads: list[torch.Tensor | None] = [None] * num_layers

    def observe(self, observation: GrugQueryBiasObservation) -> None:
        if len(observation.layers) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer observations, got {len(observation.layers)}")

        for layer_idx, layer in enumerate(observation.layers):
            loads = torch.bincount(layer.selected_experts.reshape(-1), minlength=self.num_experts).float()
            if loads.shape[0] != self.num_experts:
                raise ValueError(f"layer {layer_idx} selected an expert outside [0, {self.num_experts})")
            previous = self._loads[layer_idx]
            self._loads[layer_idx] = loads if previous is None else previous + loads

    @torch.no_grad()
    def finalize_loads(self) -> torch.Tensor:
        """Return global assignment counts with shape ``[layers, experts]``."""

        if any(loads is None for loads in self._loads):
            raise RuntimeError("cannot finalize loss-free bias before observing every layer")
        result = torch.stack([loads for loads in self._loads if loads is not None], dim=0).float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result


def next_query_bias(betas: torch.Tensor) -> torch.Tensor:
    """Apply Grug's ``bias = -beta`` update and center each layer."""

    if betas.ndim != 2:
        raise ValueError(f"betas must have shape [layers, experts], got {tuple(betas.shape)}")
    bias = -betas.float()
    return bias - bias.mean(dim=-1, keepdim=True)


def next_loss_free_query_bias(
    current_bias: torch.Tensor,
    expert_loads: torch.Tensor,
    *,
    update_rate: float,
) -> torch.Tensor:
    """Apply DeepSeek's signed load-error update to an external router bias."""

    if current_bias.ndim != 2 or expert_loads.shape != current_bias.shape:
        raise ValueError(
            "current_bias and expert_loads must have matching [layers, experts] shapes; "
            f"got {tuple(current_bias.shape)} and {tuple(expert_loads.shape)}"
        )
    if not torch.isfinite(expert_loads).all() or (expert_loads < 0).any():
        raise ValueError("expert_loads must be finite and non-negative")
    if not 0.0 < update_rate < float("inf"):
        raise ValueError(f"update_rate must be finite and positive, got {update_rate}")

    mean_load = expert_loads.float().mean(dim=-1, keepdim=True)
    load_error = mean_load - expert_loads.float()
    updated_bias = current_bias.float() + update_rate * torch.sign(load_error)
    return updated_bias - updated_bias.mean(dim=-1, keepdim=True)


class GrugQueryBiasUpdater(Protocol):
    """Update mechanism used by a Grug query-bias optimizer window."""

    def begin_capture(self, model: GrugQueryBiasCaptureModel, token_mask: torch.Tensor) -> None: ...

    def observe(self, model: GrugQueryBiasCaptureModel) -> None: ...

    def apply(self, model: GrugQueryBiasCaptureModel) -> None: ...


class GrugQuantileBiasUpdater:
    """Collect and apply a Quantile Balancing target."""

    def __init__(self, model: GrugQueryBiasCaptureModel, valid_tokens: int, *, target_weight: float) -> None:
        if not 0.0 < target_weight <= 1.0:
            raise ValueError(f"target_weight must be in (0, 1], got {target_weight}")
        config = model.config
        self.target_weight = target_weight
        self.candidate_count = query_bias_candidate_count(
            valid_tokens,
            config.num_experts_per_tok,
            config.num_local_experts,
        )
        self.accumulator = GrugQueryBiasAccumulator(
            candidate_count=self.candidate_count,
            num_layers=config.num_hidden_layers,
            num_experts=config.num_local_experts,
        )

    def begin_capture(self, model: GrugQueryBiasCaptureModel, token_mask: torch.Tensor) -> None:
        model.begin_query_bias_capture(self.candidate_count, token_mask)

    def observe(self, model: GrugQueryBiasCaptureModel) -> None:
        self.accumulator.observe(
            model.take_query_bias_observation(
                candidate_count=self.candidate_count,
            )
        )

    def apply(self, model: GrugQueryBiasCaptureModel) -> None:
        current_bias = model.get_query_bias().float()
        target = next_query_bias(self.accumulator.finalize_betas()).to(current_bias)
        model.set_query_bias(torch.lerp(current_bias, target, self.target_weight))


class GrugLossFreeBiasUpdater:
    """Collect expert loads and apply one DeepSeek loss-free update."""

    def __init__(self, model: GrugQueryBiasCaptureModel, *, update_rate: float) -> None:
        if not 0.0 < update_rate < float("inf"):
            raise ValueError(f"update_rate must be finite and positive, got {update_rate}")
        config = model.config
        self.update_rate = update_rate
        self.accumulator = GrugLossFreeBiasAccumulator(
            num_layers=config.num_hidden_layers,
            num_experts=config.num_local_experts,
        )

    def begin_capture(self, model: GrugQueryBiasCaptureModel, token_mask: torch.Tensor) -> None:
        model.begin_loss_free_bias_capture(token_mask)

    def observe(self, model: GrugQueryBiasCaptureModel) -> None:
        self.accumulator.observe(model.take_loss_free_bias_observation())

    def apply(self, model: GrugQueryBiasCaptureModel) -> None:
        current_bias = model.get_query_bias().float()
        loads = self.accumulator.finalize_loads().to(current_bias)
        model.set_query_bias(next_loss_free_query_bias(current_bias, loads, update_rate=self.update_rate))


class GrugQueryBiasWindow:
    """Run one external query-bias update over an optimizer window."""

    def __init__(
        self,
        model: GrugQueryBiasCaptureModel,
        capture_plan: GrugQueryBiasCapturePlan,
        updater: GrugQueryBiasUpdater,
    ) -> None:
        self.model = model
        self.capture_plan = capture_plan
        self.updater = updater
        self._closed = False

    def begin_microbatch(self, attention_mask: torch.Tensor, local_step: int) -> bool:
        """Start capture and return false when this EP shard owns no valid tokens."""

        if self.capture_plan.valid_token_counts[local_step] == 0:
            return False
        capture_mask = self.capture_plan.shard_layout.mask_for(attention_mask, local_step)
        self.updater.begin_capture(self.model, capture_mask)
        return True

    def observe_microbatch(self) -> None:
        self.updater.observe(self.model)

    def finish(self, *, optimizer_step_succeeded: bool) -> None:
        """Apply a completed window once, or discard it after a skipped step."""

        if self._closed:
            return
        if optimizer_step_succeeded:
            self.updater.apply(self.model)
        self._closed = True
