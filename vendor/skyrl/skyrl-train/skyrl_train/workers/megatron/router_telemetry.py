"""Grug router telemetry: per-layer load statistics, collision mass, and R1 bias guards.

Load statistics accumulate from :class:`RouterObservation` emissions during training
forwards; collision mass is the per-expert routed-gradient norm of one mini-batch,
attributed to the batch's task tag. The bias guard snapshots Grug's ``expert_bias``
buffers and asserts they stay byte-identical across optimizer steps.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from skyrl_train.models.router_instrumentation import RouterObservation

ROUTER_LOAD_METRIC_PREFIX = "router/load"
ROUTER_COLLISION_METRIC_PREFIX = "router/collision"
MIXED_TASK_TAG = "mixed"
_EXPERT_NAME = re.compile(r"experts\.expert(\d+)\.")


def layer_index(router: nn.Module) -> int | None:
    """The transformer layer a router belongs to, or None outside a decoder layer."""
    layer_number = getattr(router, "layer_number", None)
    if layer_number is None:
        return None
    # MCore layer numbers are 1-based across the whole model; the router's own
    # number identifies its layer.
    return int(layer_number) - 1


class RouterLoadStats:
    """Accumulate per-(layer, expert) selection counts across router forwards."""

    def __init__(self):
        self._counts: dict[int, torch.Tensor] = {}
        self._num_experts: int | None = None

    def observe(self, observation: RouterObservation) -> None:
        selected = observation.selected_experts
        if selected.ndim != 2:
            raise ValueError(f"router observation selections must be [tokens, k]; got shape {tuple(selected.shape)}")
        num_experts = int(observation.selection_logits.shape[-1])
        if self._num_experts is None:
            self._num_experts = num_experts
        elif self._num_experts != num_experts:
            raise ValueError("router observations disagree on the expert count")
        layer = layer_index(observation.router)
        counts = self._counts.setdefault(layer if layer is not None else -1, torch.zeros(num_experts, dtype=torch.long))
        counts += torch.bincount(selected.detach().reshape(-1).to(counts.device), minlength=num_experts)[:num_experts]

    def counts(self) -> dict[int, torch.Tensor]:
        """The internal per-layer count tensors, for cross-rank gathering."""
        return dict(self._counts)

    def merge_counts(self, partials: list[dict[int, torch.Tensor]]) -> None:
        """Sum gathered per-rank counts into these totals."""
        for partial in partials:
            for layer, layer_counts in partial.items():
                if self._num_experts is None:
                    self._num_experts = int(layer_counts.shape[-1])
                existing = self._counts.get(layer)
                self._counts[layer] = layer_counts.clone() if existing is None else existing + layer_counts

    def metrics(self) -> dict[str, float]:
        """Per-layer load summary keyed for the tracker: entropy, max/mean, unhealthy experts."""
        if self._num_experts is None:
            return {}
        metrics: dict[str, float] = {}
        for layer, counts in sorted(self._counts.items()):
            total = int(counts.sum())
            if total == 0:
                continue
            prefix = f"{ROUTER_LOAD_METRIC_PREFIX}/layer{layer}"
            load = counts.double() / total
            metrics[f"{prefix}/entropy"] = float(-(load * load.clamp_min(1e-12).log()).sum())
            metrics[f"{prefix}/max_over_mean"] = float(counts.max()) / (total / self._num_experts)
            metrics[f"{prefix}/min_load"] = float(load.min())
            metrics[f"{prefix}/experts_below_10pct_of_mean"] = float((load < 0.1).sum())
        return metrics


def routed_expert_gradient_norms(model: nn.Module) -> dict[int, float]:
    """Per-expert routed-gradient L2 norm for one accumulated mini-batch.

    Experts are the ``experts.expert<i>.`` parameters of each MoE layer (MCore's
    per-expert modules); gradients come from ``main_grad`` (Megatron DDP gradient
    buffers) or ``grad``. The returned norms are local to this rank's expert shard.
    """
    squares: dict[int, float] = {}
    for name, parameter in model.named_parameters():
        match = _EXPERT_NAME.search(name)
        if match is None:
            if ".experts." in name:
                raise ValueError(f"routed-expert parameter {name} carries no expert index; extend the telemetry parser")
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is not None:
            expert = int(match.group(1))
            squares[expert] = squares.get(expert, 0.0) + float(gradient.detach().float().norm() ** 2)
    return {expert: math.sqrt(total) for expert, total in squares.items()}


def collision_metrics(norms: Mapping[int, float], task_tags: list[str | None]) -> dict[str, float]:
    """Attribute per-expert gradient norms to the mini-batch's task tag.

    Single-tag batches attribute directly; a mixed batch has no per-task gradient
    split, so its norms log under ``mixed`` alongside the per-tag row shares.
    """
    if not norms:
        return {}
    tags = {tag for tag in task_tags if tag is not None}
    attribution = next(iter(tags)) if len(tags) == 1 else MIXED_TASK_TAG
    metrics: dict[str, float] = {}
    for expert, norm in sorted(norms.items()):
        metrics[f"{ROUTER_COLLISION_METRIC_PREFIX}/{attribution}/expert{expert}"] = norm
    for tag in sorted(tags):
        known = [value for value in task_tags if value is not None]
        metrics[f"{ROUTER_COLLISION_METRIC_PREFIX}/task_share/{tag}"] = sum(1 for value in known if value == tag) / max(
            len(known), 1
        )
    return metrics


def named_expert_bias_buffers(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Grug's per-layer ``expert_bias`` buffers, the frozen half of router level R1."""
    return [
        (name, buffer) for name, buffer in model.named_buffers(remove_duplicate=False) if name.endswith("expert_bias")
    ]


def snapshot_expert_bias(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the expert-bias buffers before training starts."""
    return {name: buffer.detach().clone() for name, buffer in named_expert_bias_buffers(model)}


def assert_bias_unchanged(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Raise when a bias buffer drifted from its snapshot after an optimizer step."""
    current = dict(named_expert_bias_buffers(model))
    if set(current) != set(snapshot):
        raise ValueError("expert_bias buffer set changed during training; the frozen selection bias is violated")
    for name, buffer in current.items():
        if not torch.equal(buffer.detach(), snapshot[name].to(buffer.device)):
            raise ValueError(f"expert_bias buffer {name} changed during training; the selection bias must stay frozen")


def router_telemetry_enabled(config) -> bool:
    """Whether training collects router load statistics and collision mass."""
    return bool(OmegaConf.select(config, "trainer.router_telemetry.enabled") or False)


def merge_router_telemetry_metrics(
    actor_module: list[nn.Module],
    load_stats: RouterLoadStats,
    data_source_ids: list,
    data_source_names: list[str] | None,
    metrics_list: list[dict],
) -> None:
    """Fold this step's router load and collision metrics into the micro-batch metrics.

    Pipeline stages hold different layers and expert-parallel ranks hold different
    experts, so counts and norms are gathered across both groups; the last pipeline
    stage publishes the merged step metrics, matching the replay-metric channel.
    """
    import megatron.core.parallel_state as mpu  # guarded: telemetry runs only inside the Megatron worker

    if not metrics_list:
        return
    pipeline_group = mpu.get_pipeline_model_parallel_group()
    gathered_counts: list[dict | None] = [None] * torch.distributed.get_world_size(pipeline_group)
    torch.distributed.all_gather_object(gathered_counts, load_stats.counts(), group=pipeline_group)

    expert_group = mpu.get_expert_model_parallel_group()
    norms = routed_expert_gradient_norms(actor_module[0])
    gathered_norms: list[dict | None] = [None] * torch.distributed.get_world_size(expert_group)
    torch.distributed.all_gather_object(gathered_norms, norms, group=expert_group)

    if not mpu.is_pipeline_last_stage(ignore_virtual=True):
        return
    merged = RouterLoadStats()
    merged.merge_counts([counts for counts in gathered_counts if counts])
    metrics = merged.metrics()
    merged_norms: dict[int, float] = {}
    for shard_norms in gathered_norms:
        for expert, norm in (shard_norms or {}).items():
            merged_norms[expert] = max(merged_norms.get(expert, 0.0), norm)
    if data_source_names:
        task_ids = [int(index) for ids in data_source_ids if ids is not None for index in ids.tolist()]
        tags = [data_source_names[index] if index >= 0 else None for index in task_ids]
        metrics.update(collision_metrics(merged_norms, tags))
    metrics_list[-1].update(metrics)


def router_policy_manifest(model_type: str) -> dict[str, Any]:
    """The router policy record: weights trained, bias frozen, no aux loss."""
    return {
        "model_type": model_type,
        "router_weights": "trained",
        "selection_bias": "frozen",
        "aux_loss_coeff": 0.0,
    }
