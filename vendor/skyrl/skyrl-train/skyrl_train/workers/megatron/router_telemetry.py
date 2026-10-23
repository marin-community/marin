"""Grug router telemetry: per-layer load statistics, collision mass, and R1 bias guards.

Load statistics accumulate from :class:`RouterObservation` emissions during
training forwards. Only response-token rows count (prompt and pad rows are
masked out) and only original forwards count (activation-checkpointing
recomputes a forward during backward; those emissions are dropped). Counts are
reduced across data, context, tensor and expert-parallel ranks so every
pipeline stage's summary describes the same global mini-batch.

Collision mass is the per-expert routed-gradient norm of one mini-batch,
attributed to the batch's task tags. Gradients are read from this rank's own
gradient shards at micro-batch boundaries (before any reduce-scatter touches
them) via post-accumulation hooks, split by task at those boundaries, and
combined by sum of squares with explicit collectives over the tensor, expert
and data-parallel groups. Keys are always (layer, global expert); nothing is
ever summed across layers.

The bias guard snapshots Grug's ``expert_bias`` buffers and asserts they stay
byte-identical across optimizer steps.

Per-step vectors land in a JSONL sidecar next to the exports:
``<export_path>/router_telemetry.jsonl`` holds one record per step with the
full per-(layer, expert) load counts and per-task collision norms, which the
load-shift JSD analysis reads.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from skyrl_train.io import io
from skyrl_train.models.router_instrumentation import RouterObservation

ROUTER_LOAD_METRIC_PREFIX = "router/load"
ROUTER_COLLISION_METRIC_PREFIX = "router/collision"
ROUTER_TELEMETRY_SIDECAR_FILENAME = "router_telemetry.jsonl"

# Real Megatron parameter names for routed experts. TEGroupedMLP (Grug's
# grouped-GEMM path) names experts by an index suffix on the fused weights;
# SequentialMLP nests per-expert modules instead.
_EXPERT_INDEX_PATTERNS = (
    re.compile(r"linear_fc1\.weight(\d+)$"),
    re.compile(r"linear_fc2\.weight(\d+)$"),
    re.compile(r"experts\.expert(\d+)\."),
    re.compile(r"experts\.local_experts\.(\d+)\."),
)


def local_expert_index(parameter_name: str) -> int | None:
    """The EP-local expert index carried by a Megatron expert parameter name."""
    for pattern in _EXPERT_INDEX_PATTERNS:
        match = pattern.search(parameter_name)
        if match is not None:
            return int(match.group(1))
    return None


def build_router_layer_map(model: nn.Module) -> dict[int, int]:
    """Map ``id(router module)`` to its 0-based global transformer layer.

    mcore's ``TransformerLayer.layer_number`` is 1-based across the whole model
    (virtual-pipeline offsets included), so it is the one layer identity that
    agrees across pipeline stages and model chunks.
    """
    layer_map: dict[int, int] = {}
    for module in model.modules():
        router = getattr(module, "router", None)
        layer_number = getattr(module, "layer_number", None)
        if isinstance(router, nn.Module) and layer_number is not None:
            layer_map[id(router)] = int(layer_number) - 1
    return layer_map


def layer_index(router: nn.Module, layer_map: Mapping[int, int]) -> int:
    """The global 0-based layer of an observed router; unknown routers raise."""
    layer = layer_map.get(id(router))
    if layer is None:
        raise ValueError("router telemetry observed a router absent from the layer map; extend build_router_layer_map")
    return layer


def should_count_router_forward(*, grad_enabled: bool, gradient_checkpointing: bool) -> bool:
    """Whether a router emission is an original forward rather than a recompute.

    Activation-checkpointed training forwards run under ``no_grad`` and their
    recomputes run with grad enabled; without checkpointing the original
    forward is the only emission and runs with grad enabled.
    """
    return grad_enabled != gradient_checkpointing


class RoutedExpertParameter:
    """One routed-expert weight with its global layer and EP-local expert index."""

    __slots__ = ("layer", "local_expert", "parameter")

    def __init__(self, layer: int, local_expert: int, parameter: torch.nn.Parameter):
        self.layer = layer
        self.local_expert = local_expert
        self.parameter = parameter


def routed_expert_parameters(model: nn.Module) -> list[RoutedExpertParameter]:
    """Every routed-expert parameter under the model chunks, with layer and index.

    Experts live under ``TransformerLayer.mlp.experts``; the layer comes from
    the owning ``TransformerLayer.layer_number`` (1-based, global across
    pipeline stages), and the expert index from the Megatron parameter name
    (EP-local). Parameters under ``mlp.experts`` whose name carries no expert
    index raise rather than being silently skipped.
    """
    rows: list[RoutedExpertParameter] = []
    for module in model.modules():
        layer_number = getattr(module, "layer_number", None)
        mlp = getattr(module, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if layer_number is None or not isinstance(experts, nn.Module):
            continue
        layer = int(layer_number) - 1
        for name, parameter in experts.named_parameters():
            local_expert = local_expert_index(name)
            if local_expert is None:
                raise ValueError(f"routed-expert parameter {name} carries no expert index; extend the telemetry parser")
            rows.append(RoutedExpertParameter(layer, local_expert, parameter))
    if not rows:
        raise ValueError("router telemetry found no routed-expert parameters under the model")
    return rows


class RouterLoadStats:
    """Accumulate per-(layer, global expert) selection counts across router forwards.

    Rows are filtered to the caller's response-token mask before counting; the
    expert axis is global (Grug routers select among all experts), so EP ranks
    need no index offset here.
    """

    def __init__(self, layer_map: Mapping[int, int]):
        self.layer_map = dict(layer_map)
        self.layers = sorted(set(self.layer_map.values()))
        self._counts = {layer: torch.zeros(1, dtype=torch.float64) for layer in self.layers}
        self._num_experts: int | None = None

    def observe(self, observation: RouterObservation, row_mask: torch.Tensor | None = None) -> None:
        selected = observation.selected_experts.detach()
        if selected.ndim != 2:
            raise ValueError(f"router observation selections must be [tokens, k]; got shape {tuple(selected.shape)}")
        if row_mask is not None:
            if row_mask.shape != (selected.shape[0],):
                raise ValueError(
                    f"router response mask has {tuple(row_mask.shape)} rows for {selected.shape[0]} selections"
                )
            selected = selected[row_mask]
        num_experts = int(observation.selection_logits.shape[-1])
        if self._num_experts is None:
            self._num_experts = num_experts
            self._counts = {layer: torch.zeros(num_experts, dtype=torch.float64) for layer in self.layers}
        elif self._num_experts != num_experts:
            raise ValueError("router observations disagree on the expert count")
        layer = layer_index(observation.router, self.layer_map)
        if selected.numel() == 0:
            return
        self._counts[layer] += torch.bincount(selected.reshape(-1).cpu(), minlength=num_experts).double()

    def dense_counts(self) -> torch.Tensor:
        """The local counts as a float64 [len(layers), experts] tensor."""
        if self._num_experts is None:
            raise ValueError("router load statistics observed no forwards")
        return torch.stack([self._counts[layer] for layer in self.layers])

    def metrics(self, counts: torch.Tensor) -> dict[str, float]:
        """Per-layer load summary for the given (reduced) counts tensor."""
        return load_metrics(self.layers, counts)


def load_metrics(layers: list[int], counts: torch.Tensor) -> dict[str, float]:
    """Per-layer load summary; row ``i`` of ``counts`` belongs to ``layers[i]``."""
    num_experts = counts.shape[-1]
    metrics: dict[str, float] = {}
    for position, layer in enumerate(layers):
        layer_counts = counts[position]
        total = float(layer_counts.sum())
        if total == 0:
            continue
        prefix = f"{ROUTER_LOAD_METRIC_PREFIX}/layer{layer}"
        load = layer_counts / total
        metrics[f"{prefix}/entropy"] = float(-(load * load.clamp_min(1e-12).log()).sum())
        metrics[f"{prefix}/max_over_mean"] = float(layer_counts.max()) / (total / num_experts)
        metrics[f"{prefix}/min_load"] = float(load.min())
        # The Stage 0 routing-health gate counts experts below 10% of the MEAN load.
        metrics[f"{prefix}/experts_below_10pct_of_mean"] = float(
            (layer_counts < 0.1 * total / num_experts).sum()
        )
    return metrics


class RouterCollisionTracker:
    """Per-(layer, global expert) routed-gradient mass, split by task per micro-batch.

    Post-accumulation hooks fire once per expert parameter per micro-batch
    backward, right after mcore's DDP hook added the micro-batch gradient into
    ``main_grad`` — before any reduce-scatter, so the read is this rank's own
    gradient shard, never a mixed reduced/unreduced buffer. Each hook adds the
    change in ``||main_grad||²`` since the previous boundary to the current
    micro-batch's wave; a completed wave (every hooked expert parameter fired)
    is attributed to the wave's task mix in proportion to response-token
    counts, so a mixed θ_XR batch contributes per-task rows instead of
    collapsing into one "mixed" bucket.

    With ``overlap_grad_reduce`` the last micro-batch's reduce-scatter can land
    while its hooks still read, so that micro-batch's split carries the
    collective's numeric noise; earlier micro-batches are exact local partials.
    """

    def __init__(self, model: nn.Module, *, expert_offset: int, num_global_experts: int):
        self.num_global_experts = num_global_experts
        self._expert_offset = expert_offset
        self._key_of_parameter: dict[int, tuple[int, int]] = {}
        for row in routed_expert_parameters(model):
            global_expert = expert_offset + row.local_expert
            if not 0 <= global_expert < num_global_experts:
                raise ValueError(
                    f"expert offset {expert_offset} maps local expert {row.local_expert} outside "
                    f"the {num_global_experts} global experts"
                )
            self._key_of_parameter[id(row.parameter)] = (row.layer, global_expert)
        self._baseline: dict[int, float] = {}
        self._wave_sos: dict[tuple[int, int], float] = {}
        self._fired: set[int] = set()
        self._pending_task_tokens: list[dict[str, int]] = []
        self._task_sos: dict[tuple[str, int, int], float] = {}
        self._waves_flushed = 0
        self._hook_holders: list[Any] = []
        self._handles: list[Any] = []
        for row in routed_expert_parameters(model):
            parameter = row.parameter
            if not parameter.requires_grad:
                continue
            expanded = parameter.expand_as(parameter)
            grad_acc = expanded.grad_fn.next_functions[0][0]

            # Default-arg capture pins this hook to its own parameter.
            def hook(*_unused, parameter=parameter):
                self._on_gradient_accumulated(parameter)

            self._handles.append(grad_acc.register_hook(hook))
            self._hook_holders.append(grad_acc)

    def begin_micro_batch(self, task_token_counts: Mapping[str, int]) -> None:
        """Queue one micro-batch's task mix (task -> response tokens) for its backward wave."""
        self._pending_task_tokens.append({tag: int(tokens) for tag, tokens in task_token_counts.items()})

    def _on_gradient_accumulated(self, parameter: torch.nn.Parameter) -> None:
        key = self._key_of_parameter.get(id(parameter))
        if key is None:
            return
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            raise ValueError("router collision telemetry requires mcore DDP main_grad buffers on expert parameters")
        sum_of_squares = float(gradient.detach().float().square().sum())
        delta = max(sum_of_squares - self._baseline.get(id(parameter), 0.0), 0.0)
        self._baseline[id(parameter)] = sum_of_squares
        self._wave_sos[key] = self._wave_sos.get(key, 0.0) + delta
        self._fired.add(id(parameter))
        if len(self._fired) == len(self._hook_holders):
            self._flush_wave()

    def _flush_wave(self) -> None:
        if not self._pending_task_tokens:
            raise ValueError("router collision telemetry recorded a gradient wave with no queued micro-batch task mix")
        task_tokens = self._pending_task_tokens.pop(0)
        total_tokens = sum(task_tokens.values())
        if total_tokens == 0:
            raise ValueError("micro-batch task mix carries no response tokens; cannot split collision mass by task")
        for (layer, expert), sum_of_squares in self._wave_sos.items():
            if sum_of_squares == 0:
                continue
            for tag, tokens in task_tokens.items():
                key = (tag, layer, expert)
                self._task_sos[key] = self._task_sos.get(key, 0.0) + (tokens / total_tokens) * sum_of_squares
        self._wave_sos = {}
        self._fired = set()
        self._waves_flushed += 1

    def finish(self, num_micro_batches: int) -> None:
        """Remove the gradient hooks; raise unless every queued micro-batch produced exactly one wave."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._waves_flushed != num_micro_batches or self._pending_task_tokens:
            raise ValueError(
                f"router collision telemetry flushed {self._waves_flushed} gradient waves for "
                f"{num_micro_batches} micro-batches; the backward schedule and the task-mix "
                "queue disagree"
            )

    def scaled_task_sos(self, scale: float) -> dict[tuple[str, int, int], float]:
        """Local per-task sum of squares, scaled for tensor-parallel replication."""
        return {key: value * scale for key, value in self._task_sos.items()}


def collision_summary_metrics(task_sos: Mapping[tuple[str, int, int], float]) -> dict[str, float]:
    """Per-(task, layer) collision summaries; the full vectors go to the sidecar."""
    totals: dict[tuple[str, int], float] = {}
    maxima: dict[tuple[str, int], float] = {}
    for (tag, layer, _expert), sum_of_squares in task_sos.items():
        key = (tag, layer)
        totals[key] = totals.get(key, 0.0) + sum_of_squares
        maxima[key] = max(maxima.get(key, 0.0), math.sqrt(max(sum_of_squares, 0.0)))
    metrics: dict[str, float] = {}
    for (tag, layer), sum_of_squares in sorted(totals.items()):
        prefix = f"{ROUTER_COLLISION_METRIC_PREFIX}/{tag}/layer{layer}"
        metrics[f"{prefix}/mass"] = math.sqrt(max(sum_of_squares, 0.0))
        metrics[f"{prefix}/max_expert_share"] = maxima[(tag, layer)] / math.sqrt(max(sum_of_squares, 1e-30))
    return metrics


def collision_norms(
    task_sos: Mapping[tuple[str, int, int], float], num_experts: int
) -> dict[str, dict[str, list[float]]]:
    """Per-task, per-layer lists of per-global-expert gradient norms for the sidecar."""
    norms: dict[str, dict[int, list[float]]] = {}
    for (tag, layer, expert), sum_of_squares in task_sos.items():
        vectors = norms.setdefault(tag, {})
        vector = vectors.setdefault(layer, [0.0] * num_experts)
        vector[expert] = math.sqrt(max(sum_of_squares, 0.0))
    return {
        tag: {f"layer{layer}": vectors[layer] for layer in sorted(vectors)} for tag, vectors in sorted(norms.items())
    }


def router_telemetry_record(
    step: int,
    layers: list[int],
    load_counts: torch.Tensor,
    norms: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, Any]:
    """One sidecar record: full per-(layer, expert) vectors for this step."""
    return {
        "step": int(step),
        "layers": [int(layer) for layer in layers],
        "load_counts": {
            f"layer{layer}": [float(value) for value in load_counts[position]] for position, layer in enumerate(layers)
        },
        "collision_norms": {tag: dict(per_layer) for tag, per_layer in norms.items()},
    }


def write_router_telemetry_record(export_path: str, record: Mapping[str, Any]) -> None:
    """Append one JSON line to the telemetry sidecar under the export root."""
    separator = "/" if io.is_cloud_path(export_path) else os.sep
    sidecar = export_path.rstrip("/\\") + separator + ROUTER_TELEMETRY_SIDECAR_FILENAME
    io.makedirs(export_path, exist_ok=True)
    with io.open_file(sidecar, "a") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def merge_router_telemetry_metrics(
    load_stats: RouterLoadStats,
    collision: RouterCollisionTracker,
    metrics_list: list[dict],
    *,
    step: int,
    export_path: str,
) -> None:
    """Reduce this step's router telemetry across parallel groups and publish it.

    Every collective runs on every rank regardless of pipeline stage — an early
    return ahead of a collective deadlocks any PP > 1 run. Only the metric and
    sidecar writes are gated to the last pipeline stage (and, for the sidecar,
    to one rank inside it).
    """
    import megatron.core.parallel_state as mpu  # guarded: telemetry runs only inside the Megatron worker

    num_experts = collision.num_global_experts

    # Load counts: response-token counts are partial across DP/CP/TP and
    # disjoint across EP (alltoall token dispatch), so reduce over both groups.
    counts = load_stats.dense_counts()
    dense_reduce(counts, mpu.get_tensor_and_data_parallel_group(with_context_parallel=True))
    dense_reduce(counts, mpu.get_expert_model_parallel_group())

    # Collision mass: each rank holds unreduced local shards. Gradients shard
    # over the expert tensor group and replicate across the rest of the tensor
    # group, and stay partial across DP/CP; scale for the replication, then sum
    # the sparse per-task values over each group in turn.
    replication_scale = mpu.get_expert_tensor_parallel_world_size() / mpu.get_tensor_model_parallel_world_size()
    task_sos = collision.scaled_task_sos(replication_scale)
    for group in (
        mpu.get_expert_tensor_and_model_parallel_group(),
        mpu.get_tensor_model_parallel_group(),
        mpu.get_data_parallel_group(with_context_parallel=True),
    ):
        task_sos = sum_gathered_task_sos(task_sos, group)

    # Pipeline stages own disjoint layers; gather every stage's reduced values,
    # then only the last stage writes.
    pipeline_group = mpu.get_pipeline_model_parallel_group()
    gathered: list[Any] = [None] * torch.distributed.get_world_size(pipeline_group)
    torch.distributed.all_gather_object(gathered, (load_stats.layers, counts, task_sos), group=pipeline_group)

    if not mpu.is_pipeline_last_stage(ignore_virtual=True):
        return

    # Stages own disjoint layers: the record covers the union, in layer order.
    layers = sorted({layer for stage_layers, _, _ in gathered for layer in stage_layers})
    row_of = {layer: row for row, layer in enumerate(layers)}
    merged_counts = torch.zeros(len(layers), num_experts, dtype=torch.float64)
    merged_task_sos: dict[tuple[str, int, int], float] = {}
    for stage_layers, stage_counts, stage_task_sos in gathered:
        for index, layer in enumerate(stage_layers):
            merged_counts[row_of[layer]] += stage_counts[index]
        for key, value in stage_task_sos.items():
            merged_task_sos[key] = merged_task_sos.get(key, 0.0) + value

    metrics_list[-1].update(load_metrics(layers, merged_counts))
    metrics_list[-1].update(collision_summary_metrics(merged_task_sos))

    if is_telemetry_writer_rank(mpu):
        write_router_telemetry_record(
            export_path,
            router_telemetry_record(
                step, layers, merged_counts, collision_norms(merged_task_sos, num_experts)
            ),
        )


def sum_gathered_task_sos(task_sos: Mapping[tuple[str, int, int], float], group) -> dict[tuple[str, int, int], float]:
    """Sum sparse per-task mass over a parallel group with an object gather."""
    gathered: list[Any] = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(gathered, dict(task_sos), group=group)
    merged: dict[tuple[str, int, int], float] = {}
    for partial in gathered:
        for key, value in (partial or {}).items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


def is_telemetry_writer_rank(mpu) -> bool:
    """The single rank inside the last pipeline stage that writes the sidecar."""
    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_expert_model_parallel_rank() == 0
    )


def named_expert_bias_buffers(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Grug's per-layer ``expert_bias`` buffers: frozen while the router weights train."""
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


def router_policy_manifest(
    model_type: str,
    *,
    router_weights: str,
    selection_bias: str,
    aux_loss_coeff: float,
) -> dict[str, Any]:
    """The router policy record, derived from the live model and resolved config.

    ``router_weights`` reports whether router parameters actually train
    (``requires_grad``), ``selection_bias`` whether expert-bias buffers exist
    and are guarded, and ``aux_loss_coeff`` the resolved mcore value — none of
    these are hard-coded policy claims.
    """
    return {
        "model_type": model_type,
        "router_weights": router_weights,
        "selection_bias": selection_bias,
        "aux_loss_coeff": aux_loss_coeff,
    }


def router_telemetry_enabled(config) -> bool:
    """Whether training collects router load statistics and collision mass."""
    return bool(OmegaConf.select(config, "trainer.router_telemetry.enabled") or False)


def response_row_mask(attention_mask: torch.Tensor, loss_mask: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Router-row mask of one unpacked micro-batch: True where a row is a supervised response token.

    Megatron routes hidden states laid out sequence-major (``[s, b]`` flattened), so
    the ``[b, s]`` position mask is transposed before flattening.
    """
    batch, seq_len = attention_mask.shape
    positions = torch.zeros(batch, seq_len, dtype=torch.bool, device=loss_mask.device)
    positions[:, seq_len - num_actions :] = loss_mask.to(torch.bool)
    return positions.transpose(0, 1).reshape(-1)


class RouterTelemetryStep:
    """One mini-batch of router telemetry: load counts, per-task collision mass, and the sidecar write.

    Supports unpacked micro-batches without sequence or context parallelism, where a
    router's rows map one-to-one onto the micro-batch's positions; other layouts raise.
    """

    def __init__(self, model_chunks: list[nn.Module], config, *, gradient_checkpointing: bool):
        import megatron.core.parallel_state as mpu  # guarded: telemetry runs only inside the Megatron worker

        if len(model_chunks) != 1:
            raise ValueError("router telemetry does not support virtual pipeline parallelism")
        if config.sequence_parallel or mpu.get_context_parallel_world_size() > 1:
            raise ValueError("router telemetry requires sequence_parallel off and context parallel size 1")
        layer_map: dict[int, int] = {}
        for chunk in model_chunks:
            layer_map.update(build_router_layer_map(chunk))
        if not layer_map:
            raise ValueError("router telemetry found no routers in the model")
        num_global = int(config.num_moe_experts)
        local = num_global // mpu.get_expert_model_parallel_world_size()
        self.load_stats = RouterLoadStats(layer_map)
        self.collision = RouterCollisionTracker(
            model_chunks[0], expert_offset=mpu.get_expert_model_parallel_rank() * local, num_global_experts=num_global
        )
        self._gradient_checkpointing = gradient_checkpointing
        self._row_mask: torch.Tensor | None = None
        self._micro_batches = 0

    def begin_micro_batch(self, micro_batch, data_source_names: list[str] | None) -> None:
        """Set the response mask for this micro-batch's forward and queue its task mix for backward."""
        self._row_mask = response_row_mask(micro_batch.attention_mask, micro_batch.loss_mask, micro_batch.num_actions)
        response_tokens = micro_batch.loss_mask.to(torch.bool).sum(dim=1)
        task_tokens: dict[str, int] = {}
        ids = micro_batch.data_source_ids
        for row, tokens in enumerate(response_tokens.tolist()):
            source = int(ids[row]) if ids is not None else -1
            tag = data_source_names[source] if source >= 0 and data_source_names else "untagged"
            task_tokens[tag] = task_tokens.get(tag, 0) + int(tokens)
        self.collision.begin_micro_batch(task_tokens)
        self._micro_batches += 1

    def observe(self, observation: RouterObservation) -> None:
        if not should_count_router_forward(
            grad_enabled=torch.is_grad_enabled(), gradient_checkpointing=self._gradient_checkpointing
        ):
            return
        self.load_stats.observe(observation, row_mask=self._row_mask)

    def finish(self, metrics_list: list[dict], *, step: int, export_path: str) -> None:
        self.collision.finish(self._micro_batches)
        merge_router_telemetry_metrics(self.load_stats, self.collision, metrics_list, step=step, export_path=export_path)
