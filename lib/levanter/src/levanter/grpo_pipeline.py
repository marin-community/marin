# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-owned GRPO execution using JaxPP's standard 1F1B schedule.

MPMD placement and initialization follow Marin PR 8739, commit f32a83bc.
Model-specific stage wrappers implement the protocol below; this module owns
only scoring, objective reduction, global clipping, and optimizer execution.
"""

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any, Protocol

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.data.packing import SequencePacker, pack_documents
from levanter.grpo import GrpoConfig, grpo_loss
from levanter.grpo_model import GrpoExample
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.pipeline import reshape_batch_into_microbatches
from levanter.utils.jax_utils import barrier_sync, multihost_broadcast_sync

try:
    import jaxpp.api as pp  # pyrefly: ignore[missing-import]  # Optional pipeline extra.
except ModuleNotFoundError as error:
    if error.name != "jaxpp":
        raise
    pp = None


logger = logging.getLogger(__name__)

PIPELINE_SCHEDULE = "std_1f1b"
_SCORING_WINDOW_SIZE = 8
BATCH_AXES = ("replica_dcn", "data", "expert")
ROUTING_METRIC_NAMES = (
    "routing_assignments",
    "routing_sender_drops",
    "routing_receiver_drops",
    "routing_max_layer_drops",
    "routing_drop_layer",
)
ROUTING_MAX_METRICS = ("routing_max_layer_drops", "routing_drop_layer")
METRIC_NAMES = (
    "loss",
    "policy_loss",
    "policy_kl",
    "ppo_clip_ratio",
    "ppo_clip_ratio_low",
    "ppo_clip_ratio_high",
    "ppo_clip_pressure_low",
    "ppo_clip_pressure_high",
    "ppo_ratio_exact_unit_fraction",
    "log_ratio_abs_max",
)


class PipelineStage(Protocol):
    """An Equinox stage owning a contiguous layer range and its boundary weights."""

    def embed(self, token_ids: jax.Array) -> jax.Array: ...
    def run_blocks(self, hidden: jax.Array, segment_ids: jax.Array, position_ids: jax.Array) -> jax.Array: ...
    def run_blocks_with_stats(self, hidden: jax.Array, segment_ids: jax.Array, position_ids: jax.Array): ...
    def finish(self, hidden: jax.Array) -> jax.Array: ...
    def get_lm_head(self) -> jax.Array: ...
    def trainable_filter(self) -> Any: ...


@dataclass(frozen=True)
class GrpoPipelineConfig:
    stages: int
    microbatches: int
    max_grad_norm: float
    gradient_accum_dtype: str = "float32"
    expert_axis_size: int = 1

    def __post_init__(self):
        if self.stages < 2 or self.microbatches < self.stages:
            raise ValueError("Pipeline GRPO requires at least two stages and at least one microbatch per stage")
        if self.gradient_accum_dtype not in ("float32", "bfloat16"):
            raise ValueError("gradient_accum_dtype must be float32 or bfloat16")
        if self.expert_axis_size < 1:
            raise ValueError("expert_axis_size must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PipelineBatch:
    tokens: jax.Array
    attention_mask: jax.Array
    segment_ids: jax.Array
    position_ids: jax.Array
    advantages: jax.Array
    policy_weights: jax.Array
    kl_weights: jax.Array
    old_logprobs: jax.Array
    reference_logprobs: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PipelineState:
    trainable: tuple[Any, ...]
    frozen: tuple[Any, ...]
    opt_state: tuple[optax.OptState, ...]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _OptimizerState:
    trainable: tuple[Any, ...]
    opt_state: tuple[optax.OptState, ...]


def pipeline_batch(example: GrpoExample, *, microbatches: int) -> PipelineBatch:
    """Convert named examples once, then add an explicit temporal batch axis."""

    def sequence(value):
        return value.rearrange(("batch", "position")).array

    def response(value):
        return value.rearrange(("batch", "response")).array

    batch = PipelineBatch(
        sequence(example.tokens),
        sequence(example.attention_mask),
        sequence(example.attention_mask),
        sequence(example.position_ids),
        response(example.advantages),
        response(example.policy_weights),
        response(example.kl_weights),
        response(example.old_logprobs),
        (
            jnp.zeros_like(response(example.old_logprobs))
            if example.reference_logprobs is None
            else response(example.reference_logprobs)
        ),
    )
    return reshape_batch_into_microbatches(batch, microbatches)


def packed_pipeline_batch(
    example: GrpoExample,
    *,
    sequence_length: int,
    rows_per_microbatch: int,
    pad_token_id: int,
    minimum_microbatches: int = 1,
) -> PipelineBatch:
    """Pack complete trajectories while preserving their precomputed objectives.

    Only masked input padding is removed. Response fields move to the predictor
    of their original target token; prompt, segment-boundary, and padding targets
    receive zero objective weight. Source position IDs are preserved verbatim.
    """
    if sequence_length < 2 or rows_per_microbatch < 1 or minimum_microbatches < 1:
        raise ValueError("Packing requires sequence_length >= 2 and positive batch dimensions")
    source = jax.tree.map(lambda value: np.asarray(value[0]), pipeline_batch(example, microbatches=1))
    response_mask = np.asarray(example.response_mask.rearrange(("batch", "response")).array)
    if np.any((source.attention_mask != 0) & (source.attention_mask != 1)):
        raise ValueError("Packing requires a binary input attention mask")
    valid = source.attention_mask.astype(bool)
    lengths = valid.sum(axis=1)
    packs = pack_documents(lengths, sequence_length, slice_strategy="raise")
    microbatches = max(minimum_microbatches, (len(packs) + rows_per_microbatch - 1) // rows_per_microbatch)
    rows = microbatches * rows_per_microbatch
    tokens = np.full((rows, sequence_length), pad_token_id, dtype=np.int32)
    segments = np.full((rows, sequence_length), -1, dtype=np.int32)
    positions = np.zeros((rows, sequence_length), dtype=np.int32)
    response_names = ("advantages", "policy_weights", "kl_weights", "old_logprobs", "reference_logprobs")
    objective = {
        name: np.zeros((rows, sequence_length - 1), dtype=getattr(source, name).dtype) for name in response_names
    }
    response_start = source.tokens.shape[1] - source.old_logprobs.shape[1]
    if response_start < 1:
        raise ValueError("Each response needs a preceding prompt token")
    if np.any(response_mask > valid[:, response_start:]) or np.any(response_mask > valid[:, response_start - 1 : -1]):
        raise ValueError("Each response target and its predecessor must be attended")
    if np.any((response_mask == 0) & ((source.policy_weights != 0) | (source.kl_weights != 0))):
        raise ValueError("Objective weights must be zero outside the response mask")
    for row, documents in enumerate(packs):
        packer = SequencePacker(hax.Axis("position", sequence_length), max(len(documents), 1), pad_token_id)
        offset = 0
        for document in documents:
            indices = np.flatnonzero(valid[document])
            count = len(indices)
            packer.add_example(source.tokens[document, indices].tolist(), np.zeros(count), segment_id=document)
            positions[row, offset : offset + count] = source.position_ids[document, indices]
            for compact_target, target in enumerate(indices):
                if target < response_start or compact_target == 0 or not valid[document, target - 1]:
                    continue
                response_index = target - response_start
                predictor = offset + compact_target - 1
                for name in response_names:
                    objective[name][row, predictor] = getattr(source, name)[document, response_index]
            offset += count
        packed = packer.pack()
        tokens[row] = np.asarray(packed.tokens.array)
        segments[row] = np.asarray(packed.attn_mask.segment_ids[0].array)
    batch = PipelineBatch(
        tokens, (segments >= 0).astype(np.int32), segments, positions, *(objective[name] for name in response_names)
    )
    return reshape_batch_into_microbatches(jax.tree.map(jnp.asarray, batch), microbatches)


def _require_jaxpp():
    if pp is None:
        raise ModuleNotFoundError("GRPO pipeline execution requires the root `pipeline` extra")
    return pp


def make_pipeline_mesh(config: GrpoPipelineConfig, *, replica_axis_size: int = 1):
    """Create explicit stage/data/expert meshes without duplicating model state."""
    expert_axis_size = config.expert_axis_size
    if jax.process_count() > 1 and jax.process_count() != config.stages:
        raise ValueError("Multihost GRPO requires one JAX process per pipeline stage")
    api = _require_jaxpp()
    fixed = config.stages * replica_axis_size * expert_axis_size
    if jax.device_count() % fixed:
        raise ValueError("Device count must divide evenly into pipeline, replica, and expert axes")
    shape = (config.stages, replica_axis_size, jax.device_count() // fixed, expert_axis_size, 1)
    ordered_devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.local_hardware_id))
    devices = np.asarray(ordered_devices, dtype=object).reshape(shape)
    mesh = Mesh(devices, ("pipeline", *BATCH_AXES, "model"), axis_types=(AxisType.Explicit,) * len(shape))
    if mesh.is_multi_process:
        for stage_index in range(config.stages):
            if {device.process_index for device in devices[stage_index].flat} != {stage_index}:
                raise ValueError("Every process must own all devices of exactly one pipeline stage")
    return mesh, api.MpmdMesh(mesh, "pipeline")


def _is_array(value):
    return isinstance(value, (jax.Array, jax.ShapeDtypeStruct)) or (pp is not None and isinstance(value, pp.MpmdArray))


def _partition_specs(tree):
    def spec(value):
        if not _is_array(value):
            return None
        return (
            value.sharding.spec
            if isinstance(value.sharding, (NamedSharding, pp.MpmdSharding))
            else P(*([None] * value.ndim))
        )

    return jax.tree.map(spec, tree)


def _stage_shardings(mpmd_mesh, stage_index, tree):
    api = _require_jaxpp()

    def sharding(value):
        if not _is_array(value):
            return None
        spec = value.sharding.spec if isinstance(value.sharding, NamedSharding) else P(*([None] * value.ndim))
        return api.MpmdSharding(mpmd_mesh, mesh_ids={stage_index}, spec=spec)

    return jax.tree.map(sharding, tree)


def _owns(sharding):
    return any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat)


def initialize_pipeline_state(stages: tuple[PipelineStage, ...], optimizer, mpmd_mesh) -> PipelineState:
    """Consume source stages, place owned weights, then allocate stage-local moments."""
    api = _require_jaxpp()
    trainable, frozen = zip(*(eqx.partition(stage, stage.trainable_filter()) for stage in stages), strict=True)
    trainable = api.spmd_to_mpmd_reshard(
        mpmd_mesh,
        trainable,
        tuple(_stage_shardings(mpmd_mesh, i, stage) for i, stage in enumerate(trainable)),
        threshold=0,
    )
    frozen = api.spmd_to_mpmd_reshard(
        mpmd_mesh,
        frozen,
        tuple(_stage_shardings(mpmd_mesh, i, stage) for i, stage in enumerate(frozen)),
        threshold=0,
    )
    opt_state = []
    for i, stage in enumerate(trainable):
        sharding = NamedSharding(mpmd_mesh.unstack[i], P())

        def localize(value):
            if _is_array(value) and value.shape == ():
                if _owns(sharding):
                    return jax.device_put(np.asarray(value), sharding)
                return jax.make_array_from_single_device_arrays((), sharding, [], dtype=value.dtype)
            return value

        opt_state.append(jax.tree.map(localize, optimizer.init(stage)))
    return PipelineState(tuple(trainable), tuple(frozen), tuple(opt_state))


def _score_stages(
    trainable,
    frozen,
    batch: PipelineBatch,
    mp_policy,
    mark: Callable,
    *,
    report_routing=False,
    compute_weight_boundary: Callable = lambda value: value,
):
    hidden = None
    last = None
    routing = {
        # Integer auxiliary outputs generate float0 cotangents that JaxPP cannot
        # transfer. Cast exact per-microbatch counts back to int32 after AD.
        name: jnp.asarray(
            -1 if name == "routing_drop_layer" else 0,
            dtype=jnp.float32,
        )
        for name in ROUTING_METRIC_NAMES
    }
    for index, (params, fixed) in enumerate(zip(trainable, frozen, strict=True)):
        # Fixed routing bias remains FP32; only trainable weights enter compute precision.
        stage = eqx.combine(compute_weight_boundary(mp_policy.cast_to_compute(params)), fixed)
        if index == 0:
            hidden = stage.embed(batch.tokens)
        if report_routing:
            hidden, stage_routing = stage.run_blocks_with_stats(hidden, batch.segment_ids, batch.position_ids)
            routing = {
                name: (
                    jnp.maximum(routing[name], stage_routing[name].astype(jnp.float32))
                    if name in ROUTING_MAX_METRICS
                    else routing[name] + stage_routing[name].astype(jnp.float32)
                )
                for name in ROUTING_METRIC_NAMES
            }
        else:
            hidden = stage.run_blocks(hidden, batch.segment_ids, batch.position_ids)
        if index < len(trainable) - 1:
            if report_routing:
                hidden, routing = mark((hidden, routing))
            else:
                hidden = mark(hidden)
        last = stage
    hidden = last.finish(hidden)
    response_width = batch.old_logprobs.shape[-1]
    start = batch.tokens.shape[-1] - response_width
    if start < 1:
        raise ValueError("Response span needs a preceding prompt token")
    logprobs = -fused_linear_softmax_cross_entropy_loss(
        hidden[:, start - 1 : -1].astype(jnp.float32),
        last.get_lm_head().astype(jnp.float32),
        batch.tokens[:, start:],
        reduction="none",
        logsumexp_weight=0.0,
    )
    return (logprobs, routing) if report_routing else logprobs


def pipeline_loss(
    trainable, frozen, batch: PipelineBatch, *, config: GrpoConfig, mp_policy, mark=lambda x: x, report_routing=False
):
    """Compute one microbatch contribution; objective weights already span the batch."""
    scored = _score_stages(trainable, frozen, batch, mp_policy, mark, report_routing=report_routing)
    logprobs, routing = scored if report_routing else (scored, {})
    axes = (hax.Axis("batch", logprobs.shape[0]), hax.Axis("response", logprobs.shape[1]))
    named = tuple(
        hax.named(value, axes)
        for value in (
            logprobs,
            batch.old_logprobs,
            batch.reference_logprobs,
            batch.advantages,
            batch.policy_weights,
            batch.kl_weights,
        )
    )
    loss, metrics = grpo_loss(*named, config=config, accumulation_steps=1)
    return mark((loss, {**{name: metric.value() for name, metric in metrics.items()}, **routing}))


def _pipeline_value_and_grad(
    compute_params, frozen, batch, *, config, mp_policy, mark, gradient_accum_dtype="float32", report_routing=False
):
    def loss(params):
        return pipeline_loss(
            params, frozen, batch, config=config, mp_policy=mp_policy, mark=mark, report_routing=report_routing
        )

    result, gradients = jax.value_and_grad(loss, has_aux=True)(compute_params)
    if report_routing:
        value, metrics = result
        metrics = {
            name: (
                metric.astype(jnp.int32)
                if name in ROUTING_METRIC_NAMES and name not in ROUTING_MAX_METRICS
                else metric
            )
            for name, metric in metrics.items()
        }
        result = value, metrics
    # The default reproduces the master-to-compute cast pullback per microbatch.
    # BF16 instead rounds the temporal accumulator after every addition. Data
    # parallel reductions inside this derivative retain their existing order.
    return result, jax.tree.map(lambda gradient: gradient.astype(gradient_accum_dtype), gradients)


def make_pipeline_train_step(
    optimizer, mp_policy, sample_state, sample_batches, *, config: GrpoPipelineConfig, grpo: GrpoConfig, mpmd_mesh
):
    """Compile 1F1B gradients and global clipping; each update consumes its state."""
    api = _require_jaxpp()
    metric_names = (*METRIC_NAMES, *ROUTING_METRIC_NAMES) if config.expert_axis_size > 1 else METRIC_NAMES
    metric_ops = {
        name: api.Max if name in ("log_ratio_abs_max", *ROUTING_MAX_METRICS) else api.Add for name in metric_names
    }

    def step(state, frozen, batches):
        # Share compute weights across all in-flight microbatches. Casting inside
        # the differentiated loss retains another BF16 copy in every residual.
        compute_params = mp_policy.cast_to_compute(state.trainable)
        (value, metrics), grads = api.treduce(
            lambda batch: _pipeline_value_and_grad(
                compute_params,
                frozen,
                batch,
                config=grpo,
                mp_policy=mp_policy,
                mark=api.mark_stage_end,
                gradient_accum_dtype=config.gradient_accum_dtype,
                report_routing=config.expert_axis_size > 1,
            ),
            batches,
            schedule=api.Std1F1B(num_stages=config.stages),
            operation=((api.Add, metric_ops), api.Add),
        )
        grads = mp_policy.cast_to_param(grads)
        norms = tuple(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(stage)) for stage in grads)
        norm = jnp.sqrt(api.cross_mpmd_all_reduce(*norms))
        scale = jnp.minimum(1.0, config.max_grad_norm / jnp.maximum(norm, config.max_grad_norm))
        next_params, next_opt = [], []
        for params, state_opt, gradient in zip(state.trainable, state.opt_state, grads, strict=True):
            gradient = jax.tree.map(lambda g: g * scale, gradient)
            updates, state_opt = optimizer.update(gradient, state_opt, params)
            next_params.append(eqx.apply_updates(params, updates))
            next_opt.append(state_opt)
        metrics = {**metrics, "grad_norm": norm}
        return _OptimizerState(tuple(next_params), tuple(next_opt)), metrics

    optimizer_specs = _partition_specs(_OptimizerState(sample_state.trainable, sample_state.opt_state))
    function = api.mpmd_jit_with_loop(
        step,
        mpmd_mesh=mpmd_mesh,
        in_specs=(optimizer_specs, _partition_specs(sample_state.frozen), _partition_specs(sample_batches)),
        out_specs=(optimizer_specs, {name: P() for name in (*metric_names, "grad_norm")}),
        donate_argnums=(0,),
    )
    return function


def make_pipeline_scorer(mp_policy, sample_state, sample_microbatch, *, config: GrpoPipelineConfig, mpmd_mesh):
    """Build a forward-only MPMD scorer using the training stage placement."""
    api = _require_jaxpp()

    def score(trainable, frozen, batch):
        # AD materializes compute weights; preserve that rounding boundary in
        # forward-only scoring too, without changing the training gradients.
        return api.mark_stage_end(
            _score_stages(
                trainable,
                frozen,
                batch,
                mp_policy,
                api.mark_stage_end,
                report_routing=config.expert_axis_size > 1,
                compute_weight_boundary=partial(jax.tree.map, jax.lax.optimization_barrier),
            )
        )

    return api.mpmd_jit_by_yield(
        score,
        mpmd_mesh=mpmd_mesh,
        target_num_stages=config.stages,
        in_shardings=(
            _partition_specs(sample_state.trainable),
            _partition_specs(sample_state.frozen),
            _partition_specs(sample_microbatch),
        ),
        out_shardings=(
            (P(BATCH_AXES, None), {name: P() for name in ROUTING_METRIC_NAMES})
            if config.expert_axis_size > 1
            else P(BATCH_AXES, None)
        ),
    )


def _place_existing(tree, targets, mpmd_mesh):
    """Keep tensor state stage-owned while placing shared scalar counters."""
    api = _require_jaxpp()

    def place(value, target):
        if not _is_array(value) or not target.mesh_ids:
            return value
        source = value.first_mpmd_replica if isinstance(value, api.MpmdArray) else value
        source_ids = set(mpmd_mesh.mpmd_indices_for_mesh(value.sharding.mesh))
        if isinstance(value, api.MpmdArray) and source_ids == target.mesh_ids and value.sharding.spec == target.spec:
            return value
        if mpmd_mesh.jax_mesh.is_multi_process and not target.mesh_ids <= source_ids:
            if value.shape:
                raise ValueError(
                    f"Compiled stage input requires new owners: {source_ids} -> {target.mesh_ids}, shape={value.shape}"
                )
            # CSE can share Adam's step counter across stage programs. Broadcast the
            # actual scalar through the coordination service, avoiding GPU collectives.
            is_source = jax.process_index() == min(source_ids)
            scalar = np.asarray(source).item() if is_source else None
            source = np.asarray(multihost_broadcast_sync(scalar, is_source=is_source), dtype=value.dtype)
        arrays = []
        for index in sorted(target.mesh_ids):
            sharding = NamedSharding(mpmd_mesh.unstack[index], target.spec)
            if _owns(sharding):
                if source is None:
                    raise ValueError("An owned stage input needs a local source replica")
                arrays.append(jax.device_put(source, sharding))
        return api.MpmdArray(arrays, target, shape=value.shape, dtype=value.dtype)

    return jax.tree.map(place, tree, targets)


def prepare_pipeline_train(step, state, batches, mpmd_mesh):
    """Compile, place scalar state, and consume input batches into stage-owned arrays."""
    api = _require_jaxpp()
    # JaxPP establishes a stage-sized lowering mesh; an enclosing full mesh conflicts.
    with jax.set_mesh(None):
        compiled = step.compile(_OptimizerState(state.trainable, state.opt_state), state.frozen, batches)
    (optimizer_specs, frozen_specs, batch_specs), kwargs = compiled.in_shardings
    state_specs = PipelineState(optimizer_specs.trainable, frozen_specs, optimizer_specs.opt_state)
    if kwargs:
        raise ValueError("Pipeline step accepts positional state and batches")

    for index, stage in enumerate(state_specs.trainable):
        if any(target.mesh_ids != {index} for target in jax.tree.leaves(stage)):
            raise ValueError("Pipeline compilation must preserve unique parameter ownership per stage")
    for index, (stage_values, stage_targets) in enumerate(zip(state.opt_state, state_specs.opt_state, strict=True)):
        for value, target in zip(jax.tree.leaves(stage_values), jax.tree.leaves(stage_targets), strict=True):
            if value.shape and target.mesh_ids != {index}:
                raise ValueError("Pipeline compilation must preserve unique optimizer-moment ownership per stage")
    state = _place_existing(state, state_specs, mpmd_mesh)
    batches = api.spmd_to_mpmd_reshard(mpmd_mesh, batches, batch_specs, threshold=0)
    batches = _place_existing(batches, batch_specs, mpmd_mesh)
    return partial(_run_pipeline_update, compiled), state, batches


def _run_pipeline_update(compiled, state, batches):
    # JaxPP's donation lifetime fences execute on individual stage meshes.
    with jax.set_mesh(None):
        optimizer, metrics = compiled(_OptimizerState(state.trainable, state.opt_state), state.frozen, batches)
    return PipelineState(optimizer.trainable, state.frozen, optimizer.opt_state), metrics


class RoutingDropPolicy(StrEnum):
    REJECT = "reject"
    REPORT = "report"


def validate_routing_drops(metrics, *, context, policy: RoutingDropPolicy = RoutingDropPolicy.REJECT):
    """Reject routing drops unless an experiment explicitly requests reporting."""
    sender = int(metrics.get("routing_sender_drops", 0))
    receiver = int(metrics.get("routing_receiver_drops", 0))
    if not (sender or receiver):
        return
    message = f"{context}: expert capacity dropped assignments (padding included): {dict(metrics)}"
    if policy == RoutingDropPolicy.REJECT:
        raise FloatingPointError(message)
    logger.warning(message)


def _scorer_repeatability_details(first_logprobs, repeated_logprobs, policy_weights, first_routing, repeated_routing):
    active = policy_weights > 0
    difference = jnp.abs(first_logprobs.astype(jnp.float32) - repeated_logprobs.astype(jnp.float32))
    changed = first_logprobs != repeated_logprobs
    return {
        "active_logprob_max_abs": float(jnp.max(jnp.where(active, difference, 0))),
        "inactive_logprob_max_abs": float(jnp.max(jnp.where(active, 0, difference))),
        "active_logprob_changed": int(jnp.sum(changed & active)),
        "inactive_logprob_changed": int(jnp.sum(changed & ~active)),
        "first_nonfinite_logprobs": int(jnp.sum(~jnp.isfinite(first_logprobs))),
        "repeated_nonfinite_logprobs": int(jnp.sum(~jnp.isfinite(repeated_logprobs))),
        "routing": {
            name: {"first": int(first_routing[name]), "repeated": int(repeated_routing[name])}
            for name in first_routing
        },
    }


def score_pipeline_batch(
    scorer,
    state,
    batches: PipelineBatch,
    mpmd_mesh,
    *,
    routing_drop_policy: RoutingDropPolicy = RoutingDropPolicy.REJECT,
) -> PipelineBatch:
    """Freeze raw old scores and verify exact first-microbatch scorer repeatability."""
    api = _require_jaxpp()
    sample = jax.tree.map(lambda x: x[0], batches)
    with jax.set_mesh(None):
        compiled = scorer.compile(state.trainable, state.frozen, sample)
    specs, kwargs = compiled.in_shardings
    if kwargs:
        raise ValueError("Pipeline scorer accepts positional parameters and batch")
    scoring_trainable = _place_existing(state.trainable, specs[0], mpmd_mesh)
    scoring_frozen = _place_existing(state.frozen, specs[1], mpmd_mesh)
    sharding = NamedSharding(mpmd_mesh.jax_mesh, P(BATCH_AXES, None))
    report_routing = mpmd_mesh.jax_mesh.shape["expert"] > 1
    output_sharding = (
        (sharding, {name: NamedSharding(mpmd_mesh.jax_mesh, P()) for name in ROUTING_METRIC_NAMES})
        if report_routing
        else sharding
    )
    routing_totals = {name: -1 if name == "routing_drop_layer" else 0 for name in ROUTING_METRIC_NAMES}

    def checked_score(result, index):
        if not report_routing:
            return result
        logprobs, routing = result
        values = {name: int(value) for name, value in routing.items()}
        validate_routing_drops(values, context=f"GRPO scorer microbatch {index}", policy=routing_drop_policy)
        for name, value in values.items():
            routing_totals[name] = (
                max(routing_totals[name], value) if name in ROUTING_MAX_METRICS else routing_totals[name] + value
            )
        return logprobs

    stacked_sharding = NamedSharding(mpmd_mesh.jax_mesh, P(None, BATCH_AXES, None))
    # Preserve the original first-microbatch repeatability check before allowing
    # independent microbatches to overlap across stages.
    placed = api.spmd_to_mpmd_reshard(mpmd_mesh, sample, specs[2], threshold=0)
    placed = _place_existing(placed, specs[2], mpmd_mesh)
    with jax.set_mesh(None):
        score = compiled(scoring_trainable, scoring_frozen, placed)
        score = api.mpmd_to_spmd_reshard(mpmd_mesh, score, output_sharding, threshold=0)
    jax.block_until_ready(score)
    barrier_sync()
    with jax.set_mesh(None):
        repeated = compiled(scoring_trainable, scoring_frozen, placed)
        repeated = api.mpmd_to_spmd_reshard(mpmd_mesh, repeated, output_sharding, threshold=0)
    jax.block_until_ready(repeated)
    barrier_sync()
    if not all(
        bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(score), jax.tree.leaves(repeated), strict=True)
    ):
        first_logprobs = score[0] if report_routing else score
        repeated_logprobs = repeated[0] if report_routing else repeated
        details = _scorer_repeatability_details(
            first_logprobs,
            repeated_logprobs,
            sample.policy_weights,
            score[1] if report_routing else {},
            repeated[1] if report_routing else {},
        )
        raise FloatingPointError(f"Compiled GRPO scorer is not repeatable: {details}")
    logger.info("Verified exact compiled scorer repeatability on the first microbatch")
    score = checked_score(score, 0)
    scores = [jax.lax.reshape(score, (1, *score.shape), out_sharding=stacked_sharding)]
    for start in range(1, batches.tokens.shape[0], _SCORING_WINDOW_SIZE):
        stop = min(start + _SCORING_WINDOW_SIZE, batches.tokens.shape[0])
        window = tuple(jax.tree.map(lambda value, index=index: value[index], batches) for index in range(start, stop))
        window_specs = (specs[2],) * len(window)
        # An explicit positive threshold batches the tree into one resharding
        # group. Zero would split every leaf; None queries memory across hosts.
        input_bytes = sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(window))
        placed_window = api.spmd_to_mpmd_reshard(
            mpmd_mesh, window, window_specs, threshold=input_bytes * mpmd_mesh.mpmd_dim
        )
        placed_window = _place_existing(placed_window, window_specs, mpmd_mesh)
        with jax.set_mesh(None):
            pending = tuple(compiled(scoring_trainable, scoring_frozen, microbatch) for microbatch in placed_window)
            output_bytes = sum(
                int(np.prod(value.shape)) * np.dtype(value.dtype).itemsize for value in jax.tree.leaves(pending)
            )
            completed = api.mpmd_to_spmd_reshard(
                mpmd_mesh, pending, (output_sharding,) * len(pending), threshold=output_bytes * mpmd_mesh.mpmd_dim
            )
        for index, result in enumerate(completed, start=start):
            value = checked_score(result, index)
            scores.append(jax.lax.reshape(value, (1, *value.shape), out_sharding=stacked_sharding))
    if report_routing:
        logger.info("GRPO scorer routing totals (padding included): %s", routing_totals)
    return dataclasses.replace(batches, old_logprobs=jnp.concatenate(scores, axis=0))
