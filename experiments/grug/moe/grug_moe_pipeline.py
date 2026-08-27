# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical pipeline-parallel implementation for Grug MoE.

This file is intentionally model-specific. A new Grug model should copy it and
adapt the stage boundaries and stage-local forward functions. Stage boundaries
remain explicit for both the MPMD task implementation and JaxPP's automatic
equation transform.

The implementation provides standard 1F1B, an explicit zero-bubble schedule,
JaxPP's equation-level automatic zero-bubble transform, and automatic DualPipeV
with two logical stages on each physical pipeline rank.
Standard 1F1B supports any positive microbatch count, including counts smaller
than the number of stages. Zero-bubble requires at least one microbatch per
stage and retains stage-local VJP residual arrays between its input- and
weight-gradient tasks. DualPipeV also requires at least one microbatch per
logical stage.

The intended call sequence is visible in the public function names: construct
the mesh, initialize and place the state, split and place each batch, build the
step once, pass it through ``prepare_explicit_step``, then call it for each batch.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.jax_utils import named_call
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.pipeline import (
    PipelineDirection,
    evenly_partition_layers,
    split_batch_into_microbatches,
    standard_1f1b_stage_schedule,
)

from experiments.grug.moe.model import (
    BATCH_AXES,
    Block,
    GatedNorm,
    GrugModelConfig,
    RMSNorm,
    Transformer,
)

try:
    import jaxpp.api as jaxpp
    from jaxpp.experimental import mpmd
    from jaxpp.experimental.mpmd import LoweredMpmdFun
except ModuleNotFoundError:
    jaxpp = None
    mpmd = None


TRAIN_LOSS_KEY = "train/loss"
_QB_BETA_PER_LAYER_KEY = "qb_beta_per_layer"

type _ArrayValue = jax.Array | jax.ShapeDtypeStruct | jaxpp.MpmdArray
type _StagePullback = Callable[[tuple[jax.Array, jax.Array]], tuple[GrugMoePipelineStage, jax.Array]]


@dataclass(frozen=True)
class GrugMoePipelineConfig:
    stages: int
    microbatches: int
    physical_stages: int | None = None

    def __post_init__(self) -> None:
        if self.stages < 2:
            raise ValueError(f"pipeline parallelism requires at least 2 stages, got {self.stages}")
        if self.microbatches <= 0:
            raise ValueError(f"microbatches must be positive, got {self.microbatches}")
        if self.physical_stages is not None:
            if self.physical_stages < 2:
                raise ValueError(f"pipeline parallelism requires at least 2 physical stages, got {self.physical_stages}")
            if self.stages != 2 * self.physical_stages:
                raise ValueError(
                    "virtual pipeline parallelism requires exactly two logical stages per physical stage; "
                    f"got {self.stages} logical and {self.physical_stages} physical stages"
                )

    @property
    def mpmd_stages(self) -> int:
        return self.stages if self.physical_stages is None else self.physical_stages


def make_pipeline_mesh(
    config: GrugMoePipelineConfig,
    *,
    expert_axis_size: int,
    replica_axis_size: int | None = None,
):
    """Build the concrete Grug mesh and wrap it as a JaxPP MPMD mesh."""
    pp, _ = _require_jaxpp()
    if replica_axis_size is None:
        replica_axis_size = max(1, jax.process_count() // config.mpmd_stages)
    fixed_axes = config.mpmd_stages * replica_axis_size * expert_axis_size
    if jax.device_count() % fixed_axes != 0:
        raise ValueError(
            f"device count {jax.device_count()} must be divisible by stages ({config.mpmd_stages}) * "
            f"replicas ({replica_axis_size}) * experts ({expert_axis_size})"
        )

    data_axis_size = jax.device_count() // fixed_axes
    shape = (config.mpmd_stages, replica_axis_size, data_axis_size, expert_axis_size, 1)
    axis_names = ("pipeline", "replica_dcn", "data", "expert", "model")
    devices = np.asarray(jax.devices(), dtype=object).reshape(shape)
    mesh = Mesh(devices, axis_names, axis_types=(AxisType.Explicit,) * len(axis_names))
    if mesh.is_multi_process:
        local_stages = {int(np.argwhere(devices == device)[0][0]) for device in jax.local_devices()}
        if len(local_stages) != 1:
            raise ValueError(f"each JAX process must own exactly one pipeline stage; got {sorted(local_stages)}")
    return mesh, pp.MpmdMesh(mesh, "pipeline")


class GrugMoePipelineStage(eqx.Module):
    """The parameters and layer range owned by one pipeline stage."""

    token_embed: jax.Array | None
    embed_norm: RMSNorm | None
    embed_gated_norm: GatedNorm | None
    output_proj: jax.Array | None
    blocks: tuple[Block, ...]
    final_norm: RMSNorm | None
    final_gated_norm: GatedNorm | None
    config: GrugModelConfig = eqx.field(static=True)
    stage_index: int = eqx.field(static=True)
    start_layer: int = eqx.field(static=True)
    end_layer: int = eqx.field(static=True)

    @named_call
    def embed(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        if self.token_embed is None or self.embed_norm is None or self.embed_gated_norm is None:
            raise ValueError("only stage 0 owns the token embedding")
        hidden = self.token_embed.at[token_ids].get(out_sharding=P(BATCH_AXES))
        return self.embed_gated_norm(self.embed_norm(hidden))

    @named_call
    def run_blocks(
        self,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array | None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
        remat_policy = (
            jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
            if cfg.remat_mode == "save_moe"
            else None
        )

        block_metrics = []
        for local_index, block in enumerate(self.blocks):
            layer_index = self.start_layer + local_index
            is_long = layer_index % 4 == 3 or layer_index == cfg.num_layers - 1
            layer_mask = long_mask if is_long else short_mask
            hidden, metrics = eqx.filter_checkpoint(block, policy=remat_policy)(
                hidden,
                layer_mask,
                is_long and not cfg.disable_pko,
                is_long and cfg.disable_long_rope,
            )
            block_metrics.append(metrics)

        return hidden, _stack_router_metrics(block_metrics)

    @named_call
    def finish(self, hidden: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        if self.final_norm is None or self.final_gated_norm is None:
            raise ValueError("only the final stage owns the final norms")
        return self.final_gated_norm(self.final_norm(hidden))

    @named_call
    def cross_entropy_loss(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        logsumexp_weight: float | None,
        reduction: str = "mean",
    ) -> jax.Array:
        if self.output_proj is None:
            raise ValueError("only the final stage owns the output projection")
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight.astype(jnp.float32),
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=jnp.float32,
        )

    def local_router_loss(self, router_metrics: dict[str, jax.Array]) -> jax.Array:
        coefficient = jnp.asarray(self.config.router_z_loss_coef / self.config.num_layers, dtype=jnp.float32)
        return coefficient * jnp.sum(router_metrics["router_z_loss_per_layer"])


@register_dataclass
@dataclass(frozen=True)
class GrugMoePipelineState:
    step: jax.Array
    params: tuple[GrugMoePipelineStage, ...]
    opt_state: tuple[optax.OptState, ...]
    pending_qb_betas: tuple[jax.Array, ...]


@register_dataclass
@dataclass(frozen=True)
class GrugMoeAutomaticPipelineState:
    """Array state for JaxPP's automatic pipeline transform."""

    step: jax.Array
    trainable_params: tuple[GrugMoePipelineStage, ...]
    opt_state: tuple[optax.OptState, ...]
    pending_qb_betas: tuple[jax.Array, ...]


@register_dataclass
@dataclass(frozen=True)
class StageForwardWithPullbackResult:
    output: jax.Array
    next_qb_betas: jax.Array
    router_loss: jax.Array
    pullback: _StagePullback


@dataclass(frozen=True)
class _InitializedMpmdStageState:
    step: jax.Array
    opt_state: tuple[optax.OptState, ...]
    pending_qb_betas: tuple[jax.Array, ...]


@dataclass(frozen=True)
class PreparedAutomaticMpmdStep:
    step: Callable[..., tuple[GrugMoeAutomaticPipelineState, dict[str, jax.Array]]]
    state: GrugMoeAutomaticPipelineState
    batches: GrugLmExample
    loss_denominator: jax.Array


@dataclass(frozen=True)
class _ExplicitZeroBubbleTasks:
    stage_forwards: tuple[Callable[..., tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]], ...]
    last_forward: Callable[..., tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]]
    stage0_weight: Callable[..., GrugMoePipelineStage]
    input_gradients: tuple[Callable[..., jax.Array], ...]
    weight_gradients: tuple[Callable[..., GrugMoePipelineStage], ...]


def split_transformer(
    model: Transformer,
    num_stages: int,
    *,
    layer_counts: tuple[int, ...] | None = None,
) -> tuple[GrugMoePipelineStage, ...]:
    """Split a Grug MoE transformer into explicit, contiguous stage pytrees."""
    if layer_counts is None:
        ranges = evenly_partition_layers(len(model.blocks), num_stages)
    else:
        if len(layer_counts) != num_stages:
            raise ValueError(f"expected {num_stages} layer counts, got {len(layer_counts)}")
        if any(count <= 0 for count in layer_counts):
            raise ValueError(f"layer counts must be positive, got {layer_counts}")
        if sum(layer_counts) != len(model.blocks):
            raise ValueError(f"layer counts sum to {sum(layer_counts)}, but model has {len(model.blocks)} layers")
        boundaries = np.cumsum((0, *layer_counts))
        ranges = tuple((int(start), int(end)) for start, end in itertools.pairwise(boundaries))
    stages = []
    for stage_index, (start_layer, end_layer) in enumerate(ranges):
        is_first = stage_index == 0
        is_last = stage_index == num_stages - 1
        stages.append(
            GrugMoePipelineStage(
                token_embed=model.token_embed if is_first else None,
                embed_norm=model.embed_norm if is_first else None,
                embed_gated_norm=model.embed_gated_norm if is_first else None,
                output_proj=model.output_proj if is_last else None,
                blocks=model.blocks[start_layer:end_layer],
                final_norm=model.final_norm if is_last else None,
                final_gated_norm=model.final_gated_norm if is_last else None,
                config=model.config,
                stage_index=stage_index,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
    return tuple(stages)


def merge_stages(stages: tuple[GrugMoePipelineStage, ...]) -> Transformer:
    """Reassemble stage-local parameters into the ordinary Grug model pytree."""
    if len(stages) < 2:
        raise ValueError("at least two pipeline stages are required")
    for expected_index, stage in enumerate(stages):
        if stage.stage_index != expected_index:
            raise ValueError(f"expected stage {expected_index}, got {stage.stage_index}")

    first = stages[0]
    last = stages[-1]
    if first.token_embed is None or first.embed_norm is None or first.embed_gated_norm is None:
        raise ValueError("stage 0 is missing embedding parameters")
    if last.output_proj is None or last.final_norm is None or last.final_gated_norm is None:
        raise ValueError("final stage is missing output parameters")
    return Transformer(
        token_embed=first.token_embed,
        embed_norm=first.embed_norm,
        embed_gated_norm=first.embed_gated_norm,
        output_proj=last.output_proj,
        blocks=tuple(block for stage in stages for block in stage.blocks),
        final_norm=last.final_norm,
        final_gated_norm=last.final_gated_norm,
        config=first.config,
    )


def split_automatic_stages(
    model: Transformer,
    *,
    num_stages: int,
    layer_counts: tuple[int, ...] | None = None,
) -> tuple[tuple[GrugMoePipelineStage, ...], tuple[GrugMoePipelineStage, ...]]:
    stages = split_transformer(model, num_stages, layer_counts=layer_counts)
    trainable_stages = []
    static_stages = []
    for stage in stages:
        trainable_stage, static_stage = eqx.partition(stage, eqx.is_array)
        for block_index in range(len(stage.blocks)):
            trainable_stage = eqx.tree_at(
                lambda current, index=block_index: current.blocks[index].mlp.router_bias,
                trainable_stage,
                None,
            )
        trainable_stages.append(trainable_stage)
        static_stages.append(static_stage)
    return tuple(trainable_stages), tuple(static_stages)


def _process_has_sharding(sharding: NamedSharding) -> bool:
    process_index = jax.process_index()
    return any(device.process_index == process_index for device in sharding.mesh.devices.flat)


def _empty_sharded_array(shape: tuple[int, ...], dtype, sharding: NamedSharding) -> jax.Array:
    return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)


def _stage_local_scalar(value: jax.Array, sharding: NamedSharding) -> jax.Array:
    if not _process_has_sharding(sharding):
        return _empty_sharded_array((), value.dtype, sharding)
    return jax.device_put(np.asarray(value), sharding)


def _localize_optimizer_scalars(mpmd_mesh, stage_index: int, opt_state):
    stage_mesh = mpmd_mesh.unstack[stage_index]

    def localize(value):
        if _is_array(value) and value.shape == ():
            return _stage_local_scalar(value, NamedSharding(stage_mesh, P()))
        return value

    return jax.tree.map(localize, opt_state)


def _initialize_mpmd_stage_state(
    stages: tuple[GrugMoePipelineStage, ...],
    optimizer: optax.GradientTransformation,
    mpmd_mesh,
    stage_to_mpmd_index: tuple[int, ...],
    memory_threshold: int | None,
) -> _InitializedMpmdStageState:
    pp, _ = _require_jaxpp()
    qb_targets = tuple(
        pp.MpmdSharding(mpmd_mesh, mesh_ids={mpmd_index}, spec=P(None, None)) for mpmd_index in stage_to_mpmd_index
    )
    pending_qb_betas = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        tuple(jnp.zeros((len(stage.blocks), stage.config.num_experts), dtype=jnp.float32) for stage in stages),
        qb_targets,
        threshold=memory_threshold,
    )
    opt_state = tuple(
        _localize_optimizer_scalars(mpmd_mesh, mpmd_index, optimizer.init(stage))
        for mpmd_index, stage in zip(stage_to_mpmd_index, stages, strict=True)
    )
    step = _stage_local_scalar(jnp.array(0, dtype=jnp.int32), NamedSharding(mpmd_mesh.unstack[0], P()))
    return _InitializedMpmdStageState(
        step=step,
        opt_state=opt_state,
        pending_qb_betas=pending_qb_betas,
    )


def make_mpmd_pipeline_state(
    model: Transformer,
    optimizer: optax.GradientTransformation,
    mpmd_mesh,
    *,
    num_stages: int,
    layer_counts: tuple[int, ...] | None = None,
    memory_threshold: int | None = None,
) -> GrugMoePipelineState:
    """Place parameters before initializing optimizer state to avoid peak replication."""
    pp, _ = _require_jaxpp()
    stages = split_transformer(model, num_stages, layer_counts=layer_counts)
    stage_targets = tuple(_mpmd_sharding_tree(mpmd_mesh, stage_index, stage) for stage_index, stage in enumerate(stages))
    stages = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        stages,
        stage_targets,
        threshold=memory_threshold,
    )
    initialized = _initialize_mpmd_stage_state(
        stages,
        optimizer,
        mpmd_mesh,
        tuple(range(num_stages)),
        memory_threshold,
    )
    return GrugMoePipelineState(
        step=initialized.step,
        params=stages,
        opt_state=initialized.opt_state,
        pending_qb_betas=initialized.pending_qb_betas,
    )


def make_mpmd_automatic_pipeline_state(
    model: Transformer,
    optimizer: optax.GradientTransformation,
    mpmd_mesh,
    *,
    num_stages: int,
    layer_counts: tuple[int, ...] | None = None,
    stage_to_mpmd_index: tuple[int, ...] | None = None,
    memory_threshold: int | None = None,
) -> tuple[GrugMoeAutomaticPipelineState, tuple[GrugMoePipelineStage, ...]]:
    """Place each stage before creating its optimizer state."""
    pp, _ = _require_jaxpp()
    if stage_to_mpmd_index is None:
        stage_to_mpmd_index = tuple(range(num_stages))
    if len(stage_to_mpmd_index) != num_stages:
        raise ValueError(f"expected {num_stages} stage placements, got {len(stage_to_mpmd_index)}")
    trainable_stages, static_stages = split_automatic_stages(
        model,
        num_stages=num_stages,
        layer_counts=layer_counts,
    )
    trainable_targets = tuple(
        _mpmd_sharding_tree(mpmd_mesh, mpmd_index, stage)
        for mpmd_index, stage in zip(stage_to_mpmd_index, trainable_stages, strict=True)
    )
    trainable_stages = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        trainable_stages,
        trainable_targets,
        threshold=memory_threshold,
    )
    initialized = _initialize_mpmd_stage_state(
        trainable_stages,
        optimizer,
        mpmd_mesh,
        stage_to_mpmd_index,
        memory_threshold,
    )
    return (
        GrugMoeAutomaticPipelineState(
            step=initialized.step,
            trainable_params=trainable_stages,
            opt_state=initialized.opt_state,
            pending_qb_betas=initialized.pending_qb_betas,
        ),
        static_stages,
    )


def staged_loss(
    stages: tuple[GrugMoePipelineStage, ...],
    batch: GrugLmExample,
    *,
    logsumexp_weight: float | None = None,
) -> jax.Array:
    """Reference sequential execution of the split model, used for parity checks."""
    hidden = stages[0].embed(batch.tokens)
    router_loss = jnp.array(0.0, dtype=jnp.float32)
    for stage in stages:
        hidden, metrics = stage.run_blocks(hidden, batch.attn_mask)
        router_loss = router_loss + stage.local_router_loss(metrics)
    hidden = stages[-1].finish(hidden)
    return (
        stages[-1].cross_entropy_loss(
            hidden,
            batch.tokens,
            batch.loss_weight,
            logsumexp_weight=logsumexp_weight,
        )
        + router_loss
    )


def microbatched_staged_loss(
    stages: tuple[GrugMoePipelineStage, ...],
    batch: GrugLmExample,
    *,
    num_microbatches: int,
    logsumexp_weight: float | None = None,
) -> jax.Array:
    """Sequential oracle for the loss scaling used by the MPMD microbatch step."""
    cross_entropy_sum = jnp.array(0.0, dtype=jnp.float32)
    loss_denominator = jnp.array(0.0, dtype=jnp.float32)
    router_loss = jnp.array(0.0, dtype=jnp.float32)
    for microbatch in split_batch_into_microbatches(batch, num_microbatches):
        hidden = stages[0].embed(microbatch.tokens)
        for stage in stages:
            hidden, metrics = stage.run_blocks(hidden, microbatch.attn_mask)
            router_loss = router_loss + stage.local_router_loss(metrics) / num_microbatches
        hidden = stages[-1].finish(hidden)
        cross_entropy_sum = cross_entropy_sum + stages[-1].cross_entropy_loss(
            hidden,
            microbatch.tokens,
            microbatch.loss_weight,
            logsumexp_weight=logsumexp_weight,
            reduction="sum",
        )
        loss_denominator = loss_denominator + jnp.sum(microbatch.loss_weight.astype(jnp.float32))
    cross_entropy = jnp.where(
        loss_denominator != 0,
        cross_entropy_sum / loss_denominator,
        jnp.zeros_like(cross_entropy_sum),
    )
    return cross_entropy + router_loss


def batches_for_pipeline(batch: GrugLmExample, config: GrugMoePipelineConfig):
    """Create the explicit ``[microbatch][stage]`` batch structure JaxPP consumes."""
    microbatches = split_batch_into_microbatches(batch, config.microbatches)

    def copy_for_stage(microbatch):
        return jax.tree.map(
            lambda value: jnp.array(value, copy=True) if isinstance(value, jax.Array) else value,
            microbatch,
        )

    return tuple(tuple(copy_for_stage(microbatch) for _ in range(config.stages)) for microbatch in microbatches)


def stacked_microbatches(batch: GrugLmExample, num_microbatches: int) -> GrugLmExample:
    """Stack a batch on the leading temporal axis consumed by ``jaxpp.treduce``."""
    microbatches = split_batch_into_microbatches(batch, num_microbatches)
    return jax.tree.map(lambda *values: jnp.stack(values), *microbatches)


def _stack_router_metrics(block_metrics: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    keys = (
        "routing_entropy",
        "routing_counts",
        "load_balancing_loss",
        "router_z_loss",
        "qb_beta",
        "capacity_overflow",
    )
    return {f"{key}_per_layer": jnp.stack([metrics[key] for metrics in block_metrics]) for key in keys}


def _apply_qb_betas(stage: GrugMoePipelineStage, qb_betas: jax.Array) -> GrugMoePipelineStage:
    blocks = list(stage.blocks)
    for index, block in enumerate(blocks):
        new_bias = -qb_betas[index]
        new_bias = new_bias - jnp.mean(new_bias)
        new_mlp = eqx.tree_at(lambda mlp: mlp.router_bias, block.mlp, new_bias)
        blocks[index] = eqx.tree_at(lambda current: current.mlp, block, new_mlp)
    return eqx.tree_at(lambda current: current.blocks, stage, tuple(blocks))


def stage_forward_with_pullback(
    params: GrugMoePipelineStage,
    qb_betas: jax.Array,
    hidden: jax.Array,
    batch: GrugLmExample,
    mp_policy: jmp.Policy,
    *,
    router_loss_scale: float,
) -> StageForwardWithPullbackResult:
    """Run a stage once and retain the linearized backward for split scheduling."""

    def forward(stage_params, stage_hidden):
        stage_params = mp_policy.cast_to_compute(_apply_qb_betas(stage_params, qb_betas))
        output, metrics = stage_params.run_blocks(stage_hidden, batch.attn_mask)
        router_loss = stage_params.local_router_loss(metrics) * router_loss_scale
        return (output, router_loss), metrics[_QB_BETA_PER_LAYER_KEY]

    (output, router_loss), pullback, next_qb_betas = jax.vjp(forward, params, hidden, has_aux=True)
    return StageForwardWithPullbackResult(
        output=output,
        next_qb_betas=next_qb_betas,
        router_loss=router_loss,
        pullback=pullback,
    )


def stage_pullback_input_gradient(pullback, hidden_cotangent: jax.Array) -> jax.Array:
    """Evaluate only the activation-gradient output of a reusable stage pullback."""
    router_loss_cotangent = jnp.ones((), dtype=jnp.float32)
    return pullback((hidden_cotangent, router_loss_cotangent))[1]


def stage_pullback_weight_gradient(pullback, hidden_cotangent: jax.Array) -> GrugMoePipelineStage:
    """Evaluate only the parameter-gradient output of a reusable stage pullback."""
    router_loss_cotangent = jnp.ones((), dtype=jnp.float32)
    return pullback((hidden_cotangent, router_loss_cotangent))[0]


def _require_jaxpp():
    if jaxpp is None or mpmd is None:
        raise ModuleNotFoundError("The canonical Grug pipeline requires `uv sync --extra pipeline`.")
    return jaxpp, mpmd


def automatic_stage_to_mpmd_indices(
    config: GrugMoePipelineConfig,
    schedule_name: str,
) -> tuple[int, ...]:
    """Return the physical MPMD rank that owns each logical automatic stage."""
    schedule = _automatic_schedule(config, schedule_name)
    return tuple(int(schedule.get_mpmd_idx(stage_index)) for stage_index in range(config.stages))


def _automatic_schedule(config: GrugMoePipelineConfig, schedule_name: str):
    pp, _ = _require_jaxpp()
    if schedule_name == "zero_bubble":
        return pp.ZeroBubble(num_stages=config.stages)
    if schedule_name == "dualpipe_v":
        return pp.DualPipeV(num_stages=config.stages, mpmd_dim=config.mpmd_stages)
    raise ValueError(f"unknown automatic pipeline schedule: {schedule_name}")


def _is_array(value: object) -> TypeGuard[_ArrayValue]:
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct)):
        return True
    return jaxpp is not None and isinstance(value, jaxpp.MpmdArray)


def _named_sharding_tree(mesh, tree):
    def sharding(value):
        if not _is_array(value):
            return None
        if isinstance(value.sharding, NamedSharding):
            return NamedSharding(mesh, value.sharding.spec)
        return NamedSharding(mesh, P(*([None] * value.ndim)))

    return jax.tree.map(sharding, tree)


def _partition_spec_tree(tree):
    def partition_spec(value):
        if not _is_array(value):
            return None
        if isinstance(value.sharding, NamedSharding):
            return value.sharding.spec
        if jaxpp is not None and isinstance(value.sharding, jaxpp.MpmdSharding):
            return value.sharding.spec
        return P(*([None] * value.ndim))

    return jax.tree.map(partition_spec, tree)


def _mpmd_sharding_tree(mpmd_mesh, stage_index: int, tree):
    pp, _ = _require_jaxpp()

    def sharding(value):
        if not _is_array(value):
            return None
        spec = value.sharding.spec if isinstance(value.sharding, NamedSharding) else P(*([None] * value.ndim))
        return pp.MpmdSharding(mpmd_mesh, mesh_ids={stage_index}, spec=spec)

    return jax.tree.map(sharding, tree)


def _state_named_shardings(mpmd_mesh, state: GrugMoePipelineState) -> GrugMoePipelineState:
    return dataclasses.replace(
        state,
        step=NamedSharding(mpmd_mesh.unstack[0], P()),
        params=tuple(
            _named_sharding_tree(mpmd_mesh.unstack[stage_index], stage) for stage_index, stage in enumerate(state.params)
        ),
        opt_state=tuple(
            _named_sharding_tree(mpmd_mesh.unstack[stage_index], opt_state)
            for stage_index, opt_state in enumerate(state.opt_state)
        ),
        pending_qb_betas=tuple(
            NamedSharding(mpmd_mesh.unstack[stage_index], P(None, None)) for stage_index in range(len(state.params))
        ),
    )


def place_pipeline_batches(mpmd_mesh, batches_by_microbatch, *, memory_threshold: int | None = None):
    """Move every microbatch copy onto the stage that will consume it."""
    pp, _ = _require_jaxpp()
    placed = []
    for stage_batches in batches_by_microbatch:
        placed.append(
            tuple(
                pp.spmd_to_mpmd_reshard(
                    mpmd_mesh,
                    batch,
                    _mpmd_sharding_tree(mpmd_mesh, stage_index, batch),
                    threshold=memory_threshold,
                )
                for stage_index, batch in enumerate(stage_batches)
            )
        )
    return tuple(placed)


@dataclass(frozen=True)
class _LocalExplicitStep:
    lowered: LoweredMpmdFun

    def __call__(self, state: GrugMoePipelineState, batches_by_microbatch):
        flat_args, args_tree = jax.tree.flatten((state, batches_by_microbatch))
        if args_tree != jax.tree.structure(self.lowered.in_shardings):
            raise ValueError("lowered pipeline step received an unexpected input tree")

        local_jaxpr = self.lowered._local_jaxpr
        local_outputs = self.lowered.eval_local(*(flat_args[index] for index in local_jaxpr.global_invar_indices))
        jax.block_until_ready(local_outputs)
        multihost_utils.sync_global_devices("grug_pipeline_step")
        outputs_by_index = dict(zip(local_jaxpr.global_outvar_indices, local_outputs, strict=True))

        local_mesh = self.lowered.mpmd_mesh.unstack[self.lowered.mpmd_mesh.my_mpmd_axis_index]
        local_loss = jax.device_put(jnp.zeros((), dtype=jnp.float32), NamedSharding(local_mesh, P()))
        fallback = (state, {TRAIN_LOSS_KEY: local_loss})
        flat_fallback, fallback_tree = jax.tree.flatten(fallback)
        flat_output_shape, output_tree = jax.tree.flatten(self.lowered.out_shape)
        if fallback_tree != output_tree:
            raise ValueError("lowered pipeline output does not match its fallback tree")
        flat_outputs = [outputs_by_index.get(index, flat_fallback[index]) for index in range(len(flat_output_shape))]
        return jax.tree.unflatten(output_tree, flat_outputs)


def prepare_explicit_step(step, state: GrugMoePipelineState, batches_by_microbatch, mpmd_mesh):
    """Lower to process-local execution when the MPMD mesh spans processes."""
    if not mpmd_mesh.jax_mesh.is_multi_process:
        return step
    return _LocalExplicitStep(step.lower(state, batches_by_microbatch))


def make_automatic_pipeline_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    static_stages: tuple[GrugMoePipelineStage, ...],
    sample_state: GrugMoeAutomaticPipelineState,
    sample_batches: GrugLmExample,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    schedule_name: str = "zero_bubble",
    logsumexp_weight: float | None = None,
):
    """Build a JaxPP automatic ZeroBubble or DualPipeV optimizer step.

    ``static_stages`` and ``sample_state`` must come from the same automatic
    state initializer. ``sample_batches`` must have the leading microbatch axis
    produced by :func:`stacked_microbatches`.

    The stage loop stays model-specific. JaxPP uses the explicit markers to
    place forward work and partition each reverse jaxpr into activation- and
    weight-gradient tasks for the selected schedule.
    """
    pp, _ = _require_jaxpp()
    schedule = _automatic_schedule(config, schedule_name)

    def pipeline_step(
        state: GrugMoeAutomaticPipelineState,
        batches: GrugLmExample,
        loss_denominator: jax.Array,
    ):
        def loss_fn(trainable_stages: tuple[GrugMoePipelineStage, ...], batch: GrugLmExample):
            hidden = None
            router_loss = jnp.array(0.0, dtype=jnp.float32)
            next_qb_betas = []
            last_stage = None
            for stage_index, (trainable_stage, static_stage) in enumerate(
                zip(trainable_stages, static_stages, strict=True)
            ):
                stage = eqx.combine(trainable_stage, static_stage)
                stage = mp_policy.cast_to_compute(_apply_qb_betas(stage, state.pending_qb_betas[stage_index]))
                if stage_index == 0:
                    hidden = stage.embed(batch.tokens)
                assert hidden is not None
                hidden, router_metrics = stage.run_blocks(hidden, batch.attn_mask)
                router_loss = router_loss + stage.local_router_loss(router_metrics) / config.microbatches
                next_qb_betas.append(router_metrics[_QB_BETA_PER_LAYER_KEY])
                if stage_index < config.stages - 1:
                    hidden = pp.mark_stage_end(hidden)
                last_stage = stage

            assert last_stage is not None
            hidden = last_stage.finish(hidden)
            cross_entropy_sum = last_stage.cross_entropy_loss(
                hidden,
                batch.tokens,
                batch.loss_weight,
                logsumexp_weight=logsumexp_weight,
                reduction="sum",
            )
            loss = cross_entropy_sum / loss_denominator + router_loss
            loss = pp.mark_stage_end(loss)
            return loss, tuple(next_qb_betas)

        (loss, next_qb_betas), grads = pp.treduce(
            lambda batch: jax.value_and_grad(loss_fn, has_aux=True)(state.trainable_params, batch),
            batches,
            schedule=schedule,
            operation=((pp.Add, tuple(pp.Add for _ in range(config.stages))), pp.Add),
        )
        next_params = []
        next_opt_state = []
        for params, opt_state, stage_grads in zip(
            state.trainable_params,
            state.opt_state,
            grads,
            strict=True,
        ):
            updates, stage_opt_state = optimizer.update(stage_grads, opt_state, params)
            next_params.append(eqx.apply_updates(params, updates))
            next_opt_state.append(stage_opt_state)
        next_state = dataclasses.replace(
            state,
            step=state.step + jnp.array(1, dtype=state.step.dtype),
            trainable_params=tuple(next_params),
            opt_state=tuple(next_opt_state),
            pending_qb_betas=tuple(beta / config.microbatches for beta in next_qb_betas),
        )
        return next_state, {TRAIN_LOSS_KEY: loss}

    return pp.mpmd_jit_with_loop(
        pipeline_step,
        mpmd_mesh=mpmd_mesh,
        in_specs=(_partition_spec_tree(sample_state), _partition_spec_tree(sample_batches), P()),
        out_specs=(_partition_spec_tree(sample_state), {TRAIN_LOSS_KEY: P()}),
    )


def prepare_automatic_mpmd_step(
    step,
    state: GrugMoeAutomaticPipelineState,
    batches: GrugLmExample,
    loss_denominator: jax.Array,
    mpmd_mesh,
    *,
    memory_threshold: int | None = None,
) -> PreparedAutomaticMpmdStep:
    """Compile with stage-local state and place only the still-SPMD batch inputs."""
    pp, _ = _require_jaxpp()
    compiled = step.compile(state, batches, loss_denominator)
    args_shardings, kwargs_shardings = compiled.in_shardings
    if kwargs_shardings:
        raise ValueError("automatic pipeline step does not accept keyword arguments")

    def place_initial_scalar(value, target):
        if not _is_array(value) or isinstance(value, pp.MpmdArray):
            return value
        if value.shape != ():
            return value
        mesh_ids = target.mesh_ids or {0}
        local_arrays = []
        for stage_index in sorted(mesh_ids):
            sharding = NamedSharding(mpmd_mesh.unstack[stage_index], target.spec)
            if _process_has_sharding(sharding):
                local_arrays.append(jax.device_put(np.zeros((), dtype=value.dtype), sharding))
        return pp.MpmdArray(
            local_arrays,
            target,
            shape=(),
            dtype=value.dtype,
        )

    state = jax.tree.map(place_initial_scalar, state, args_shardings[0])
    if memory_threshold is None and all(device.memory_stats() is None for device in jax.local_devices()):
        memory_threshold = 0
    batches, loss_denominator = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        (batches, loss_denominator),
        (args_shardings[1], args_shardings[2]),
        threshold=memory_threshold,
    )
    return PreparedAutomaticMpmdStep(
        step=compiled,
        state=state,
        batches=batches,
        loss_denominator=loss_denominator,
    )


def make_explicit_1f1b_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    sample_state: GrugMoePipelineState,
    sample_batches,
    logsumexp_weight: float | None = None,
):
    """Build the explicit JaxPP MPMD standard-1F1B optimizer step.

    ``sample_state`` must come from :func:`make_mpmd_pipeline_state`, and
    ``sample_batches`` must be placed with :func:`place_pipeline_batches`.
    """
    return _make_explicit_step(
        optimizer,
        mp_policy,
        config=config,
        mpmd_mesh=mpmd_mesh,
        sample_state=sample_state,
        sample_batches=sample_batches,
        logsumexp_weight=logsumexp_weight,
        zero_bubble_schedules=None,
    )


def make_explicit_zero_bubble_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    sample_state: GrugMoePipelineState,
    sample_batches,
    logsumexp_weight: float | None = None,
):
    """Build an explicit JaxPP zero-bubble optimizer step with reusable VJP residuals."""
    pp, _ = _require_jaxpp()
    schedules = pp.ZeroBubble(num_stages=config.stages).tasks(config.microbatches)
    planner_schedules = tuple(
        tuple((task.fwd_or_bwd.name, task.mubatch_idx) for task in stage_tasks) for stage_tasks in schedules
    )
    return _make_explicit_step(
        optimizer,
        mp_policy,
        config=config,
        mpmd_mesh=mpmd_mesh,
        sample_state=sample_state,
        sample_batches=sample_batches,
        logsumexp_weight=logsumexp_weight,
        zero_bubble_schedules=planner_schedules,
    )


def _make_explicit_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    sample_state: GrugMoePipelineState,
    sample_batches,
    logsumexp_weight: float | None,
    zero_bubble_schedules: tuple[tuple[tuple[str, int], ...], ...] | None,
):
    _, explicit_mpmd = _require_jaxpp()
    num_stages = len(sample_state.params)
    if num_stages != config.stages:
        raise ValueError(f"config has {config.stages} stages but state has {num_stages}")
    if len(sample_batches) != config.microbatches:
        raise ValueError(f"expected {config.microbatches} microbatches, got {len(sample_batches)}")
    for microbatch_index, stage_batches in enumerate(sample_batches):
        if len(stage_batches) != num_stages:
            raise ValueError(
                f"microbatch {microbatch_index} has {len(stage_batches)} stage batches; expected {num_stages}"
            )

    activation_shardings = tuple(
        NamedSharding(mpmd_mesh.unstack[stage_index], P(BATCH_AXES, None, None)) for stage_index in range(num_stages)
    )
    qb_shardings = tuple(
        NamedSharding(mpmd_mesh.unstack[stage_index], P(None, None)) for stage_index in range(num_stages)
    )
    router_loss_shardings = tuple(
        NamedSharding(mpmd_mesh.unstack[stage_index], P()) for stage_index in range(num_stages)
    )
    loss_shardings = (
        NamedSharding(mpmd_mesh.unstack[0], P()),
        NamedSharding(mpmd_mesh.unstack[-1], P()),
    )
    param_shardings = tuple(
        _named_sharding_tree(mpmd_mesh.unstack[stage_index], params)
        for stage_index, params in enumerate(sample_state.params)
    )
    opt_state_shardings = tuple(
        _named_sharding_tree(mpmd_mesh.unstack[stage_index], opt_state)
        for stage_index, opt_state in enumerate(sample_state.opt_state)
    )
    batch_shardings = tuple(
        tuple(
            _named_sharding_tree(mpmd_mesh.unstack[stage_index], batch)
            for stage_index, batch in enumerate(stage_batches)
        )
        for stage_batches in sample_batches
    )
    input_shardings = (_state_named_shardings(mpmd_mesh, sample_state), batch_shardings)

    def stage0_forward(params: GrugMoePipelineStage, qb_betas: jax.Array, batch: GrugLmExample):
        params = mp_policy.cast_to_compute(_apply_qb_betas(params, qb_betas))
        hidden = params.embed(batch.tokens)
        hidden, metrics = params.run_blocks(hidden, batch.attn_mask)
        return hidden, metrics[_QB_BETA_PER_LAYER_KEY], params.local_router_loss(metrics) / config.microbatches

    def stage_forward(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        hidden: jax.Array,
        batch: GrugLmExample,
    ):
        params = mp_policy.cast_to_compute(_apply_qb_betas(params, qb_betas))
        hidden, metrics = params.run_blocks(hidden, batch.attn_mask)
        return hidden, metrics[_QB_BETA_PER_LAYER_KEY], params.local_router_loss(metrics) / config.microbatches

    def stage0_backward(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        batch: GrugLmExample,
        hidden_cotangent: jax.Array,
    ):
        def projected_loss(stage_params):
            stage_params = mp_policy.cast_to_compute(_apply_qb_betas(stage_params, qb_betas))
            hidden = stage_params.embed(batch.tokens)
            hidden, metrics = stage_params.run_blocks(hidden, batch.attn_mask)
            activation_term = jnp.sum(hidden.astype(jnp.float32) * hidden_cotangent.astype(jnp.float32))
            return activation_term + stage_params.local_router_loss(metrics) / config.microbatches

        return jax.grad(projected_loss)(params)

    def stage_backward(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        hidden: jax.Array,
        batch: GrugLmExample,
        hidden_cotangent: jax.Array,
    ):
        def projected_loss(stage_params, stage_hidden):
            stage_params = mp_policy.cast_to_compute(_apply_qb_betas(stage_params, qb_betas))
            stage_hidden, metrics = stage_params.run_blocks(stage_hidden, batch.attn_mask)
            activation_term = jnp.sum(stage_hidden.astype(jnp.float32) * hidden_cotangent.astype(jnp.float32))
            return activation_term + stage_params.local_router_loss(metrics) / config.microbatches

        return jax.grad(projected_loss, argnums=(0, 1))(params, hidden)

    def last_stage_loss_and_grads(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        hidden: jax.Array,
        batch: GrugLmExample,
        loss_denominator: jax.Array,
    ):
        def loss_fn(stage_params, stage_hidden):
            stage_params = mp_policy.cast_to_compute(_apply_qb_betas(stage_params, qb_betas))
            stage_hidden, metrics = stage_params.run_blocks(stage_hidden, batch.attn_mask)
            stage_hidden = stage_params.finish(stage_hidden)
            loss = stage_params.cross_entropy_loss(
                stage_hidden,
                batch.tokens,
                batch.loss_weight,
                logsumexp_weight=logsumexp_weight,
                reduction="sum",
            )
            loss = jnp.where(loss_denominator != 0, loss / loss_denominator, jnp.zeros_like(loss))
            loss = loss + stage_params.local_router_loss(metrics) / config.microbatches
            return loss, metrics[_QB_BETA_PER_LAYER_KEY]

        (loss, next_qb_betas), (grads, hidden_cotangent) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        return loss, next_qb_betas, grads, hidden_cotangent

    def last_stage_loss_with_pullback(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        hidden: jax.Array,
        batch: GrugLmExample,
        loss_denominator: jax.Array,
    ):
        def loss_fn(stage_params, stage_hidden):
            stage_params = mp_policy.cast_to_compute(_apply_qb_betas(stage_params, qb_betas))
            stage_hidden, metrics = stage_params.run_blocks(stage_hidden, batch.attn_mask)
            stage_hidden = stage_params.finish(stage_hidden)
            loss = stage_params.cross_entropy_loss(
                stage_hidden,
                batch.tokens,
                batch.loss_weight,
                logsumexp_weight=logsumexp_weight,
                reduction="sum",
            )
            loss = jnp.where(loss_denominator != 0, loss / loss_denominator, jnp.zeros_like(loss))
            return loss + stage_params.local_router_loss(metrics) / config.microbatches, metrics[_QB_BETA_PER_LAYER_KEY]

        loss, pullback, next_qb_betas = jax.vjp(loss_fn, params, hidden, has_aux=True)
        return loss, next_qb_betas, pullback

    def last_stage_pullback_input_gradient(pullback):
        return pullback(jnp.ones((), dtype=jnp.float32))[1]

    def last_stage_pullback_weight_gradient(pullback):
        return pullback(jnp.ones((), dtype=jnp.float32))[0]

    def update_stage(params, opt_state, grads):
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), next_opt_state

    def add_trees(left, right):
        return jax.tree.map(lambda x, y: x + y, left, right)

    def average_tree(tree):
        scale = jnp.asarray(1.0 / config.microbatches, dtype=jnp.float32)
        return jax.tree.map(lambda value: value * scale, tree)

    def loss_weight_sum(batch: GrugLmExample):
        return jnp.sum(batch.loss_weight.astype(jnp.float32))

    pullback_trees = None
    pullback_shardings = None
    if zero_bubble_schedules is not None:
        microbatch_size, sequence_length = sample_batches[0][0].tokens.shape
        hidden_shape = (microbatch_size, sequence_length, sample_state.params[0].config.hidden_dim)
        pullback_shapes = []
        pullback_shapes.append(None)
        for stage_index in range(1, num_stages):
            hidden = jax.ShapeDtypeStruct(
                hidden_shape,
                mp_policy.compute_dtype,
                sharding=activation_shardings[stage_index],
            )
            stage_mesh = mpmd_mesh.unstack[stage_index]
            with jax.set_mesh(stage_mesh):
                if stage_index == num_stages - 1:
                    denominator = jax.ShapeDtypeStruct((), jnp.float32, sharding=loss_shardings[1])
                    shape = jax.eval_shape(
                        last_stage_loss_with_pullback,
                        sample_state.params[stage_index],
                        sample_state.pending_qb_betas[stage_index],
                        hidden,
                        sample_batches[0][stage_index],
                        denominator,
                    )[2]
                else:
                    shape = jax.eval_shape(
                        lambda params, qb_betas, stage_hidden, batch: stage_forward_with_pullback(
                            params,
                            qb_betas,
                            stage_hidden,
                            batch,
                            mp_policy,
                            router_loss_scale=1.0 / config.microbatches,
                        ),
                        sample_state.params[stage_index],
                        sample_state.pending_qb_betas[stage_index],
                        hidden,
                        sample_batches[0][stage_index],
                    ).pullback
            pullback_shapes.append(shape)
        # JAX's VJP pytree metadata embeds a trace-specific closed jaxpr. Only
        # residual arrays can safely cross an independently traced JaxPP task.
        flattened_pullbacks = tuple(jax.tree.flatten(shape) for shape in pullback_shapes)
        pullback_trees = tuple(tree for _, tree in flattened_pullbacks)
        pullback_shardings = tuple(
            tuple(_named_sharding_tree(mpmd_mesh.unstack[stage_index], leaf) for leaf in leaves)
            for stage_index, (leaves, _) in enumerate(flattened_pullbacks)
        )

    loss_weight_sum_task = explicit_mpmd.task(
        loss_weight_sum,
        name="grug_1f1b_loss_weight_sum",
        out_shardings=loss_shardings[1],
    )
    loss_weight_accumulator = explicit_mpmd.task(
        add_trees,
        name="grug_1f1b_accumulate_loss_weight",
        out_shardings=loss_shardings[1],
    )
    stage0_forward_task = explicit_mpmd.task(
        stage0_forward,
        name="grug_1f1b_stage0_forward",
        out_shardings=(activation_shardings[0], qb_shardings[0], router_loss_shardings[0]),
    )
    stage_forward_tasks = tuple(
        explicit_mpmd.task(
            stage_forward,
            name=f"grug_1f1b_stage{stage_index}_forward",
            out_shardings=(
                activation_shardings[stage_index],
                qb_shardings[stage_index],
                router_loss_shardings[stage_index],
            ),
        )
        for stage_index in range(1, num_stages - 1)
    )
    qb_accumulators = tuple(
        explicit_mpmd.task(
            add_trees,
            name=f"grug_1f1b_stage{stage_index}_accumulate_qb",
            out_shardings=qb_shardings[stage_index],
        )
        for stage_index in range(num_stages)
    )
    local_router_loss_accumulators = tuple(
        explicit_mpmd.task(
            add_trees,
            name=f"grug_1f1b_stage{stage_index}_accumulate_router_loss",
            out_shardings=router_loss_shardings[stage_index],
        )
        for stage_index in range(num_stages - 1)
    )
    final_router_loss_accumulator = explicit_mpmd.task(
        add_trees,
        name="grug_1f1b_accumulate_transferred_router_loss",
        out_shardings=loss_shardings[1],
    )
    last_stage_loss_and_grads_task = explicit_mpmd.task(
        last_stage_loss_and_grads,
        name=f"grug_1f1b_stage{num_stages - 1}_loss_backward",
        out_shardings=(
            loss_shardings[1],
            qb_shardings[-1],
            param_shardings[-1],
            activation_shardings[-1],
        ),
    )
    loss_accumulator = explicit_mpmd.task(
        add_trees,
        name="grug_1f1b_accumulate_loss",
        out_shardings=loss_shardings[1],
    )
    stage0_backward_task = explicit_mpmd.task(
        stage0_backward,
        name="grug_1f1b_stage0_backward",
        out_shardings=param_shardings[0],
    )
    stage_backward_tasks = tuple(
        explicit_mpmd.task(
            stage_backward,
            name=f"grug_1f1b_stage{stage_index}_backward",
            out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
        )
        for stage_index in range(1, num_stages - 1)
    )
    gradient_accumulators = tuple(
        explicit_mpmd.task(
            add_trees,
            name=f"grug_1f1b_stage{stage_index}_accumulate_grads",
            out_shardings=param_shardings[stage_index],
        )
        for stage_index in range(num_stages)
    )
    zero_bubble_tasks = None
    if pullback_shardings is not None and pullback_trees is not None:

        def flattened_stage_forward(params, qb_betas, hidden, batch):
            result = stage_forward_with_pullback(
                params,
                qb_betas,
                hidden,
                batch,
                mp_policy,
                router_loss_scale=1.0 / config.microbatches,
            )
            return (
                result.output,
                result.next_qb_betas,
                result.router_loss,
                tuple(jax.tree.leaves(result.pullback)),
            )

        def flattened_last_stage_forward(params, qb_betas, hidden, batch, loss_denominator):
            loss, next_betas, pullback = last_stage_loss_with_pullback(params, qb_betas, hidden, batch, loss_denominator)
            return loss, next_betas, tuple(jax.tree.leaves(pullback))

        def pullback_consumer(stage_index: int, function):
            pullback_tree = pullback_trees[stage_index]

            def consume(residuals, *args):
                return function(jax.tree.unflatten(pullback_tree, residuals), *args)

            return consume

        zero_bubble_tasks = _ExplicitZeroBubbleTasks(
            stage_forwards=tuple(
                explicit_mpmd.task(
                    flattened_stage_forward,
                    name=f"grug_zb_stage{stage_index}_forward",
                    out_shardings=(
                        activation_shardings[stage_index],
                        qb_shardings[stage_index],
                        router_loss_shardings[stage_index],
                        pullback_shardings[stage_index],
                    ),
                )
                for stage_index in range(1, num_stages - 1)
            ),
            last_forward=explicit_mpmd.task(
                flattened_last_stage_forward,
                name=f"grug_zb_stage{num_stages - 1}_loss_forward",
                out_shardings=(loss_shardings[1], qb_shardings[-1], pullback_shardings[-1]),
            ),
            stage0_weight=explicit_mpmd.task(
                stage0_backward,
                name="grug_zb_stage0_backward_weight",
                out_shardings=param_shardings[0],
            ),
            input_gradients=tuple(
                explicit_mpmd.task(
                    pullback_consumer(
                        stage_index,
                        (
                            last_stage_pullback_input_gradient
                            if stage_index == num_stages - 1
                            else stage_pullback_input_gradient
                        ),
                    ),
                    name=f"grug_zb_stage{stage_index}_backward_input",
                    out_shardings=activation_shardings[stage_index],
                )
                for stage_index in range(1, num_stages)
            ),
            weight_gradients=tuple(
                explicit_mpmd.task(
                    pullback_consumer(
                        stage_index,
                        (
                            last_stage_pullback_weight_gradient
                            if stage_index == num_stages - 1
                            else stage_pullback_weight_gradient
                        ),
                    ),
                    name=f"grug_zb_stage{stage_index}_backward_weight",
                    out_shardings=param_shardings[stage_index],
                )
                for stage_index in range(1, num_stages)
            ),
        )

    def accumulate(accumulated, value, task):
        if accumulated is None:
            return value
        return task(accumulated, value)

    @explicit_mpmd.mpmd(
        mpmd_mesh,
        in_shardings=input_shardings,
        donate_argnums=(0,),
        infer_donation=True,
    )
    def pipeline_step(state: GrugMoePipelineState, batches_by_microbatch):
        params = list(state.params)
        opt_state = list(state.opt_state)
        next_qb_betas = [None] * num_stages
        accumulated_grads = [None] * num_stages
        accumulated_loss = None
        accumulated_forward_router_losses = [None] * (num_stages - 1)
        stage_inputs = {}
        stage_pullbacks = {}
        stage_output_cotangents = {}
        forward_transfers = {}
        backward_transfers = {}
        completed_forwards = set()
        completed_backwards = set()
        completed_input_backwards = set()
        completed_weight_backwards = set()

        loss_denominator = None
        for stage_batches in batches_by_microbatch:
            microbatch_denominator = loss_weight_sum_task(stage_batches[-1])
            loss_denominator = accumulate(
                loss_denominator,
                microbatch_denominator,
                loss_weight_accumulator,
            )
        if loss_denominator is None:
            raise ValueError("1F1B did not accumulate a loss denominator")

        def ensure_forward(stage_index: int, microbatch_index: int) -> None:
            key = (stage_index, microbatch_index)
            if key in completed_forwards:
                return
            stage_batches = batches_by_microbatch[microbatch_index]
            if stage_index == 0:
                hidden, qb_betas, router_loss = stage0_forward_task(
                    params[0], state.pending_qb_betas[0], stage_batches[0]
                )
                next_qb_betas[0] = accumulate(
                    next_qb_betas[0],
                    qb_betas,
                    qb_accumulators[0],
                )
                accumulated_forward_router_losses[0] = accumulate(
                    accumulated_forward_router_losses[0],
                    router_loss,
                    local_router_loss_accumulators[0],
                )
                forward_transfers[(1, microbatch_index)] = explicit_mpmd.transfer(
                    hidden,
                    out_shardings=activation_shardings[1],
                )
                completed_forwards.add(key)
                return

            ensure_forward(stage_index - 1, microbatch_index)
            hidden = forward_transfers[key].done()
            stage_inputs[key] = hidden
            if stage_index == num_stages - 1:
                completed_forwards.add(key)
                return

            if zero_bubble_tasks is None:
                hidden, qb_betas, router_loss = stage_forward_tasks[stage_index - 1](
                    params[stage_index], state.pending_qb_betas[stage_index], hidden, stage_batches[stage_index]
                )
            else:
                hidden, qb_betas, router_loss, pullback = zero_bubble_tasks.stage_forwards[stage_index - 1](
                    params[stage_index], state.pending_qb_betas[stage_index], hidden, stage_batches[stage_index]
                )
                stage_pullbacks[key] = pullback
            next_qb_betas[stage_index] = accumulate(
                next_qb_betas[stage_index],
                qb_betas,
                qb_accumulators[stage_index],
            )
            accumulated_forward_router_losses[stage_index] = accumulate(
                accumulated_forward_router_losses[stage_index],
                router_loss,
                local_router_loss_accumulators[stage_index],
            )
            forward_transfers[(stage_index + 1, microbatch_index)] = explicit_mpmd.transfer(
                hidden,
                out_shardings=activation_shardings[stage_index + 1],
            )
            completed_forwards.add(key)

        def ensure_backward(stage_index: int, microbatch_index: int) -> None:
            nonlocal accumulated_loss
            key = (stage_index, microbatch_index)
            if key in completed_backwards:
                return
            stage_batches = batches_by_microbatch[microbatch_index]
            if stage_index == num_stages - 1:
                ensure_forward(stage_index, microbatch_index)
                loss, qb_betas, grads, hidden_cotangent = last_stage_loss_and_grads_task(
                    params[stage_index],
                    state.pending_qb_betas[stage_index],
                    stage_inputs[key],
                    stage_batches[stage_index],
                    loss_denominator,
                )
                accumulated_loss = accumulate(
                    accumulated_loss,
                    loss,
                    loss_accumulator,
                )
                next_qb_betas[stage_index] = accumulate(
                    next_qb_betas[stage_index],
                    qb_betas,
                    qb_accumulators[stage_index],
                )
                accumulated_grads[stage_index] = accumulate(
                    accumulated_grads[stage_index],
                    grads,
                    gradient_accumulators[stage_index],
                )
                backward_transfers[(stage_index - 1, microbatch_index)] = explicit_mpmd.transfer(
                    hidden_cotangent,
                    out_shardings=activation_shardings[stage_index - 1],
                )
                completed_backwards.add(key)
                return

            ensure_backward(stage_index + 1, microbatch_index)
            hidden_cotangent = backward_transfers[key].done()
            if stage_index == 0:
                grads = stage0_backward_task(params[0], state.pending_qb_betas[0], stage_batches[0], hidden_cotangent)
                accumulated_grads[0] = accumulate(
                    accumulated_grads[0],
                    grads,
                    gradient_accumulators[0],
                )
                completed_backwards.add(key)
                return

            ensure_forward(stage_index, microbatch_index)
            grads, hidden_cotangent = stage_backward_tasks[stage_index - 1](
                params[stage_index],
                state.pending_qb_betas[stage_index],
                stage_inputs[key],
                stage_batches[stage_index],
                hidden_cotangent,
            )
            accumulated_grads[stage_index] = accumulate(
                accumulated_grads[stage_index],
                grads,
                gradient_accumulators[stage_index],
            )
            backward_transfers[(stage_index - 1, microbatch_index)] = explicit_mpmd.transfer(
                hidden_cotangent,
                out_shardings=activation_shardings[stage_index - 1],
            )
            completed_backwards.add(key)

        def ensure_input_backward(stage_index: int, microbatch_index: int) -> None:
            nonlocal accumulated_loss
            if zero_bubble_tasks is None:
                raise ValueError("zero-bubble tasks were not initialized")
            key = (stage_index, microbatch_index)
            if key in completed_input_backwards:
                return
            stage_batches = batches_by_microbatch[microbatch_index]
            if stage_index == num_stages - 1:
                ensure_forward(stage_index, microbatch_index)
                loss, qb_betas, pullback = zero_bubble_tasks.last_forward(
                    params[stage_index],
                    state.pending_qb_betas[stage_index],
                    stage_inputs[key],
                    stage_batches[stage_index],
                    loss_denominator,
                )
                stage_pullbacks[key] = pullback
                accumulated_loss = accumulate(accumulated_loss, loss, loss_accumulator)
                next_qb_betas[stage_index] = accumulate(
                    next_qb_betas[stage_index], qb_betas, qb_accumulators[stage_index]
                )
                hidden_cotangent = zero_bubble_tasks.input_gradients[stage_index - 1](pullback)
                backward_transfers[(stage_index - 1, microbatch_index)] = explicit_mpmd.transfer(
                    hidden_cotangent,
                    out_shardings=activation_shardings[stage_index - 1],
                )
                completed_input_backwards.add(key)
                return

            ensure_input_backward(stage_index + 1, microbatch_index)
            hidden_cotangent = backward_transfers[key].done()
            if stage_index == 0:
                grads = zero_bubble_tasks.stage0_weight(
                    params[0],
                    state.pending_qb_betas[0],
                    stage_batches[0],
                    hidden_cotangent,
                )
                accumulated_grads[0] = accumulate(accumulated_grads[0], grads, gradient_accumulators[0])
                completed_input_backwards.add(key)
                completed_weight_backwards.add(key)
                return

            ensure_forward(stage_index, microbatch_index)
            stage_output_cotangents[key] = hidden_cotangent
            input_cotangent = zero_bubble_tasks.input_gradients[stage_index - 1](stage_pullbacks[key], hidden_cotangent)
            backward_transfers[(stage_index - 1, microbatch_index)] = explicit_mpmd.transfer(
                input_cotangent,
                out_shardings=activation_shardings[stage_index - 1],
            )
            completed_input_backwards.add(key)

        def ensure_weight_backward(stage_index: int, microbatch_index: int) -> None:
            if zero_bubble_tasks is None:
                raise ValueError("zero-bubble tasks were not initialized")
            key = (stage_index, microbatch_index)
            if key in completed_weight_backwards:
                return
            ensure_input_backward(stage_index, microbatch_index)
            if stage_index == 0:
                return
            if stage_index == num_stages - 1:
                grads = zero_bubble_tasks.weight_gradients[stage_index - 1](stage_pullbacks[key])
            else:
                grads = zero_bubble_tasks.weight_gradients[stage_index - 1](
                    stage_pullbacks[key], stage_output_cotangents[key]
                )
            accumulated_grads[stage_index] = accumulate(
                accumulated_grads[stage_index], grads, gradient_accumulators[stage_index]
            )
            completed_weight_backwards.add(key)

        if zero_bubble_schedules is None:
            schedules = tuple(
                standard_1f1b_stage_schedule(
                    num_stages=num_stages,
                    num_microbatches=config.microbatches,
                    stage_index=stage_index,
                )
                for stage_index in range(num_stages)
            )
            for task_index in range(2 * config.microbatches):
                for stage_index, schedule in enumerate(schedules):
                    task = schedule[task_index]
                    if task.direction is PipelineDirection.FORWARD:
                        ensure_forward(stage_index, task.microbatch)
                    else:
                        ensure_backward(stage_index, task.microbatch)
        else:
            for task_index in range(max(len(schedule) for schedule in zero_bubble_schedules)):
                for stage_index, schedule in enumerate(zero_bubble_schedules):
                    if task_index >= len(schedule):
                        continue
                    task_type, microbatch_index = schedule[task_index]
                    if task_type == "FWD":
                        ensure_forward(stage_index, microbatch_index)
                    elif task_type == "BWD_I":
                        ensure_input_backward(stage_index, microbatch_index)
                    elif task_type == "BWD_W":
                        ensure_weight_backward(stage_index, microbatch_index)
                    else:
                        raise ValueError(f"unexpected zero-bubble task type: {task_type}")

        if accumulated_loss is None:
            raise ValueError("1F1B did not accumulate a loss")
        if any(router_loss is None for router_loss in accumulated_forward_router_losses):
            raise ValueError("1F1B did not accumulate every forward-stage router loss")
        accumulated_forward_router_loss = None
        for router_loss in accumulated_forward_router_losses:
            transferred_router_loss = explicit_mpmd.transfer(router_loss, out_shardings=loss_shardings[1]).done()
            accumulated_forward_router_loss = accumulate(
                accumulated_forward_router_loss,
                transferred_router_loss,
                final_router_loss_accumulator,
            )
        if accumulated_forward_router_loss is None:
            raise ValueError("1F1B did not transfer forward-stage router loss")
        loss = explicit_mpmd.task(
            add_trees,
            name="grug_1f1b_add_forward_router_loss",
            out_shardings=loss_shardings[1],
        )(accumulated_loss, accumulated_forward_router_loss)
        averaged_qb_betas = tuple(
            explicit_mpmd.task(
                average_tree,
                name=f"grug_1f1b_stage{stage_index}_average_qb",
                out_shardings=qb_shardings[stage_index],
            )(stage_qb_betas)
            for stage_index, stage_qb_betas in enumerate(next_qb_betas)
        )
        for stage_index in range(num_stages):
            params[stage_index], opt_state[stage_index] = explicit_mpmd.task(
                update_stage,
                name=f"grug_1f1b_stage{stage_index}_update",
                out_shardings=(param_shardings[stage_index], opt_state_shardings[stage_index]),
            )(params[stage_index], opt_state[stage_index], accumulated_grads[stage_index])

        next_step = explicit_mpmd.task(
            lambda step: step + jnp.array(1, dtype=step.dtype),
            name="grug_1f1b_increment_step",
            out_shardings=loss_shardings[0],
        )(state.step)
        stage0_loss = explicit_mpmd.transfer(loss, out_shardings=loss_shardings[0]).done()
        next_state = dataclasses.replace(
            state,
            step=next_step,
            params=tuple(params),
            opt_state=tuple(opt_state),
            pending_qb_betas=averaged_qb_betas,
        )
        return next_state, {TRAIN_LOSS_KEY: stage0_loss}

    return pipeline_step
