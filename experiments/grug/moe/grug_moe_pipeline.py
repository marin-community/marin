# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic pipeline-parallel training for Grug MoE."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeGuard

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.jax_utils import named_call
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.pipeline import evenly_partition_layers, split_batch_into_microbatches

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
except ModuleNotFoundError:
    jaxpp = None
    mpmd = None


TRAIN_LOSS_KEY = "train/loss"
_QB_BETA_PER_LAYER_KEY = "qb_beta_per_layer"
_PIPELINE_AXIS = "pipeline"

type _ArrayValue = jax.Array | jax.ShapeDtypeStruct | jaxpp.MpmdArray


class AutomaticPipelineSchedule(StrEnum):
    ZERO_BUBBLE = "zero_bubble"
    DUALPIPE_V = "dualpipe_v"


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
    pp, _ = _jaxpp_modules()
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
    axis_names = (_PIPELINE_AXIS, "replica_dcn", "data", "expert", "model")
    devices = np.asarray(jax.devices(), dtype=object).reshape(shape)
    mesh = Mesh(devices, axis_names, axis_types=(AxisType.Explicit,) * len(axis_names))
    if mesh.is_multi_process:
        local_stages = {int(np.argwhere(devices == device)[0][0]) for device in jax.local_devices()}
        if len(local_stages) != 1:
            raise ValueError(f"each JAX process must own exactly one pipeline stage; got {sorted(local_stages)}")
    return mesh, pp.MpmdMesh(mesh, _PIPELINE_AXIS)


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
class GrugMoeAutomaticPipelineState:
    """Array state for JaxPP's automatic pipeline transform."""

    step: jax.Array
    trainable_params: tuple[GrugMoePipelineStage, ...]
    opt_state: tuple[optax.OptState, ...]
    pending_qb_betas: tuple[jax.Array, ...]


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


def split_transformer(
    model: Transformer,
    num_stages: int,
    *,
    layer_counts: tuple[int, ...] | None = None,
) -> tuple[GrugMoePipelineStage, ...]:
    """Split a Grug MoE transformer into contiguous stage pytrees."""
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
    """Partition automatic-schedule stages into trainable and static pytrees.

    Returns:
        The trainable and static stage tuples. Router biases remain static
        because the pending QB update supplies them separately.
    """
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
    pp, _ = _jaxpp_modules()
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
    """Return automatic state and static stages using the requested placement."""
    pp, _ = _jaxpp_modules()
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
    """Return the loss from sequential execution of split stages."""
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
    """Return the loss after sequential microbatch accumulation."""
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


def _jaxpp_modules():
    if jaxpp is None or mpmd is None:
        raise ModuleNotFoundError("The canonical Grug pipeline requires `uv sync --extra pipeline`.")
    return jaxpp, mpmd


def automatic_stage_to_mpmd_indices(
    config: GrugMoePipelineConfig,
    schedule_name: AutomaticPipelineSchedule,
) -> tuple[int, ...]:
    """Return the physical MPMD rank that owns each logical automatic stage."""
    schedule = _automatic_schedule(config, schedule_name)
    return tuple(int(schedule.get_mpmd_idx(stage_index)) for stage_index in range(config.stages))


def _automatic_schedule(config: GrugMoePipelineConfig, schedule_name: AutomaticPipelineSchedule):
    pp, _ = _jaxpp_modules()
    if schedule_name == AutomaticPipelineSchedule.ZERO_BUBBLE:
        return pp.ZeroBubble(num_stages=config.stages)
    if schedule_name == AutomaticPipelineSchedule.DUALPIPE_V:
        return pp.DualPipeV(num_stages=config.stages, mpmd_dim=config.mpmd_stages)
    raise ValueError(f"unknown automatic pipeline schedule: {schedule_name}")


def _is_array(value: object) -> TypeGuard[_ArrayValue]:
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct)):
        return True
    return jaxpp is not None and isinstance(value, jaxpp.MpmdArray)


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
    pp, _ = _jaxpp_modules()

    def sharding(value):
        if not _is_array(value):
            return None
        spec = value.sharding.spec if isinstance(value.sharding, NamedSharding) else P(*([None] * value.ndim))
        return pp.MpmdSharding(mpmd_mesh, mesh_ids={stage_index}, spec=spec)

    return jax.tree.map(sharding, tree)


def make_automatic_pipeline_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    static_stages: tuple[GrugMoePipelineStage, ...],
    sample_state: GrugMoeAutomaticPipelineState,
    sample_batches: GrugLmExample,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    schedule_name: AutomaticPipelineSchedule = AutomaticPipelineSchedule.ZERO_BUBBLE,
    logsumexp_weight: float | None = None,
):
    """Build a JaxPP automatic ZeroBubble or DualPipeV optimizer step.

    ``static_stages`` and ``sample_state`` must come from the same automatic
    state initializer. ``sample_batches`` must have the leading microbatch axis
    produced by :func:`stacked_microbatches`.

    """
    pp, _ = _jaxpp_modules()
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
    pp, _ = _jaxpp_modules()
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
