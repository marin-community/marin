# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAXPP prototype using the full hero model blocks and routing semantics."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
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
from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES, reduce_moe_routing_stats
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.pipeline import evenly_partition_layers

from experiments.grug.moe_hero_ep.model import (
    _BATCH_AXES as BATCH_AXES,
)
from experiments.grug.moe_hero_ep.model import (
    _CE_BLOCK_SIZES,
    _EMBED_PARTITION_SPEC,
    _LM_HEAD_PARTITION_SPEC,
    LAYER_CARRY_REMAT_NAME,
    OFFLOAD_CARRY_REMAT_MODE,
    Block,
    GatedNorm,
    GrugModelConfig,
    RMSNorm,
    Transformer,
    _batch_reshard,
    _embedding_gather,
    _init_weight,
    _unstacked_blocks,
)
from experiments.grug.moe_hero_ep.train import _tree_to_memory_kind

try:
    import jaxpp.api as jaxpp
except ModuleNotFoundError as error:
    if error.name != "jaxpp":
        raise
    jaxpp = None

if jaxpp is None:
    mpmd = None
else:
    from jaxpp import jax_primitives as mpmd_primitives
    from jaxpp.experimental import mpmd


TRAIN_LOSS_KEY = "train/loss"
_QB_BETA_PER_LAYER_KEY = "qb_beta_per_layer"
_PIPELINE_AXIS = "pipeline"
_HOST_MEMORY_KIND = "pinned_host"

type _ArrayValue = jax.Array | jax.ShapeDtypeStruct | jaxpp.MpmdArray


class AutomaticPipelineSchedule(StrEnum):
    STANDARD_1F1B = "standard_1f1b"
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
    # Hero parameter shardings name context even when context parallelism is disabled.
    shape = (config.mpmd_stages, replica_axis_size, data_axis_size, expert_axis_size, 1, 1)
    axis_names = (_PIPELINE_AXIS, *BATCH_AXES, "context", "model")
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
    start_layer: int = eqx.field(static=True)
    end_layer: int = eqx.field(static=True)

    @eqx.filter_checkpoint
    @named_call
    def embed(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        if self.token_embed is None or self.embed_norm is None or self.embed_gated_norm is None:
            raise ValueError("only stage 0 owns the token embedding")
        # Recompute lookup/norm intermediates from IDs: retaining them costs several
        # full-width tensors per microbatch even when block carries are offloaded.
        hidden = _embedding_gather(self.token_embed, token_ids)
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
        if segment_ids is not None:
            ids = _batch_reshard(segment_ids[0])
            segment_ids = (ids, ids)
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        elif cfg.remat_mode == OFFLOAD_CARRY_REMAT_MODE:
            # Block names its input inside remat, so the backward residual includes a
            # host copy of the stage input instead of retaining it in device memory.
            remat_policy = jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=[],
                names_which_can_be_offloaded=[LAYER_CARRY_REMAT_NAME],
                offload_src="device",
                offload_dst=_HOST_MEMORY_KIND,
            )
        else:
            remat_policy = None

        batch_size, seq_len = hidden.shape[:2]
        long_bounds, valid = fa4_cute_segment_bounds(
            long_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=None
        )
        short_bounds, _ = fa4_cute_segment_bounds(
            short_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=cfg.sliding_window
        )
        long_bounds, short_bounds, valid = map(_batch_reshard, (long_bounds, short_bounds, valid))
        block_metrics = []
        for local_index, block in enumerate(self.blocks):
            layer_index = self.start_layer + local_index
            is_long = (layer_index + 1) % cfg.global_every == 0 or layer_index == cfg.num_layers - 1
            layer_mask = long_mask.with_fa4_bounds(long_bounds if is_long else short_bounds, valid)
            hidden, metrics = eqx.filter_checkpoint(block, policy=remat_policy)(
                hidden,
                layer_mask,
                disable_rope=is_long,
                is_global=is_long,
            )
            block_metrics.append(metrics)

        stacked = jax.tree.map(lambda *values: jnp.stack(values), *block_metrics)
        reduced = reduce_moe_routing_stats(
            stacked,
            num_experts=cfg.num_experts,
            num_experts_per_token=cfg.num_experts_per_token,
        )
        return hidden, {f"{key}_per_layer": value for key, value in reduced.items()}

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
            implementation="xla_fast_bwd",
            block_sizes=_CE_BLOCK_SIZES,
        )


@register_dataclass
@dataclass(frozen=True)
class GrugMoeAutomaticPipelineState:
    """Array state for JaxPP's automatic pipeline transform."""

    trainable_params: tuple[GrugMoePipelineStage, ...]
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
    blocks = _unstacked_blocks(model)
    if layer_counts is None:
        ranges = evenly_partition_layers(len(blocks), num_stages)
    else:
        if len(layer_counts) != num_stages:
            raise ValueError(f"expected {num_stages} layer counts, got {len(layer_counts)}")
        if any(count <= 0 for count in layer_counts):
            raise ValueError(f"layer counts must be positive, got {layer_counts}")
        if sum(layer_counts) != len(blocks):
            raise ValueError(f"layer counts sum to {sum(layer_counts)}, but model has {len(blocks)} layers")
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
                blocks=blocks[start_layer:end_layer],
                final_norm=model.final_norm if is_last else None,
                final_gated_norm=model.final_gated_norm if is_last else None,
                config=model.config,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
    return tuple(stages)


def initialize_stage_local_pipeline_state(
    model_config: GrugModelConfig,
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    mpmd_mesh,
    *,
    num_stages: int,
    seed: int = 0,
    stage_to_mpmd_index: tuple[int, ...] | None = None,
    offload_opt_state: bool = False,
) -> tuple[GrugMoeAutomaticPipelineState, tuple[GrugMoePipelineStage, ...]]:
    """Initialize only the layers owned by this process, with canonical hero keys."""
    pp, _ = _jaxpp_modules()
    if stage_to_mpmd_index is None:
        stage_to_mpmd_index = tuple(range(num_stages))
    if len(stage_to_mpmd_index) != num_stages:
        raise ValueError("each logical stage needs one physical placement")
    ranges = evenly_partition_layers(model_config.num_layers, num_stages)
    params, states, betas, static_stages = [], [], [], []
    for stage_index, (start, end) in enumerate(ranges):
        physical_index = stage_to_mpmd_index[stage_index]
        stage_mesh = mpmd_mesh.unstack[physical_index]

        def initialize(start=start, end=end):
            cfg = model_config
            embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = jax.random.split(
                jax.random.PRNGKey(seed), cfg.num_layers + 4
            )
            first, last = start == 0, end == cfg.num_layers
            stage = GrugMoePipelineStage(
                token_embed=(
                    jax.sharding.reshard(
                        _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std),
                        _EMBED_PARTITION_SPEC,
                    )
                    if first
                    else None
                ),
                embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps) if first else None,
                embed_gated_norm=(
                    GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key) if first else None
                ),
                output_proj=(
                    jax.sharding.reshard(
                        _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std),
                        _LM_HEAD_PARTITION_SPEC,
                    )
                    if last
                    else None
                ),
                blocks=tuple(Block.init(cfg, key=block_keys[i]) for i in range(start, end)),
                final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps) if last else None,
                final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key) if last else None,
                config=cfg,
                start_layer=start,
                end_layer=end,
            )
            stage = mp_policy.cast_to_param(stage)
            trainable, static = eqx.partition(stage, eqx.is_array)
            for i in range(len(stage.blocks)):
                trainable = eqx.tree_at(lambda current, index=i: current.blocks[index].mlp.router_bias, trainable, None)
            opt_state = optimizer.init(trainable)
            if offload_opt_state:
                opt_state = _tree_to_memory_kind(opt_state, _HOST_MEMORY_KIND)
            return (
                trainable,
                opt_state,
                jnp.zeros((end - start, cfg.num_experts), dtype=jnp.float32),
            ), static

        with jax.set_mesh(stage_mesh):
            shapes, static = eqx.filter_eval_shape(initialize)
            owns_stage = _process_has_sharding(NamedSharding(stage_mesh, P()))
            values = eqx.filter_jit(initialize)()[0] if owns_stage else shapes

        def to_mpmd(value, shape, memory_kind="device", physical_index=physical_index, owns_stage=owns_stage):
            spec = shape.sharding.spec if isinstance(shape.sharding, NamedSharding) else P()
            target = pp.MpmdSharding(mpmd_mesh, mesh_ids={physical_index}, spec=spec, memory_kind=memory_kind)
            return pp.MpmdArray([value] if owns_stage else [], target, shape=shape.shape, dtype=shape.dtype)

        stage_params = jax.tree.map(to_mpmd, values[0], shapes[0])
        # Abstract shape tracing does not retain memory kind; specify it for every process.
        opt_memory_kind = _HOST_MEMORY_KIND if offload_opt_state else "device"
        opt_state = jax.tree.map(partial(to_mpmd, memory_kind=opt_memory_kind), values[1], shapes[1])
        qb = jax.tree.map(to_mpmd, values[2], shapes[2])
        params.append(stage_params)
        states.append(opt_state)
        betas.append(qb)
        static_stages.append(static)
    return GrugMoeAutomaticPipelineState(tuple(params), tuple(states), tuple(betas)), tuple(static_stages)


def _process_has_sharding(sharding: NamedSharding) -> bool:
    process_index = jax.process_index()
    return any(device.process_index == process_index for device in sharding.mesh.devices.flat)


def _apply_qb_betas(stage: GrugMoePipelineStage, qb_betas: jax.Array) -> GrugMoePipelineStage:
    blocks = list(stage.blocks)
    for index, block in enumerate(blocks):
        new_bias = -qb_betas[index]
        new_bias = (new_bias - jnp.mean(new_bias)).astype(block.mlp.router.dtype)
        new_mlp = eqx.tree_at(
            lambda mlp: mlp.router_bias,
            block.mlp,
            new_bias,
            is_leaf=lambda value: value is None,
        )
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
    if schedule_name == AutomaticPipelineSchedule.STANDARD_1F1B:
        return pp.Std1F1B(num_stages=config.stages)
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


def make_automatic_pipeline_step(
    optimizer: optax.GradientTransformation,
    mp_policy: jmp.Policy,
    static_stages: tuple[GrugMoePipelineStage, ...],
    sample_state: GrugMoeAutomaticPipelineState,
    sample_batches: GrugLmExample,
    *,
    config: GrugMoePipelineConfig,
    mpmd_mesh,
    schedule_name: AutomaticPipelineSchedule = AutomaticPipelineSchedule.STANDARD_1F1B,
    logsumexp_weight: float | None = None,
    offload_opt_state: bool = False,
):
    """Build a JAXPP optimizer step using the selected pipeline schedule.

    ``static_stages`` and ``sample_state`` must come from the same automatic
    state initializer. ``sample_batches`` must have the leading microbatch axis
    produced by :func:`levanter.pipeline.reshape_batch_into_microbatches`.

    """
    pp, _ = _jaxpp_modules()
    schedule = _automatic_schedule(config, schedule_name)

    def pipeline_step(
        state: GrugMoeAutomaticPipelineState,
        batches: GrugLmExample,
        loss_denominator: jax.Array,
    ):
        compute_params = mp_policy.cast_to_compute(state.trainable_params)

        def loss_fn(trainable_stages: tuple[GrugMoePipelineStage, ...], batch: GrugLmExample):
            hidden = None
            next_qb_betas = []
            last_stage = None
            for stage_index, (trainable_stage, static_stage) in enumerate(
                zip(trainable_stages, static_stages, strict=True)
            ):
                stage = eqx.combine(trainable_stage, static_stage)
                stage = _apply_qb_betas(stage, state.pending_qb_betas[stage_index])
                if stage_index == 0:
                    hidden = stage.embed(batch.tokens)
                assert hidden is not None
                hidden, router_metrics = stage.run_blocks(hidden, batch.attn_mask)
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
            loss = cross_entropy_sum / loss_denominator
            loss = pp.mark_stage_end(loss)
            return loss, tuple(next_qb_betas)

        (loss, next_qb_betas), grads = pp.treduce(
            lambda batch: jax.value_and_grad(loss_fn, has_aux=True)(compute_params, batch),
            batches,
            schedule=schedule,
            operation=((pp.Add, tuple(pp.Add for _ in range(config.stages))), pp.Add),
        )
        grads = mp_policy.cast_to_param(grads)
        next_params = []
        next_opt_state = []
        for params, opt_state, stage_grads in zip(
            state.trainable_params,
            state.opt_state,
            grads,
            strict=True,
        ):
            if offload_opt_state:
                opt_state = _tree_to_memory_kind(opt_state, "device")
            updates, stage_opt_state = optimizer.update(stage_grads, opt_state, params)
            if offload_opt_state:
                stage_opt_state = _tree_to_memory_kind(stage_opt_state, _HOST_MEMORY_KIND)
            next_params.append(mp_policy.cast_to_param(eqx.apply_updates(params, updates)))
            next_opt_state.append(stage_opt_state)
        next_state = dataclasses.replace(
            state,
            trainable_params=tuple(next_params),
            opt_state=tuple(next_opt_state),
            pending_qb_betas=tuple(beta / config.microbatches for beta in next_qb_betas),
        )
        return next_state, {TRAIN_LOSS_KEY: loss}

    state_shardings = _partition_spec_tree(sample_state)
    if offload_opt_state:

        def host_sharding(value):
            if not _is_array(value):
                return None
            return NamedSharding(mpmd_mesh.lowering_mesh(), value.sharding.spec, memory_kind=_HOST_MEMORY_KIND)

        state_shardings = dataclasses.replace(
            state_shardings, opt_state=jax.tree.map(host_sharding, sample_state.opt_state)
        )
    return pp.mpmd_jit_with_loop(
        pipeline_step,
        mpmd_mesh=mpmd_mesh,
        in_specs=(state_shardings, _partition_spec_tree(sample_batches), P()),
        out_specs=(state_shardings, {TRAIN_LOSS_KEY: P()}),
    )


@dataclass(frozen=True)
class ParkedPipelineState:
    state: GrugMoeAutomaticPipelineState
    original_shardings: tuple[jaxpp.MpmdSharding, ...]
    local_device_bytes: int


def _copy_array_to_host(array: jax.Array) -> jax.Array:
    return jax.device_put(array, array.sharding.with_memory_kind(_HOST_MEMORY_KIND), may_alias=False)


def park_pipeline_state(state: GrugMoeAutomaticPipelineState) -> ParkedPipelineState:
    """Copy device state to host, then invalidate all original local device buffers.

    Callers must discard aliases of the original state and restore the returned
    state before training. Already-host-resident optimizer arrays remain intact.
    """
    pp, _ = _jaxpp_modules()
    originals, tree = jax.tree.flatten(state)
    parked = []
    original_shardings = []
    device_arrays = {}
    local_device_bytes = 0
    for value in originals:
        assert isinstance(value, pp.MpmdArray)
        original = value._mpmd_sharding
        original_shardings.append(original)
        if original.memory_kind == _HOST_MEMORY_KIND:
            parked.append(value)
            continue
        local = value.to_mpmd_local_array
        assert local is None or isinstance(local, jax.Array), "Each process must own one pipeline stage"
        host_arrays = []
        if local is not None:
            host = _copy_array_to_host(local)
            host.block_until_ready()
            host_arrays.append(host)
            if id(local) not in device_arrays:
                local_device_bytes += sum(shard.data.nbytes for shard in local.addressable_shards)
                device_arrays[id(local)] = local
        parked.append(
            pp.MpmdArray(
                host_arrays,
                dataclasses.replace(original, memory_kind=_HOST_MEMORY_KIND),
                shape=value.shape,
                dtype=value.dtype,
            )
        )
    # Finish every copy before deleting anything: aliases can occur across state leaves.
    for array in device_arrays.values():
        array.delete()
    return ParkedPipelineState(jax.tree.unflatten(tree, parked), tuple(original_shardings), local_device_bytes)


def restore_pipeline_state(parked: ParkedPipelineState) -> GrugMoeAutomaticPipelineState:
    """Restore parked local buffers and their original MPMD placement metadata."""
    pp, _ = _jaxpp_modules()
    values, tree = jax.tree.flatten(parked.state)
    restored = []
    for value, original in zip(values, parked.original_shardings, strict=True):
        if original.memory_kind == _HOST_MEMORY_KIND:
            restored.append(value)
            continue
        local = value.to_mpmd_local_array
        arrays = []
        if local is not None:
            target = local.sharding.with_memory_kind(original.memory_kind)
            array = jax.device_put(local, target, may_alias=False)
            array.block_until_ready()
            arrays.append(array)
        restored.append(pp.MpmdArray(arrays, original, shape=value.shape, dtype=value.dtype))
    return jax.tree.unflatten(tree, restored)


def precompile_automatic_mpmd_step(step) -> int:
    _jaxpp_modules()
    return mpmd_primitives.precompile_pipeline_tasks(step.local_jaxpr, step.mpmd_mesh)


def prepare_automatic_mpmd_step(
    step,
    state: GrugMoeAutomaticPipelineState,
    batches: GrugLmExample,
    loss_denominator: jax.Array,
    mpmd_mesh,
    *,
    memory_threshold: int | None = None,
) -> PreparedAutomaticMpmdStep:
    """Compile with stage-local state and place the remaining SPMD inputs."""
    pp, _ = _jaxpp_modules()
    compiled = step.compile(state, batches, loss_denominator)
    args_shardings, kwargs_shardings = compiled.in_shardings
    if kwargs_shardings:
        raise ValueError("automatic pipeline step does not accept keyword arguments")

    # Compilation may share counters across stages or prune unused hyperparameters.
    # Gather only scalar metadata so placement preserves nonzero optimizer values.
    scalars = [value for value in jax.tree.leaves(state) if _is_array(value) and value.shape == ()]
    scalar_report = np.zeros((len(scalars), 2), dtype=np.float64)
    for index, value in enumerate(scalars):
        local = value.to_mpmd_local_array if isinstance(value, pp.MpmdArray) else value
        if local is not None:
            scalar_report[index] = (1, float(np.asarray(local)))
    reports = np.asarray(multihost_utils.process_allgather(scalar_report, tiled=False)).reshape(-1, len(scalars), 2)
    owners = reports[:, :, 0].sum(axis=0)
    if np.any(owners == 0):
        raise ValueError("pipeline optimizer scalar has no owning process")
    scalar_values = iter(reports[:, :, 1].sum(axis=0) / owners)

    def place_initial_scalar(value, target):
        if not _is_array(value):
            return value
        if value.shape != ():
            return value
        scalar = np.asarray(next(scalar_values), dtype=value.dtype)
        mesh_ids = target.mesh_ids
        if not mesh_ids:
            return value
        local_arrays = []
        for stage_index in sorted(mesh_ids):
            sharding = NamedSharding(mpmd_mesh.unstack[stage_index], target.spec, memory_kind=target.memory_kind)
            if _process_has_sharding(sharding):
                local_arrays.append(jax.device_put(scalar, sharding))
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
