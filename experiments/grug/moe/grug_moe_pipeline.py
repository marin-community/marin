# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical explicit pipeline-parallel implementation for Grug MoE.

This file is intentionally model-specific. A new Grug model should copy it and
adapt the stage boundaries and stage-local forward functions. The orchestration
uses JaxPP only for explicit MPMD tasks and transfers; it does not ask JaxPP to
discover stages from a traced model.

The implementation is standard 1F1B. It supports any positive microbatch count,
including counts smaller than the number of stages. Low counts leave bubbles,
but do not change the numerical result.

The intended call sequence is visible in the public function names: construct
the mesh, initialize and place the state, split and place each batch, build the
step once, pass it through ``prepare_explicit_step``, then call it for each batch.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

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


JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
_BATCH_AXES = ("replica_dcn", "data", "expert")


@dataclass(frozen=True)
class GrugMoePipelineConfig:
    stages: int
    microbatches: int

    def __post_init__(self) -> None:
        if self.stages < 2:
            raise ValueError(f"pipeline parallelism requires at least 2 stages, got {self.stages}")
        if self.microbatches <= 0:
            raise ValueError(f"microbatches must be positive, got {self.microbatches}")


def make_pipeline_mesh(
    config: GrugMoePipelineConfig,
    *,
    expert_axis_size: int,
    replica_axis_size: int | None = None,
):
    """Build the concrete Grug mesh and wrap it as a JaxPP MPMD mesh."""
    pp, _ = _require_jaxpp()
    if replica_axis_size is None:
        replica_axis_size = max(1, jax.process_count() // config.stages)
    fixed_axes = config.stages * replica_axis_size * expert_axis_size
    if jax.device_count() % fixed_axes != 0:
        raise ValueError(
            f"device count {jax.device_count()} must be divisible by stages ({config.stages}) * "
            f"replicas ({replica_axis_size}) * experts ({expert_axis_size})"
        )

    data_axis_size = jax.device_count() // fixed_axes
    shape = (config.stages, replica_axis_size, data_axis_size, expert_axis_size, 1)
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
    num_stages: int = eqx.field(static=True)

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.num_stages - 1

    @named_call
    def embed(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        if self.token_embed is None or self.embed_norm is None or self.embed_gated_norm is None:
            raise ValueError("only stage 0 owns the token embedding")
        hidden = self.token_embed.at[token_ids].get(out_sharding=P(_BATCH_AXES))
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


def split_transformer(model: Transformer, num_stages: int) -> tuple[GrugMoePipelineStage, ...]:
    """Split a Grug MoE transformer into explicit, contiguous stage pytrees."""
    ranges = evenly_partition_layers(len(model.blocks), num_stages)
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
                num_stages=num_stages,
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


def make_pipeline_state(
    model: Transformer,
    optimizer: optax.GradientTransformation,
    *,
    num_stages: int,
) -> GrugMoePipelineState:
    stages = split_transformer(model, num_stages)
    return GrugMoePipelineState(
        step=jnp.array(0, dtype=jnp.int32),
        params=stages,
        opt_state=tuple(optimizer.init(stage) for stage in stages),
        pending_qb_betas=tuple(
            jnp.zeros((len(stage.blocks), model.config.num_experts), dtype=jnp.float32) for stage in stages
        ),
    )


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


def make_mpmd_pipeline_state(
    model: Transformer,
    optimizer: optax.GradientTransformation,
    mpmd_mesh,
    *,
    num_stages: int,
    memory_threshold: int | None = None,
) -> GrugMoePipelineState:
    """Place parameters before initializing optimizer state to avoid peak replication."""
    pp, _ = _require_jaxpp()
    stages = split_transformer(model, num_stages)
    stage_targets = tuple(_mpmd_sharding_tree(mpmd_mesh, stage_index, stage) for stage_index, stage in enumerate(stages))
    stages = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        stages,
        stage_targets,
        threshold=memory_threshold,
    )

    qb_betas = tuple(jnp.zeros((len(stage.blocks), model.config.num_experts), dtype=jnp.float32) for stage in stages)
    qb_targets = tuple(
        pp.MpmdSharding(mpmd_mesh, mesh_ids={stage_index}, spec=P(None, None)) for stage_index in range(num_stages)
    )
    qb_betas = pp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        qb_betas,
        qb_targets,
        threshold=memory_threshold,
    )
    opt_state = tuple(
        _localize_optimizer_scalars(mpmd_mesh, stage_index, optimizer.init(stage))
        for stage_index, stage in enumerate(stages)
    )
    step = _stage_local_scalar(jnp.array(0, dtype=jnp.int32), NamedSharding(mpmd_mesh.unstack[0], P()))
    return GrugMoePipelineState(
        step=step,
        params=stages,
        opt_state=opt_state,
        pending_qb_betas=qb_betas,
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


def _require_jaxpp():
    if jaxpp is None or mpmd is None:
        raise ModuleNotFoundError(
            "The canonical Grug pipeline requires the patched JaxPP runtime. "
            "Install the pinned revision and apply experiments/grug/moe/jaxpp_jax_0_11_inline.patch."
        )
    return jaxpp, mpmd


def jaxpp_setup_scripts(*, revision: str = JAXPP_REVISION) -> tuple[str, ...]:
    """Install the pinned, patched JaxPP runtime in an Iris worker environment."""
    return (
        "\n".join(
            (
                "set -euxo pipefail",
                'cd "$IRIS_WORKDIR"',
                "uv pip install --link-mode symlink cupy-cuda13x",
                "rm -rf /tmp/jaxpp",
                "git clone --quiet --filter=blob:none https://github.com/NVIDIA/jaxpp.git /tmp/jaxpp",
                f"git -C /tmp/jaxpp checkout --quiet {revision}",
                "git -C /tmp/jaxpp apply --unidiff-zero "
                '"$IRIS_WORKDIR/experiments/grug/moe/jaxpp_jax_0_11_inline.patch"',
                "uv pip install --link-mode symlink --no-deps /tmp/jaxpp",
            )
        )
        + "\n",
    )


def _is_array(value: Any) -> bool:
    return hasattr(value, "sharding") and hasattr(value, "ndim")


def _named_sharding_tree(mesh, tree):
    def sharding(value):
        if not _is_array(value):
            return None
        if isinstance(value.sharding, NamedSharding):
            return NamedSharding(mesh, value.sharding.spec)
        return NamedSharding(mesh, P(*([None] * value.ndim)))

    return jax.tree.map(sharding, tree)


def _apply_updates(params, updates):
    def apply_one(param, update):
        if param is None:
            return None
        return (param + update).astype(param.dtype)

    return jax.tree.map(apply_one, params, updates, is_leaf=lambda value: value is None)


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


def place_pipeline_state(
    mpmd_mesh,
    state: GrugMoePipelineState,
    *,
    memory_threshold: int | None = None,
) -> GrugMoePipelineState:
    """Move each stage's state from the ordinary JAX mesh to its MPMD mesh."""
    pp, _ = _require_jaxpp()
    target = dataclasses.replace(
        state,
        step=pp.MpmdSharding(mpmd_mesh, mesh_ids={0}, spec=P()),
        params=tuple(
            _mpmd_sharding_tree(mpmd_mesh, stage_index, stage) for stage_index, stage in enumerate(state.params)
        ),
        opt_state=tuple(
            _mpmd_sharding_tree(mpmd_mesh, stage_index, opt_state)
            for stage_index, opt_state in enumerate(state.opt_state)
        ),
        pending_qb_betas=tuple(
            pp.MpmdSharding(mpmd_mesh, mesh_ids={stage_index}, spec=P(None, None))
            for stage_index in range(len(state.params))
        ),
    )
    return pp.spmd_to_mpmd_reshard(mpmd_mesh, state, target, threshold=memory_threshold)


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
    lowered: Any

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
        fallback = (state, {"train/loss": local_loss})
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

    ``sample_state`` and ``sample_batches`` must already be placed with
    :func:`place_pipeline_state` and :func:`place_pipeline_batches`.
    """
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
        NamedSharding(mpmd_mesh.unstack[stage_index], P(_BATCH_AXES, None, None)) for stage_index in range(num_stages)
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
        return hidden, metrics["qb_beta_per_layer"], params.local_router_loss(metrics) / config.microbatches

    def stage_forward(
        params: GrugMoePipelineStage,
        qb_betas: jax.Array,
        hidden: jax.Array,
        batch: GrugLmExample,
    ):
        params = mp_policy.cast_to_compute(_apply_qb_betas(params, qb_betas))
        hidden, metrics = params.run_blocks(hidden, batch.attn_mask)
        return hidden, metrics["qb_beta_per_layer"], params.local_router_loss(metrics) / config.microbatches

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
            return loss, metrics["qb_beta_per_layer"]

        (loss, next_qb_betas), (grads, hidden_cotangent) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        return loss, next_qb_betas, grads, hidden_cotangent

    def update_stage(params, opt_state, grads):
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        return _apply_updates(params, updates), next_opt_state

    def add_trees(left, right):
        return jax.tree.map(lambda x, y: x + y, left, right)

    def average_tree(tree):
        scale = jnp.asarray(1.0 / config.microbatches, dtype=jnp.float32)
        return jax.tree.map(lambda value: value * scale, tree)

    def loss_weight_sum(batch: GrugLmExample):
        return jnp.sum(batch.loss_weight.astype(jnp.float32))

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
    router_loss_accumulator = explicit_mpmd.task(
        add_trees,
        name="grug_1f1b_accumulate_router_loss",
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
        accumulated_forward_router_loss = None
        stage_inputs = {}
        forward_transfers = {}
        backward_transfers = {}
        completed_forwards = set()
        completed_backwards = set()

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
            nonlocal accumulated_forward_router_loss
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
                router_loss = explicit_mpmd.transfer(router_loss, out_shardings=loss_shardings[1]).done()
                accumulated_forward_router_loss = accumulate(
                    accumulated_forward_router_loss,
                    router_loss,
                    router_loss_accumulator,
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

            hidden, qb_betas, router_loss = stage_forward_tasks[stage_index - 1](
                params[stage_index], state.pending_qb_betas[stage_index], hidden, stage_batches[stage_index]
            )
            next_qb_betas[stage_index] = accumulate(
                next_qb_betas[stage_index],
                qb_betas,
                qb_accumulators[stage_index],
            )
            router_loss = explicit_mpmd.transfer(router_loss, out_shardings=loss_shardings[1]).done()
            accumulated_forward_router_loss = accumulate(
                accumulated_forward_router_loss,
                router_loss,
                router_loss_accumulator,
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

        if accumulated_loss is None:
            raise ValueError("1F1B did not accumulate a loss")
        if accumulated_forward_router_loss is None:
            raise ValueError("1F1B did not accumulate forward-stage router loss")
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
        return next_state, {"train/loss": stage0_loss}

    return pipeline_step
