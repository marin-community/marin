# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.expert_selection import ExpertSelectionMethod, select_experts
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)


class CheckpointLoader(Protocol):
    def __call__(
        self,
        tree: dict[str, object],
        checkpoint_path: str,
        *,
        mesh: Mesh | None,
        allow_partial: bool,
    ) -> dict[str, object]: ...


class InitializationMode(StrEnum):
    FULL_STATE = "full_state"
    WEIGHTS_ONLY = "weights_only"


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    sharding_dump_path: str | None = None
    initialization_mode: InitializationMode = InitializationMode.FULL_STATE
    initialization_source_model: GrugModelConfig | None = None
    initialization_expert_offset: int | None = None
    initialization_expert_selection_method: ExpertSelectionMethod | None = None


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True
    nested_expert_counts: tuple[int, ...] = ()
    """Fixed prefix expert banks evaluated in addition to the full model."""
    nested_expert_ranges: tuple[tuple[int, int], ...] = ()
    """Named by range in metrics; each pair is a half-open ``(start, end)`` bank."""


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # GPU processes per task: > 1 runs one JAX process per GPU (multi-controller)
    # via the iris.runtime.multigpu supervisor instead of one process per node.
    processes_per_task: int = 1


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
    mp: jmp.Policy,
    expert_selection: tuple[tuple[int, ...], ...] | None = None,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Eval receives the fp32 master params; cast to compute dtype so the forward matches
        # training (and some GPU attention kernels accept only bf16/fp16, not fp32).
        model = mp.cast_to_compute(model)
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        expert_eligibility = None
        if expert_selection is not None:
            if len(expert_selection) not in (1, model.config.num_layers):
                raise ValueError(f"expert_selection must have one shared set or {model.config.num_layers} layer sets")
            layer_eligibility = jnp.zeros(
                (len(expert_selection), model.config.num_experts),
                dtype=jnp.bool_,
            )
            for layer, selected_experts in enumerate(expert_selection):
                if len(set(selected_experts)) != len(selected_experts):
                    raise ValueError("expert_selection contains duplicate experts")
                if not all(0 <= expert < model.config.num_experts for expert in selected_experts):
                    raise ValueError("expert_selection contains an out-of-range expert")
                layer_eligibility = layer_eligibility.at[layer, jnp.asarray(selected_experts)].set(True)
            if len(expert_selection) == 1:
                expert_eligibility = jnp.broadcast_to(
                    layer_eligibility[0, None, :],
                    (batch.tokens.shape[0], model.config.num_experts),
                )
            else:
                expert_eligibility = jnp.broadcast_to(
                    layer_eligibility[:, None, :],
                    (model.config.num_layers, batch.tokens.shape[0], model.config.num_experts),
                )
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
            include_mtp=False,  # eval reports pure next-token loss, not the main+MTP training objective
            expert_eligibility=expert_eligibility,
        )[0]
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        head_dim=model_config.inferred_head_dim,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    # QB router biases to apply on the NEXT step: [num_layers, num_experts]. All-zero (a no-op) unless
    # cfg.qb_routing is on, so the model's forward is byte-for-byte unchanged when QB is off.
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from the QB betas computed on the previous step.

    Every grug layer is MoE and lives in the single stacked scan body, so ``router_bias`` is the
    [num_layers, num_experts] leaf ``stacked_blocks.stacked.mlp.router_bias`` and this is one
    vectorized assignment. Per-layer mean-centering keeps each layer's bias zero-sum. When QB is off
    the betas are all-zero, so the bias stays zero and never enters the (plain top-k) forward.
    """
    new_bias = -qb_betas
    new_bias = new_bias - jnp.mean(new_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_bias)


# SCALE_OFFLOAD_OPT_STATE=1 parks the optimizer state on pinned host memory between steps so it
# is off-HBM during forward/backward (freeing HBM for the memory-aware scheduler to async the
# backward re-gathers), streaming it back to device only for the optimizer update. Viable on GB200
# via the ~900 GB/s NVLink-C2C Grace<->Blackwell link; ~50ms/direction hides under the ~15s step.
_OFFLOAD_OPT_STATE = os.environ.get("SCALE_OFFLOAD_OPT_STATE") == "1"
# SCALE_OFFLOAD_MASTER_PARAMS=1 additionally parks the fp32 master weights on pinned host between
# steps. To keep them off-HBM during forward/backward the cast to bf16 is hoisted OUT of the
# differentiated loss (so value_and_grad tracks the bf16 params, not the fp32 master); the fp32
# master is streamed to device only for that cast and again for the optimizer update. Numerically
# identical to the on-device path -- the cast's gradient is identity, so dL/dbf16 == dL/dfp32.
_OFFLOAD_MASTER_PARAMS = os.environ.get("SCALE_OFFLOAD_MASTER_PARAMS") == "1"


def _opt_state_to_memory_kind(tree, kind: str):
    """Move optimizer-state array leaves to ``kind`` (``pinned_host`` to offload, ``device`` to
    bring back), preserving each leaf's PartitionSpec. Works both eagerly and inside jit
    (explicit-sharding mode exposes the traced leaf's sharding via ``jax.typeof``)."""

    def _move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        sharding = jax.typeof(leaf).sharding
        mesh = getattr(sharding, "mesh", None)
        if mesh is None or len(getattr(mesh, "axis_names", ())) == 0:
            # Scalars / replicated hyperparams (inject_hyperparams count, learning_rate) carry an
            # empty abstract mesh; leave them on device (negligible HBM) to avoid a mesh mismatch.
            return leaf
        return jax.device_put(leaf, sharding.with_memory_kind(kind))

    return jax.tree.map(_move, tree)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    opt_state = optimizer.init(params)
    if _OFFLOAD_OPT_STATE:
        opt_state = _opt_state_to_memory_kind(opt_state, "pinned_host")
    host_params = _opt_state_to_memory_kind(params, "pinned_host") if _OFFLOAD_MASTER_PARAMS else params
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=host_params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros_like(params.stacked_blocks.stacked.mlp.router_bias),
    )


def init_weights_only_from_checkpoint(
    state: GrugTrainState,
    checkpoint_path: str,
    *,
    mesh: Mesh | None,
    load_ema: bool,
    _load_fn: CheckpointLoader = load_checkpoint,
) -> GrugTrainState:
    """Load model and router-bias state while retaining a fresh optimizer and step."""
    exemplar: dict[str, object] = {
        "params": state.params,
        "pending_qb_betas": state.pending_qb_betas,
    }
    loaded = _load_fn(exemplar, checkpoint_path, mesh=mesh, allow_partial=True)
    updates: dict[str, object] = {
        "params": loaded["params"],
        "pending_qb_betas": loaded["pending_qb_betas"],
    }
    if load_ema and state.ema_params is not None:
        updates["ema_params"] = loaded["params"]
    return dataclasses.replace(state, **updates)


def extract_expert_range(
    model: Transformer,
    pending_qb_betas: jax.Array,
    *,
    target_model: Transformer,
    target_pending_qb_betas: jax.Array,
    expert_offset: int,
) -> tuple[Transformer, jax.Array]:
    """Extract one contiguous routed-expert bank into a smaller physical model."""
    source_config = model.config
    target_config = target_model.config
    expert_count = target_config.num_experts
    expert_end = expert_offset + expert_count
    if target_config.mtp_depth or source_config.mtp_depth:
        raise ValueError("expert-range extraction does not support MTP models")
    if expert_offset < 0 or expert_end > source_config.num_experts:
        raise ValueError("requested expert range falls outside the source model")
    if target_config.nested_expert_counts:
        raise ValueError("the extracted target must use an ordinary single-bank router")
    source_ranges = source_config.router_balance_group_ranges
    try:
        balance_group = source_ranges.index((expert_offset, expert_end))
    except ValueError as error:
        raise ValueError("source model has no independent QB state for the requested expert range") from error

    source_blocks = model.stacked_blocks.stacked
    source_mlp = source_blocks.mlp
    source_experts = source_mlp.expert_mlp
    target_mlp = target_model.stacked_blocks.stacked.mlp
    target_experts = target_mlp.expert_mlp

    def output_sharding(target: jax.Array) -> NamedSharding | None:
        return target.sharding if isinstance(target.sharding, NamedSharding) else None

    def expert_slice(weights: jax.Array | None, target_weights: jax.Array | None) -> jax.Array | None:
        if weights is None:
            if target_weights is not None:
                raise ValueError("source and target expert parameter layouts differ")
            return None
        if target_weights is None:
            raise ValueError("source and target expert parameter layouts differ")
        return weights.at[:, expert_offset:expert_end].get(out_sharding=output_sharding(target_weights))

    extracted_experts = dataclasses.replace(
        source_experts,
        w_gate=expert_slice(source_experts.w_gate, target_experts.w_gate),
        w_up=expert_slice(source_experts.w_up, target_experts.w_up),
        w_down=expert_slice(source_experts.w_down, target_experts.w_down),
        w_gate_up=expert_slice(source_experts.w_gate_up, target_experts.w_gate_up),
    )
    extracted_mlp = dataclasses.replace(
        source_mlp,
        router=source_mlp.router.at[..., expert_offset:expert_end].get(out_sharding=output_sharding(target_mlp.router)),
        router_bias=source_mlp.router_bias.at[:, balance_group, expert_offset:expert_end].get(
            out_sharding=output_sharding(target_mlp.router_bias)
        ),
        expert_mlp=extracted_experts,
        cfg=target_config,
    )
    target_blocks = dataclasses.replace(
        target_model.stacked_blocks.stacked,
        rms_attn=source_blocks.rms_attn,
        attn_gated_norm=source_blocks.attn_gated_norm,
        attn=dataclasses.replace(source_blocks.attn, cfg=target_config),
        rms_mlp=source_blocks.rms_mlp,
        mlp_gated_norm=source_blocks.mlp_gated_norm,
        mlp=extracted_mlp,
        shared=source_blocks.shared,
        sconv_attn=source_blocks.sconv_attn,
        sconv_mlp=source_blocks.sconv_mlp,
    )
    extracted_model = dataclasses.replace(
        target_model,
        token_embed=model.token_embed,
        over_encoding=model.over_encoding,
        embed_norm=model.embed_norm,
        embed_gated_norm=model.embed_gated_norm,
        output_proj=model.output_proj,
        stacked_blocks=dataclasses.replace(target_model.stacked_blocks, stacked=target_blocks),
        final_norm=model.final_norm,
    )
    extracted_pending_qb_betas = pending_qb_betas.at[:, balance_group, expert_offset:expert_end].get(
        out_sharding=output_sharding(target_pending_qb_betas)
    )
    return extracted_model, extracted_pending_qb_betas


def init_expert_range_from_checkpoint(
    state: GrugTrainState,
    checkpoint_path: str,
    *,
    source_model: Transformer,
    mesh: Mesh | None,
    expert_offset: int,
    load_ema: bool,
    _load_fn: CheckpointLoader = load_checkpoint,
) -> GrugTrainState:
    """Load a larger nested checkpoint and initialize a fresh smaller trainer."""
    exemplar = {
        "params": source_model,
        "pending_qb_betas": jnp.zeros_like(source_model.stacked_blocks.stacked.mlp.router_bias),
    }
    loaded = _load_fn(exemplar, checkpoint_path, mesh=mesh, allow_partial=True)
    params, pending_qb_betas = extract_expert_range(
        loaded["params"],
        loaded["pending_qb_betas"],
        target_model=state.params,
        target_pending_qb_betas=state.pending_qb_betas,
        expert_offset=expert_offset,
    )
    return dataclasses.replace(
        state,
        params=params,
        pending_qb_betas=pending_qb_betas,
        ema_params=params if load_ema and state.ema_params is not None else state.ema_params,
    )


def extract_expert_selection(
    model: Transformer,
    pending_qb_betas: jax.Array,
    *,
    target_model: Transformer,
    target_pending_qb_betas: jax.Array,
    expert_selection: tuple[tuple[int, ...], ...],
) -> tuple[Transformer, jax.Array]:
    """Gather one possibly layer-specific expert bank into a physical model."""
    source_config = model.config
    target_config = target_model.config
    expert_count = target_config.num_experts
    if target_config.mtp_depth or source_config.mtp_depth:
        raise ValueError("expert-selection extraction does not support MTP models")
    if target_config.nested_expert_counts:
        raise ValueError("the extracted target must use an ordinary single-bank router")
    if len(expert_selection) == 1:
        expert_selection = expert_selection * source_config.num_layers
    if len(expert_selection) != source_config.num_layers:
        raise ValueError("expert selection must contain one shared bank or one bank per layer")
    if any(len(layer_selection) != expert_count for layer_selection in expert_selection):
        raise ValueError("every expert selection must match the target expert count")
    if any(len(set(layer_selection)) != expert_count for layer_selection in expert_selection):
        raise ValueError("expert selection contains duplicate experts")
    if any(
        expert < 0 or expert >= source_config.num_experts
        for layer_selection in expert_selection
        for expert in layer_selection
    ):
        raise ValueError("expert selection contains an out-of-range expert")

    source_blocks = model.stacked_blocks.stacked
    source_mlp = source_blocks.mlp
    source_experts = source_mlp.expert_mlp
    target_mlp = target_model.stacked_blocks.stacked.mlp
    target_experts = target_mlp.expert_mlp
    selection = jnp.asarray(expert_selection, dtype=jnp.int32)
    layer_indices = jnp.arange(source_config.num_layers, dtype=jnp.int32)[:, None]

    def output_sharding(target: jax.Array) -> NamedSharding | None:
        return target.sharding if isinstance(target.sharding, NamedSharding) else None

    def gather_experts(weights: jax.Array | None, target_weights: jax.Array | None) -> jax.Array | None:
        if weights is None:
            if target_weights is not None:
                raise ValueError("source and target expert parameter layouts differ")
            return None
        if target_weights is None:
            raise ValueError("source and target expert parameter layouts differ")
        return weights.at[layer_indices, selection].get(out_sharding=output_sharding(target_weights))

    hidden_indices = jnp.arange(source_config.hidden_dim, dtype=jnp.int32)[None, :, None]
    router = source_mlp.router.at[
        layer_indices[:, :, None],
        hidden_indices,
        selection[:, None, :],
    ].get(out_sharding=output_sharding(target_mlp.router))
    source_router_bias = source_mlp.router_bias[:, 0, :] if source_mlp.router_bias.ndim == 3 else source_mlp.router_bias
    source_pending_qb_betas = pending_qb_betas[:, 0, :] if pending_qb_betas.ndim == 3 else pending_qb_betas

    extracted_experts = dataclasses.replace(
        source_experts,
        w_gate=gather_experts(source_experts.w_gate, target_experts.w_gate),
        w_up=gather_experts(source_experts.w_up, target_experts.w_up),
        w_down=gather_experts(source_experts.w_down, target_experts.w_down),
        w_gate_up=gather_experts(source_experts.w_gate_up, target_experts.w_gate_up),
    )
    extracted_mlp = dataclasses.replace(
        source_mlp,
        router=router,
        router_bias=source_router_bias.at[layer_indices, selection].get(
            out_sharding=output_sharding(target_mlp.router_bias)
        ),
        expert_mlp=extracted_experts,
        cfg=target_config,
    )
    target_blocks = dataclasses.replace(
        target_model.stacked_blocks.stacked,
        rms_attn=source_blocks.rms_attn,
        attn_gated_norm=source_blocks.attn_gated_norm,
        attn=dataclasses.replace(source_blocks.attn, cfg=target_config),
        rms_mlp=source_blocks.rms_mlp,
        mlp_gated_norm=source_blocks.mlp_gated_norm,
        mlp=extracted_mlp,
        shared=source_blocks.shared,
        sconv_attn=source_blocks.sconv_attn,
        sconv_mlp=source_blocks.sconv_mlp,
    )
    extracted_model = dataclasses.replace(
        target_model,
        token_embed=model.token_embed,
        over_encoding=model.over_encoding,
        embed_norm=model.embed_norm,
        embed_gated_norm=model.embed_gated_norm,
        output_proj=model.output_proj,
        stacked_blocks=dataclasses.replace(target_model.stacked_blocks, stacked=target_blocks),
        final_norm=model.final_norm,
    )
    extracted_pending_qb_betas = source_pending_qb_betas.at[layer_indices, selection].get(
        out_sharding=output_sharding(target_pending_qb_betas)
    )
    return extracted_model, extracted_pending_qb_betas


def init_expert_selection_from_checkpoint(
    state: GrugTrainState,
    checkpoint_path: str,
    *,
    source_model: Transformer,
    mesh: Mesh | None,
    method: ExpertSelectionMethod,
    load_ema: bool,
    _load_fn: CheckpointLoader = load_checkpoint,
) -> GrugTrainState:
    """Load a larger checkpoint and initialize a selected physical expert bank."""
    exemplar = {
        "params": source_model,
        "pending_qb_betas": jnp.zeros_like(source_model.stacked_blocks.stacked.mlp.router_bias),
    }
    loaded = _load_fn(exemplar, checkpoint_path, mesh=mesh, allow_partial=True)
    expert_selection = select_experts(
        loaded["params"],
        expert_count=state.params.config.num_experts,
        method=method,
        seed=0,
    )
    if expert_selection is None:
        raise ValueError("expert selection requires a smaller target model")
    params, pending_qb_betas = extract_expert_selection(
        loaded["params"],
        loaded["pending_qb_betas"],
        target_model=state.params,
        target_pending_qb_betas=state.pending_qb_betas,
        expert_selection=expert_selection,
    )
    return dataclasses.replace(
        state,
        params=params,
        pending_qb_betas=pending_qb_betas,
        ema_params=params if load_ema and state.ema_params is not None else state.ema_params,
    )


def _scheduled_mtp_weight(config: GrugModelConfig, step: jax.Array, num_train_steps: int) -> jax.Array | None:
    """MTP loss weight for the current step: constant ``mtp_loss_weight`` until ``mtp_decay_start_frac``
    of training, then ``mtp_final_loss_weight`` for the tail. None when no schedule is configured."""
    if config.mtp_final_loss_weight is None:
        return None
    frac = step.astype(jnp.float32) / num_train_steps
    return jnp.where(frac >= config.mtp_decay_start_frac, config.mtp_final_loss_weight, config.mtp_loss_weight)


def training_expert_eligibility(
    model_config: GrugModelConfig,
    *,
    batch_size: int,
    step: jax.Array,
) -> jax.Array | None:
    """Return deterministic sequence- or layer-indexed fixed-prefix eligibility."""
    fraction = model_config.nested_batch_fraction
    if fraction == 0.0:
        return None

    period = 1 if fraction == 1.0 else round(1.0 / fraction)
    row_ids = jnp.arange(batch_size, dtype=jnp.int32)
    schedule_ids = row_ids + step
    restricted_rows = schedule_ids % period == 0
    event_ids = schedule_ids // period
    level_ids = event_ids % len(model_config.nested_expert_counts)
    nested_ranges = model_config.nested_expert_ranges
    nested_counts = jnp.asarray([count for count, _ in nested_ranges], dtype=jnp.int32)
    nested_offsets = jnp.asarray([offset for _, offset in nested_ranges], dtype=jnp.int32)
    eligible_counts = nested_counts[level_ids]
    eligible_offsets = nested_offsets[level_ids]
    expert_ids = jnp.arange(model_config.num_experts, dtype=jnp.int32)
    nested = (expert_ids[None, :] >= eligible_offsets[:, None]) & (
        expert_ids[None, :] < eligible_offsets[:, None] + eligible_counts[:, None]
    )
    if model_config.nested_layer_fraction == 1.0:
        return jnp.where(restricted_rows[:, None], nested, True)

    layer_period = round(1.0 / model_config.nested_layer_fraction)
    layer_ids = jnp.arange(model_config.num_layers, dtype=jnp.int32)
    restricted_layers = (layer_ids[:, None] + event_ids[None, :]) % layer_period == 0
    restricted_layer_rows = restricted_layers & restricted_rows[None, :]
    return jnp.where(restricted_layer_rows[:, :, None], nested[None, :, :], True)


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    num_train_steps: int,
    watch_config: WatchConfig | None = None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
        # Apply the pending QB betas (from the previous step) to the router biases inside JIT, before
        # the forward. All-zero when QB is off, so qb_params is numerically identical to state.params.
        qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas) if ema_beta is not None else None
        mtp_w = _scheduled_mtp_weight(state.params.config, state.step, num_train_steps)
        expert_eligibility = training_expert_eligibility(
            state.params.config,
            batch_size=batch.tokens.shape[0],
            step=state.step,
        )

        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            loss, qb_beta, aux = compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                mtp_loss_weight=mtp_w,
                expert_eligibility=expert_eligibility,
            )
            return loss, (qb_beta, aux)

        if _OFFLOAD_MASTER_PARAMS:
            # Host-resident master: stream to device, cast to bf16, and differentiate the bf16 params
            # (cast hoisted out of the loss) so the fp32 master is dead through fwd/bwd and XLA frees
            # its HBM. Re-stream the fp32 for the update below. dL/dbf16 == dL/dfp32 (cast grad is id).
            qb_params_cast = _apply_qb_betas(_opt_state_to_memory_kind(state.params, "device"), state.pending_qb_betas)
            bf16_params = mp.cast_to_compute(qb_params_cast)

            def offload_loss_fn(bp):
                loss, qb_beta, aux = bp.next_token_loss(
                    batch.tokens,
                    batch.loss_weight,
                    mask=batch.attn_mask,
                    reduction="mean",
                    logsumexp_weight=z_loss,
                    mtp_loss_weight=mtp_w,
                    expert_eligibility=expert_eligibility,
                )
                return loss, (qb_beta, aux)

            (loss, (qb_beta_per_layer, aux)), grads = jax.value_and_grad(offload_loss_fn, has_aux=True)(bf16_params)
            grads = mp.cast_to_param(grads)  # bf16 grads -> fp32 to match the master/optimizer
            qb_params = _apply_qb_betas(_opt_state_to_memory_kind(state.params, "device"), state.pending_qb_betas)
        else:
            qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
            (loss, (qb_beta_per_layer, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        # aux carries the main/MTP loss split (and scheduled MTP weight) when MTP is on; empty otherwise.
        metrics = {"train/loss": loss, **{f"train/{key}": value for key, value in aux.items()}}
        if expert_eligibility is not None:
            expert_ids = jnp.arange(state.params.config.num_experts)
            for nested_count, nested_offset in state.params.config.nested_expert_ranges:
                bank = (expert_ids >= nested_offset) & (expert_ids < nested_offset + nested_count)
                restricted = jnp.all(expert_eligibility == bank, axis=-1)
                if restricted.ndim == 1:
                    sequence_fraction = layer_sequence_fraction = jnp.mean(restricted.astype(jnp.float32))
                else:
                    sequence_fraction = jnp.mean(jnp.any(restricted, axis=0).astype(jnp.float32))
                    layer_sequence_fraction = jnp.mean(restricted.astype(jnp.float32))
                bank_name = f"e{nested_count}" if nested_offset == 0 else f"e{nested_count}_offset{nested_offset}"
                metrics[f"train/nested/{bank_name}_sequence_fraction"] = sequence_fraction
                metrics[f"train/nested/{bank_name}_layer_sequence_fraction"] = layer_sequence_fraction
        # Optimizer state is host-resident between steps when offloading; stream it to device
        # only here (after backward) for the update, then send the new state back to host below.
        opt_state_in = _opt_state_to_memory_kind(state.opt_state, "device") if _OFFLOAD_OPT_STATE else state.opt_state
        updates, opt_state = optimizer.update(grads, opt_state_in, qb_params)
        params = optax.apply_updates(qb_params, updates)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
                params,
            )

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=opt_state_in,
                model_tree_type=type(state.params),
            )

        if _OFFLOAD_OPT_STATE:
            opt_state = _opt_state_to_memory_kind(opt_state, "pinned_host")
        if _OFFLOAD_MASTER_PARAMS:
            params = _opt_state_to_memory_kind(params, "pinned_host")

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=qb_beta_per_layer,
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        num_train_steps=trainer.num_train_steps,
        watch_config=watch_config if watch_config.is_enabled else None,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    with set_mesh(mesh):
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)

        # SCALE_DISABLE_CHECKPOINT=1 skips checkpoint creation entirely (including the forced final
        # save), so short profiling/debug runs reach tracker.finish() — the sharded tensorstore save
        # can crash at large scale and would otherwise block the profile upload.
        if os.environ.get("SCALE_DISABLE_CHECKPOINT") == "1":
            checkpointer = None
        else:
            checkpointer = trainer.checkpointer.create(run_id)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        if config.trainer.initialization_mode is InitializationMode.WEIGHTS_ONLY:
            if int(state.step) == 0 and trainer.initialize_from is not None:
                source_config = config.trainer.initialization_source_model
                expert_offset = config.trainer.initialization_expert_offset
                selection_method = config.trainer.initialization_expert_selection_method
                if source_config is None and expert_offset is None and selection_method is None:
                    state = init_weights_only_from_checkpoint(
                        state,
                        trainer.initialize_from,
                        mesh=mesh,
                        load_ema=config.trainer.ema_beta is not None,
                    )
                elif source_config is not None and expert_offset is not None and selection_method is None:

                    @jax.jit
                    def _init_source_model(source_key):
                        return trainer.mp.cast_to_param(Transformer.init(source_config, key=source_key))

                    state = init_expert_range_from_checkpoint(
                        state,
                        trainer.initialize_from,
                        source_model=_init_source_model(model_key),
                        mesh=mesh,
                        expert_offset=expert_offset,
                        load_ema=config.trainer.ema_beta is not None,
                    )
                elif source_config is not None and expert_offset is None and selection_method is not None:

                    @jax.jit
                    def _init_source_model(source_key):
                        return trainer.mp.cast_to_param(Transformer.init(source_config, key=source_key))

                    state = init_expert_selection_from_checkpoint(
                        state,
                        trainer.initialize_from,
                        source_model=_init_source_model(model_key),
                        mesh=mesh,
                        method=selection_method,
                        load_ema=config.trainer.ema_beta is not None,
                    )
                else:
                    raise ValueError("invalid expert-extraction initialization configuration")
        dump_grug_state_sharding_run_artifact(
            state,
            log_dir=trainer.log_dir,
            run_id=run_id,
            path_override=config.trainer.sharding_dump_path,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        nested_evaluators = []
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
                mp=trainer.mp,
            )
            for nested_count in eval_cfg.nested_expert_counts:
                nested_evaluator = build_tagged_evaluator(
                    data_config=config.data,
                    max_seq_len=config.model.max_seq_len,
                    mesh=mesh,
                    eval_cfg=eval_cfg,
                    mp=trainer.mp,
                    expert_selection=(tuple(range(nested_count)),),
                )
                if nested_evaluator is not None:
                    nested_evaluators.append((nested_count, nested_evaluator))
            for expert_start, expert_end in eval_cfg.nested_expert_ranges:
                nested_evaluator = build_tagged_evaluator(
                    data_config=config.data,
                    max_seq_len=config.model.max_seq_len,
                    mesh=mesh,
                    eval_cfg=eval_cfg,
                    mp=trainer.mp,
                    expert_selection=(tuple(range(expert_start, expert_end)),),
                )
                if nested_evaluator is not None:
                    nested_evaluators.append((f"e{expert_end - expert_start}_offset{expert_start}", nested_evaluator))

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=trainer.num_train_steps)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        # jax.profiler cannot reliably write to object stores, so the trace goes to local disk and
        # is uploaded to SCALE_PROFILER_UPLOAD (an fsspec URL) after training — fsspec handles s3,
        # so the trace survives the ephemeral pod for offline ingestion.
        prof_local_dir = f"/tmp/grug-profiler/{run_id}/profiler"
        if profiler_enabled:
            state_callbacks.add_hook(
                callbacks.profile(
                    prof_local_dir,
                    profiler_cfg.start_step,
                    profiler_num_steps,
                    profiler_cfg.perfetto_link,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )
                for nested_count, nested_evaluator in nested_evaluators:
                    nested_name = f"e{nested_count}" if isinstance(nested_count, int) else nested_count
                    state_callbacks.add_hook(
                        cb_tagged_evaluate(
                            nested_evaluator,
                            prefix=f"{eval_cfg.prefix}/nested_{nested_name}",
                            eval_current=eval_cfg.eval_current,
                            eval_ema=eval_ema,
                        ),
                        every=interval,
                    )

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        # Main optimization loop.
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])

                if jnp.isnan(metrics["train/loss"]):
                    logger.error(f"NaN loss at step {int(state.step)}. Stopping training.")
                    break
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    # train/loss is logged by the step-info callback above; forward the rest of the
                    # metrics (e.g. the main/MTP loss split and scheduled MTP weight) to the tracker.
                    extra_metrics = {key: value for key, value in metrics.items() if key != "train/loss"}
                    if extra_metrics:
                        levanter.tracker.log(extra_metrics, step=step)
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=int(state.step))
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

        prof_upload = os.environ.get("SCALE_PROFILER_UPLOAD")
        if profiler_enabled and prof_upload and jax.process_index() == 0:
            fs = fsspec.core.get_fs_token_paths(prof_upload, mode="wb")[0]
            fs.put(os.path.join(prof_local_dir, "*"), prof_upload.rstrip("/"), recursive=True)
            logger.info(f"Uploaded profiler trace to {prof_upload}")

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
    )


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
]
