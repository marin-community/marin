# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
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
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, _histogram_from_expert_counts
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

HERO_EP_RUNTIME_ENV = {
    "JAX_ENABLE_PGLE": "false",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
_XLA_FLAG_DEFAULTS = (
    "--xla_gpu_experimental_parallel_collective_overlap_limit=4",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
)
# TODO(https://github.com/marin-community/marin/issues/5675): Re-enable XLA GPU
# command buffers after the CUDA graph failure is fixed.
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="


def _apply_hero_ep_runtime_defaults() -> None:
    os.environ.update(HERO_EP_RUNTIME_ENV)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    flag_defaults = (*_XLA_FLAG_DEFAULTS, XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG)
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    xla_flags.extend(flag for flag in flag_defaults if flag.partition("=")[0] not in explicit_names)
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.
    # Keep disabled except on model sizes where Grace-Blackwell host offload has been measured.
    # The d6144 EP64 runs used it; d5120 required a 135 GiB pinned-host arena and regressed.
    offload_opt_state: bool = False
    # A short throughput gate leaves this off. A compute-optimal run needs it: the loop already
    # restores from the latest committed checkpoint, so without a writer an interrupted run
    # restarts at step 0.
    save_checkpoints: bool = False

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

    # Split each optimizer step into this many forward/backward passes. The EP dispatch buffers are
    # `capacity_factor * tokens_per_shard * top_k` rows, and `tokens_per_shard` is the microbatch's
    # local slice, so this is the lever that fits a shape whose routing capacity does not.
    # It changes MFU, so hold it constant across arms of a comparison.
    microbatches: int = 1


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
    # via the iris.hooks.multigpu_main supervisor instead of one process per node.
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
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)))
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
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=model_config.num_shared_experts if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
        sliding_window=model_config.sliding_window,
        global_every=model_config.global_every,
        local_kv_heads=model_config.local_kv_heads,
        global_kv_heads=model_config.global_kv_heads,
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
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_bias = -qb_betas
    new_bias = new_bias - jnp.mean(new_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_bias)


def _optimizer_state_to_memory_kind(tree, memory_kind: str):
    """Move named-sharded optimizer arrays to a JAX memory kind."""

    def _move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        sharding = jax.typeof(leaf).sharding
        mesh = getattr(sharding, "mesh", None)
        if mesh is None or len(getattr(mesh, "axis_names", ())) == 0:
            # Scalar optimizer metadata carries no named mesh and is negligible in HBM.
            return leaf
        return jax.device_put(leaf, sharding.with_memory_kind(memory_kind))

    return jax.tree.map(_move, tree)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    offload_opt_state: bool = False,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    num_moe_layers = model_config.num_layers
    opt_state = optimizer.init(params)
    if offload_opt_state:
        opt_state = _optimizer_state_to_memory_kind(opt_state, "pinned_host")
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _drop_metrics(
    dropped_assignments: jax.Array,
    *,
    batch_size: int,
    sequence_length: int,
    top_k: int,
    num_layers: int,
) -> dict[str, int | float]:
    # Global assignment totals can exceed int32; float32 would also round large drop counts.
    dropped_assignments_host = int(dropped_assignments)
    total_assignments = batch_size * sequence_length * top_k * num_layers
    return {
        "moe/dropped_assignments": dropped_assignments_host,
        "moe/drop_fraction": dropped_assignments_host / total_assignments,
    }


_ROUTING_COUNTS_KEY = "train/router/routing_counts_per_layer"
_ROUTING_HIST_SUFFIX = "/routing_hist"
# Router metrics arrive as a bare dict of raw arrays with no reduction metadata, so gradient
# accumulation has to classify every key by hand. Counts must sum: `_drop_metrics` divides
# `moe/dropped_assignments` by the FULL batch's assignment total, so folding it as a mean would
# understate the drop rate by exactly the microbatch count, silently. Every other scalar is a
# per-token or per-layer mean. Routing histograms are a pure function of the counts, so they are
# rebuilt from the summed counts rather than folded field by field.
_SUMMED_METRICS: frozenset[str] = frozenset({"moe/dropped_assignments", _ROUTING_COUNTS_KEY})


def _slice_microbatch(batch: GrugLmExample, start: int, size: int) -> GrugLmExample:
    """Take one microbatch out of a batch, leaving the attention mask intact.

    Only a batch-independent mask survives the slice untouched. Segment ids, THD segment metadata,
    and precomputed FA4 bounds are all shaped [B, ...], so slicing the tokens without them
    misaligns attention. Reject those rather than guess at the right slice.
    """
    mask = batch.attn_mask
    if any(field is not None for field in (mask.segment_ids, mask.thd_segment_metadata, mask.fa4_bounds)):
        raise ValueError(
            "microbatches > 1 needs a batch-independent attention mask; "
            "segment ids, THD metadata, and FA4 bounds are all per-example"
        )
    return dataclasses.replace(
        batch,
        tokens=jax.lax.slice_in_dim(batch.tokens, start, start + size, axis=0),
        loss_weight=jax.lax.slice_in_dim(batch.loss_weight, start, start + size, axis=0),
    )


def _fold_metrics(per_microbatch: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-microbatch metric dicts into one dict for the whole batch.

    Raises on a key it cannot classify, so a metric added later cannot quietly land in the wrong
    bucket. See `_SUMMED_METRICS` for why the distinction matters.
    """
    folded: dict[str, Any] = {}
    for key in per_microbatch[0]:
        values = [metrics[key] for metrics in per_microbatch]
        if key in _SUMMED_METRICS:
            folded[key] = sum(values[1:], start=values[0])
        elif key.endswith(_ROUTING_HIST_SUFFIX):
            continue  # rebuilt below from the summed counts
        elif values[0] is None:
            folded[key] = None
        elif isinstance(values[0], jax.Array):
            folded[key] = sum(values[1:], start=values[0]) / len(values)
        else:
            raise TypeError(f"metric {key!r} has unfoldable type {type(values[0]).__name__}")

    summed_counts = folded[_ROUTING_COUNTS_KEY]
    for key in per_microbatch[0]:
        if key.endswith(_ROUTING_HIST_SUFFIX):
            layer = int(key.split("/")[-2].removeprefix("layer_"))
            folded[key] = _histogram_from_expert_counts(summed_counts[layer])
    return folded


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    offload_opt_state: bool = False,
    microbatches: int = 1,
):
    if microbatches < 1:
        raise ValueError(f"microbatches must be positive, got {microbatches}")
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
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        def loss_fn(params, micro):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                micro.tokens,
                micro.loss_weight,
                mask=micro.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        if microbatches == 1:
            (loss, summarized_metrics), grads = grad_fn(qb_params, batch)
        else:
            # Accumulate in place rather than collecting a list: the gradient tree is parameter
            # sized, so holding one per microbatch would cost more memory than the smaller
            # activations save. Adding each microbatch's grads immediately lets XLA free them.
            micro_size = batch.tokens.shape[0] // microbatches
            loss = None
            grads = None
            per_microbatch = []
            for index in range(microbatches):
                micro = _slice_microbatch(batch, index * micro_size, micro_size)
                (micro_loss, micro_metrics), micro_grads = grad_fn(qb_params, micro)
                loss = micro_loss if loss is None else loss + micro_loss
                grads = micro_grads if grads is None else jax.tree_util.tree_map(jnp.add, grads, micro_grads)
                per_microbatch.append(micro_metrics)
            # Each microbatch loss is already a token mean over an equal slice, so averaging the
            # means reproduces the whole-batch mean, and likewise for its gradient.
            loss = loss / microbatches
            grads = jax.tree_util.tree_map(lambda leaf: leaf / microbatches, grads)
            summarized_metrics = _fold_metrics(per_microbatch)
        metrics = {"train/loss": loss, **summarized_metrics}
        opt_state_in = (
            _optimizer_state_to_memory_kind(state.opt_state, "device") if offload_opt_state else state.opt_state
        )
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

        if offload_opt_state:
            opt_state = _optimizer_state_to_memory_kind(opt_state, "pinned_host")

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
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
        watch_config=watch_config if watch_config.is_enabled else None,
        offload_opt_state=config.trainer.offload_opt_state,
        microbatches=config.trainer.microbatches,
    )
    if trainer.train_batch_size % config.trainer.microbatches != 0:
        raise ValueError(
            f"train_batch_size={trainer.train_batch_size} must divide evenly into "
            f"microbatches={config.trainer.microbatches}"
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
                offload_opt_state=config.trainer.offload_opt_state,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id) if config.trainer.save_checkpoints else None
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
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
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
            )

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
        if profiler_enabled:
            state_callbacks.add_hook(
                profiler_cfg.build(
                    str(trainer.log_dir / run_id / "profiler"),
                    run_id=run_id,
                    num_steps=profiler_num_steps,
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

                if not jnp.isfinite(metrics["train/loss"]):
                    raise RuntimeError(f"Non-finite loss ({float(metrics['train/loss'])}) at step {int(state.step)}.")
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )
                    if "moe/dropped_assignments" in metrics:
                        drop_metrics = _drop_metrics(
                            metrics["moe/dropped_assignments"],
                            batch_size=batch.tokens.shape[0],
                            sequence_length=batch.tokens.shape[1],
                            top_k=config.model.num_experts_per_token,
                            num_layers=config.model.num_layers,
                        )
                        levanter.tracker.log(drop_metrics, step=step)

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

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    # Dispatch snapshots os.environ for the child task, so apply the hero defaults first.
    _apply_hero_ep_runtime_defaults()
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
