# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import math
import posixpath
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
from levanter.data import AsyncDataset, DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text import GrugLmExample, LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.fsspec_utils import join_path
from levanter.utils.jax_utils import barrier_sync, parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator
from rigging.filesystem import url_to_fs

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.model import GrugModelConfig, Transformer

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)


LiveParamMode = Literal["param", "compute_with_master"]
CompileDiagnosticMode = Literal["off", "compile", "compile_only"]


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 0.0  # Weight on logsumexp (z-loss) stabilization term.

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    model_axis_size: int = 1
    live_param_mode: LiveParamMode = "param"
    """Parameter storage mode.

    "param" keeps the train state parameter tree in the jmp param dtype.
    "compute_with_master" keeps a live compute-dtype parameter tree for
    forward/backward and a separate param-dtype master tree for optimizer state
    and updates.
    """
    compile_diagnostic: CompileDiagnosticMode = "off"
    compile_diagnostic_log_hlo: bool = False
    compile_diagnostic_progress_interval: float = 60.0
    compile_diagnostic_barrier_timeout: float = 1800.0
    sharding_audit: bool = False
    sharding_audit_only: bool = False
    sharding_audit_min_bytes: int = 1 << 20
    sharding_audit_max_entries: int = 40


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
    output_path: str | None = None
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


def _should_log_loop_metrics(*, step: int, log_every: int) -> bool:
    return step % log_every == 0


def _mirror_profiler_dir_to_output(profile_dir: Path, output_path: str | None) -> str | None:
    if output_path is None:
        return None
    if not profile_dir.exists():
        logger.warning("Grug profiler directory does not exist; skipping mirror: %s", profile_dir)
        return None

    target_url = join_path(output_path, "profiler")
    fs, target_path = url_to_fs(target_url)
    if fs.exists(target_path):
        fs.rm(target_path, recursive=True)
    fs.makedirs(target_path, exist_ok=True)

    for source_path in profile_dir.rglob("*"):
        relative_path = source_path.relative_to(profile_dir).as_posix()
        destination_path = posixpath.join(target_path.rstrip("/"), relative_path)
        if source_path.is_dir():
            fs.makedirs(destination_path, exist_ok=True)
            continue
        fs.makedirs(posixpath.dirname(destination_path), exist_ok=True)
        fs.put_file(str(source_path), destination_path)

    logger.info("Mirrored Grug profiler directory to %s", target_url)
    return target_url


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    master_params: Transformer | None
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_blocks = list(model.blocks)
    moe_idx = 0
    for i, block in enumerate(model.blocks):
        if block.mlp is None:
            continue
        new_bias = -qb_betas[moe_idx]
        new_bias = new_bias - jnp.mean(new_bias)
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_bias)
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
        moe_idx += 1
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    live_param_mode: LiveParamMode = "param",
) -> GrugTrainState:
    param_params = mp.cast_to_param(Transformer.init(model_config, key=key))
    if live_param_mode == "param":
        params = param_params
        master_params = None
    elif live_param_mode == "compute_with_master":
        params = mp.cast_to_compute(param_params)
        master_params = param_params
    else:
        raise ValueError(f"Unknown live_param_mode={live_param_mode!r}")

    num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=master_params,
        opt_state=optimizer.init(param_params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
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
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        with jax.named_scope("apply_qb_betas"):
            qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
            if state.master_params is None:
                qb_master_params = qb_params
            else:
                qb_master_params = _apply_qb_betas(state.master_params, state.pending_qb_betas)

            if ema_beta is not None:
                qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
            else:
                qb_ema_params = None

        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        with jax.named_scope("forward_backward"):
            (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        metrics = {"train/loss": loss, **summarized_metrics}
        with jax.named_scope("optimizer_update"):
            updates, opt_state = optimizer.update(grads, state.opt_state, qb_master_params)
        with jax.named_scope("apply_updates"):
            if state.master_params is None:
                params = optax.apply_updates(qb_params, updates)
                master_params = None
            else:
                master_params = optax.apply_updates(qb_master_params, updates)
                params = mp.cast_to_compute(master_params)

        with jax.named_scope("ema_update"):
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
            with jax.named_scope("watch_stats"):
                watch_stats = compute_watch_stats(
                    watch_targets=watch_targets,
                    include_norms=watch_config.include_norms,
                    include_per_parameter_norms=watch_config.include_per_parameter_norms,
                    include_histogram=watch_config.include_histograms,
                    split_scan_layers=watch_config.split_scan_layers,
                    params=qb_params,
                    grads=grads,
                    updates=updates,
                    opt_state=state.opt_state,
                    model_tree_type=type(state.params),
                )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            master_params=master_params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


def _dtype_size(dtype) -> int:
    try:
        return jnp.dtype(dtype).itemsize
    except TypeError:
        return 0


def _format_tree_path(path) -> str:
    return jax.tree_util.keystr(path).lstrip(".") or "<root>"


def _sharding_spec_and_shard_count(leaf) -> tuple[str, int]:
    sharding = getattr(leaf, "sharding", None)
    if sharding is None:
        sharding = getattr(jax.typeof(leaf), "sharding", None)
    spec = getattr(sharding, "spec", None)
    mesh = getattr(sharding, "mesh", None)
    if spec is None:
        return type(sharding).__name__ if sharding is not None else "<none>", 1

    axes: set[str] = set()
    for part in tuple(spec):
        if part is None:
            continue
        if isinstance(part, tuple):
            axes.update(str(axis) for axis in part)
        else:
            axes.add(str(part))

    shard_count = 1
    mesh_shape = getattr(mesh, "shape", {}) if mesh is not None else {}
    for axis in axes:
        shard_count *= int(mesh_shape.get(axis, 1))
    return str(spec), shard_count


def _log_sharding_audit(name: str, tree, *, min_bytes: int, max_entries: int) -> None:
    if jax.process_index() != 0:
        return

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: x is None)
    entries: list[tuple[int, int, str, tuple[int, ...], str, str]] = []
    total_bytes = 0
    replicated_large_bytes = 0
    replicated_large_count = 0
    lightly_sharded_large_bytes = 0
    lightly_sharded_large_count = 0

    for path, leaf in leaves_with_paths:
        if leaf is None or not hasattr(leaf, "shape"):
            continue
        shape = tuple(int(dim) for dim in leaf.shape)
        size = int(math.prod(shape)) if shape else 1
        nbytes = size * _dtype_size(getattr(leaf, "dtype", None))
        total_bytes += nbytes
        spec, shard_count = _sharding_spec_and_shard_count(leaf)
        if nbytes >= min_bytes and shard_count <= 1:
            replicated_large_count += 1
            replicated_large_bytes += nbytes
        if nbytes >= min_bytes and shard_count < jax.device_count():
            lightly_sharded_large_count += 1
            lightly_sharded_large_bytes += nbytes
        entries.append((nbytes, shard_count, _format_tree_path(path), shape, str(getattr(leaf, "dtype", "")), spec))

    entries.sort(reverse=True, key=lambda entry: entry[0])
    logger.info(
        "Grug sharding audit %s: leaves=%s total_global_bytes=%.3fGiB large_replicated=%s %.3fGiB "
        "large_less_than_global_device_count=%s %.3fGiB min_bytes=%s global_device_count=%s",
        name,
        len(entries),
        total_bytes / (1024**3),
        replicated_large_count,
        replicated_large_bytes / (1024**3),
        lightly_sharded_large_count,
        lightly_sharded_large_bytes / (1024**3),
        min_bytes,
        jax.device_count(),
    )
    for nbytes, shard_count, path, shape, dtype, spec in entries[:max_entries]:
        logger.info(
            "Grug sharding audit %s leaf path=%s shape=%s dtype=%s global_bytes=%.3fMiB "
            "shard_count=%s per_shard_bytes=%.3fMiB spec=%s",
            name,
            path,
            shape,
            dtype,
            nbytes / (1024**2),
            shard_count,
            nbytes / max(1, shard_count) / (1024**2),
            spec,
        )


def _run_with_progress_logging(label: str, interval: float, fn):
    start = time.perf_counter()
    reporter: subprocess.Popen | None = None
    logger.info("Grug compile diagnostic starting: %s", label)
    if interval > 0:
        reporter_code = (
            "import sys, time\n"
            f"interval = {interval!r}\n"
            f"label = {label!r}\n"
            "start = time.perf_counter()\n"
            "while True:\n"
            "    time.sleep(interval)\n"
            "    elapsed = time.perf_counter() - start\n"
            '    print(f"Grug compile diagnostic still running: {label} elapsed={elapsed:.1f}s", '
            "file=sys.stderr, flush=True)\n"
        )
        reporter = subprocess.Popen(
            [sys.executable, "-c", reporter_code],
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

    try:
        return fn()
    finally:
        if reporter is not None:
            reporter.terminate()
            try:
                reporter.wait(timeout=1)
            except subprocess.TimeoutExpired:
                reporter.kill()
                reporter.wait(timeout=1)
        logger.info("Grug compile diagnostic finished: %s elapsed=%.1fs", label, time.perf_counter() - start)


def _run_compile_diagnostic(
    *,
    train_step,
    state: GrugTrainState,
    iterator,
    mode: CompileDiagnosticMode,
    log_hlo: bool,
    progress_interval: float,
    barrier_timeout: float,
):
    if mode == "off":
        return None, False
    if mode not in ("compile", "compile_only"):
        raise ValueError("compile_diagnostic must be one of: off, compile, compile_only")

    logger.info("Grug compile diagnostic enabled: mode=%s", mode)
    batch_start = time.perf_counter()
    batch = next(iterator)
    logger.info("Grug compile diagnostic fetched batch in %.1fs", time.perf_counter() - batch_start)

    lowered = _run_with_progress_logging(
        "train_step.lower",
        progress_interval,
        lambda: train_step.lower(state, batch, compute_watch=False),
    )
    if log_hlo:
        logger.info("Grug compile diagnostic lowered StableHLO:\n%s", lowered.as_text())

    compiled = _run_with_progress_logging("train_step.compile", progress_interval, lowered.compile)
    del compiled
    logger.info("Grug compile diagnostic waiting at post-compile barrier: timeout=%.1fs", barrier_timeout)
    barrier_sync(timeout=barrier_timeout)
    logger.info("Grug compile diagnostic post-compile barrier complete")
    return batch, mode == "compile_only"


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
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    if config.model.num_heads % config.trainer.model_axis_size != 0:
        raise ValueError(
            f"num_heads={config.model.num_heads} must be divisible by "
            f"model_axis_size={config.trainer.model_axis_size}; attention shards the head axis over model"
        )

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=config.trainer.model_axis_size,
    )
    mesh_shape = {axis: int(size) for axis, size in mesh.shape.items()}
    batch_shards = mesh_shape["replica_dcn"] * mesh_shape["data"] * mesh_shape["expert"]
    logger.info("Grug compact mesh shape: %s; batch_shards=%s", mesh_shape, batch_shards)
    levanter.tracker.log_summary(
        {
            **{f"grug/mesh/{axis}": size for axis, size in mesh_shape.items()},
            "grug/mesh/batch_shards": batch_shards,
        }
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
                live_param_mode=config.trainer.live_param_mode,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        if config.trainer.sharding_audit:
            _log_sharding_audit(
                "params",
                state.params,
                min_bytes=config.trainer.sharding_audit_min_bytes,
                max_entries=config.trainer.sharding_audit_max_entries,
            )
            if state.master_params is not None:
                _log_sharding_audit(
                    "master_params",
                    state.master_params,
                    min_bytes=config.trainer.sharding_audit_min_bytes,
                    max_entries=config.trainer.sharding_audit_max_entries,
                )
            _log_sharding_audit(
                "opt_state",
                state.opt_state,
                min_bytes=config.trainer.sharding_audit_min_bytes,
                max_entries=config.trainer.sharding_audit_max_entries,
            )
            if config.trainer.sharding_audit_only:
                logger.info("Grug sharding audit only complete; skipping training.")
                levanter.tracker.current_tracker().finish()
                return

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
                callbacks.profile(
                    str(trainer.log_dir / run_id / "profiler"),
                    profiler_cfg.start_step,
                    profiler_num_steps,
                    profiler_cfg.perfetto_link,
                    profiler_options=profiler_cfg.build_jax_profile_options(),
                    stop_barrier_timeout=profiler_cfg.stop_barrier_timeout,
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
        prefetched_batch, compile_only = _run_compile_diagnostic(
            train_step=train_step,
            state=state,
            iterator=iterator,
            mode=config.trainer.compile_diagnostic,
            log_hlo=config.trainer.compile_diagnostic_log_hlo,
            progress_interval=config.trainer.compile_diagnostic_progress_interval,
            barrier_timeout=config.trainer.compile_diagnostic_barrier_timeout,
        )

        # Main optimization loop.
        try:
            current_step = int(state.step)
            if compile_only:
                logger.info("Grug compile diagnostic compile_only complete; skipping train loop.")
            while not compile_only and current_step < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    if prefetched_batch is None:
                        batch = next(iterator)
                    else:
                        batch = prefetched_batch
                        prefetched_batch = None
                step_start = time.perf_counter()
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = current_step
                current_step += 1

                jax.block_until_ready(metrics["train/loss"])

                if jnp.isnan(metrics["train/loss"]):
                    logger.error("NaN loss at step %s. Stopping training.", current_step)
                    break
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    if _should_log_loop_metrics(step=step, log_every=log_every):
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

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=current_step)
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

            if profiler_enabled and jax.process_index() == 0:
                profiler_dir = trainer.log_dir / run_id / "profiler"
                _mirror_profiler_dir_to_output(profiler_dir, config.output_path)
                logger.info("Logging Grug profiler artifact: %s", profiler_dir)
                levanter.tracker.current_tracker().log_artifact(
                    profiler_dir,
                    name=f"{run_id}-profiler",
                    type="jax_profile",
                )

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
    "CompileDiagnosticMode",
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "LiveParamMode",
    "initial_state",
    "run_grug",
]
