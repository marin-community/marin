# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training loop for the minimalist dense transformer (``model_basic``).

Mirrors ``experiments/grug/moe/train.py`` but stripped of all MoE machinery:
no QB router biases, no router z-loss / capacity metrics, no ``pending_qb_betas``
in the train state. Data/mesh/eval/flop plumbing is reused from ``train.py`` since
it is model-agnostic.
"""

import dataclasses
import functools
import logging
import os
import time

import fsspec
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.eval import cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe import model_basic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.model_basic import Transformer
from experiments.grug.moe.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _make_mixture_stage_callback,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
)
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

logger = logging.getLogger(__name__)


def _basic_compute_flops(model_config: GrugModelConfig) -> tuple[float, dict[str, float]]:
    """Analytic FLOPs for the basic dense transformer.

    Differs from the MoE ``_compute_flops``: the MLP is a plain two-matrix GELU
    block (no GLU gate, so 2 matmuls not 3), and attention is dropped entirely
    when ``SCALE_MLP_ONLY`` selects attention-free blocks.
    """
    h = model_config.hidden_dim
    inter = model_config.intermediate_dim
    seq = model_config.max_seq_len
    per_layer = 2 * 2 * h * inter  # up + down projections, 2 FLOPs each, no gate
    if not model_basic.mlp_only_enabled():
        head_dim = h / model_config.num_heads
        qkv_proj = 2 * h * (model_config.num_heads * head_dim + 2 * model_config.num_kv_heads * head_dim)
        dense_proj = 2 * h * h
        seq_flops = 2 * seq**2 * model_config.num_heads * head_dim
        seq_flops += 3 * seq * seq * model_config.num_heads
        seq_flops += 2 * seq * seq * head_dim * model_config.num_heads
        per_layer += qkv_proj + dense_proj + seq_flops / seq
    lm_head = 2 * h * model_config.vocab_size
    flops_per_token = model_config.num_layers * per_layer + lm_head
    flops_per_example = 3 * flops_per_token * seq
    return flops_per_example, {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }


@register_dataclass
@dataclasses.dataclass(frozen=True)
class BasicTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None


def _zero1_shard_leaf(x):
    """Shard one array's leading axis over the 'expert' (data-parallel) axis.

    Replicated -> expert-sharded is a local slice (no communication) since the
    replicated value already lives on every device. Scalars and leaves whose
    leading axis isn't divisible by the expert size are left as-is.
    """
    if not isinstance(x, jax.Array) or x.ndim == 0:
        return x
    mesh = get_abstract_mesh()
    expert_size = mesh.shape["expert"] if "expert" in mesh.axis_names else 1
    if expert_size <= 1 or x.shape[0] % expert_size != 0:
        return x
    return reshard(x, P("expert", *([None] * (x.ndim - 1))))


def _zero1_shard_tree(tree):
    """ZeRO-1: shard a params-shaped pytree's leaves over the 'expert' axis."""
    return jax.tree_util.tree_map(_zero1_shard_leaf, tree)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    zero1: bool = False,
) -> BasicTrainState:
    params = mp.cast_to_param(model_basic.Transformer.init(model_config, key=key))
    opt_state = optimizer.init(params)
    if zero1:
        opt_state = _zero1_shard_tree(opt_state)
    return BasicTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
    )


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    zero1: bool = False,
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
    def train_step(state: BasicTrainState, batch, *, compute_watch: bool = False):
        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        metrics = {"train/loss": loss}
        if zero1:
            # ZeRO-1: run the whole Adam update in expert-sharded space. Resharding
            # grads/params from replicated -> expert-shard is a local slice (no comm);
            # the update keeps mu/nu/updates expert-sharded; then all-gather the updates
            # back to the replicated param layout — the single collective per step.
            sharded_grads = _zero1_shard_tree(grads)
            sharded_params = _zero1_shard_tree(state.params)
            updates, opt_state = optimizer.update(sharded_grads, state.opt_state, sharded_params)
            # all-gather updates from expert-shard back to the replicated param layout.
            # Under an explicit mesh with_sharding_constraint only asserts, so reshard.
            updates = jax.tree_util.tree_map(
                lambda u, ref: reshard(u, jax.typeof(ref).sharding.spec),
                updates,
                state.params,
            )
        else:
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        if ema_beta is None:
            ema_params = None
        else:
            if state.ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                state.ema_params,
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
                params=state.params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )
        return next_state, metrics, watch_stats

    return train_step


def _run_grug_basic_local(config: GrugRunConfig) -> None:
    """Entry point for the minimalist dense training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    zero1 = os.environ.get("SCALE_ZERO1") == "1"
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
        zero1=zero1,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

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
        train_loader = build_train_loader(train_dataset, batch_schedule=batch_schedule, mesh=mesh)

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
                zero1=zero1,
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
        dump_grug_state_sharding_run_artifact(
            state,
            log_dir=trainer.log_dir,
            run_id=run_id,
            path_override=config.trainer.sharding_dump_path,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _basic_compute_flops(config.model)
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

        state_callbacks = StateCallbackRunner[BasicTrainState](
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

        # Sync/NaN-check/log only every SCALE_SYNC_EVERY steps so the host can dispatch
        # steps ahead and keep the GPU fed (removes the per-step block_until_ready stall).
        # A host-side step counter avoids the per-iteration int(state.step) device syncs.
        sync_every = max(1, int(os.environ.get("SCALE_SYNC_EVERY", "1")))
        # SCALE_TIME_EXCL_LOAD=1 subtracts data-loading wall time from the step timer
        # (the old measurement boundary, which started timing after next(iterator)).
        excl_load = os.environ.get("SCALE_TIME_EXCL_LOAD") == "1"
        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0
        local_step = int(state.step)
        window_start = time.perf_counter()
        window_steps = 0
        window_load = 0.0

        try:
            while local_step < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                window_load += iterator.this_load_time
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and local_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                local_step += 1
                window_steps += 1

                if local_step % sync_every != 0 and local_step < trainer.num_train_steps:
                    continue

                jax.block_until_ready(metrics["train/loss"])
                if jnp.isnan(metrics["train/loss"]):
                    logger.error(f"NaN loss at step {local_step}. Stopping training.")
                    break
                window_wall = time.perf_counter() - window_start
                duration = (window_wall - (window_load if excl_load else 0.0)) / window_steps
                step = local_step - 1
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)
                window_start = time.perf_counter()
                window_steps = 0
                window_load = 0.0

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=local_step)
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=local_step, force=True)
                checkpointer.wait_until_finished()
            _upload_profiler_trace(trainer, run_id)

    levanter.tracker.current_tracker().finish()


def _upload_profiler_trace(trainer, run_id: str) -> None:
    """Copy the node-local profiler trace to ``PROFILE_UPLOAD_URI`` (durable object store).

    CoreWeave pods write the JAX profiler trace to local disk and have no outbound
    internet for the perfetto link, so we push the trace to R2/S3 ourselves. No-op
    unless ``PROFILE_UPLOAD_URI`` is set; only process 0 uploads.
    """
    uri = os.environ.get("PROFILE_UPLOAD_URI")
    if not uri or jax.process_index() != 0:
        return
    local_dir = str(trainer.log_dir / run_id / "profiler")
    local_fs = fsspec.filesystem("file")
    if not local_fs.exists(local_dir):
        logger.warning("No profiler trace at %s; nothing to upload.", local_dir)
        return
    dest_fs, dest_root = fsspec.core.url_to_fs(uri)
    files = local_fs.find(local_dir)
    logger.info("Uploading %d profiler file(s) to %s", len(files), uri)
    for f in files:
        rel = os.path.relpath(f, local_dir)
        dest_fs.put_file(f, f"{dest_root.rstrip('/')}/{rel}")
    logger.info("Profiler trace uploaded to %s", uri)


def run_grug_basic(config: GrugRunConfig) -> None:
    """Dispatch minimalist dense training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_basic_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
    )


__all__ = [
    "BasicTrainState",
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug_basic",
]
