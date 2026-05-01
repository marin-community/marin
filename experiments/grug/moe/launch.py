# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # If a preempted TPU gets rescheduled in a different region, scan all
    # regional buckets to find and resume from the latest checkpoint.
    enable_cross_region_ckpt_read: bool = False
    # Diagnostic for issue #5319: explicitly load from a different GCS path
    # than where the checkpointer will save to. When set, the trainer reads
    # checkpoints from this path on resume but writes new checkpoints to
    # ``output_path/checkpoints``. Used to test whether the bug is tied to
    # load_path == save_path.
    load_checkpoint_path_override: str | None = None
    # Diagnostic for issue #5319: when False, Checkpointer.__init__ skips the
    # discover_latest_checkpoint -> _load_metadata -> _last_temporary_checkpoint
    # path on startup. Used to bisect whether that path's side effects (state
    # divergence between worker 0 and others) are the bug trigger when load
    # path == save path.
    delete_old_temp_checkpoints: bool = True


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def _find_checkpoint_across_regions(output_path: str) -> str | None:
    """Search all regional marin buckets for the latest checkpoint.

    Scans each region for checkpoint subdirectories with metadata.json,
    reads the step number. Returns the gs:// checkpoints directory of the
    region with the highest step, but only if that region is different from
    the local region and has a strictly higher step. Returns None if local
    already has the best checkpoint (to avoid overriding the trainer's
    normal checkpoint discovery, which also finds temporary checkpoints).
    """
    import json

    from rigging.filesystem import REGION_TO_DATA_BUCKET

    if not output_path.startswith("gs://"):
        return None
    parts = output_path.split("/", 3)
    if len(parts) < 4:
        return None
    local_bucket = parts[2]
    suffix = parts[3]
    checkpoint_suffix = os.path.join(suffix, "checkpoints")

    import gcsfs

    fs = gcsfs.GCSFileSystem()
    best_step = -1
    best_path = None
    local_step = -1

    for bucket in REGION_TO_DATA_BUCKET.values():
        candidate = f"{bucket}/{checkpoint_suffix}"
        try:
            subdirs = fs.ls(candidate)
        except FileNotFoundError:
            continue
        for subdir in subdirs:
            metadata_path = f"{subdir}/metadata.json"
            try:
                with fs.open(metadata_path) as f:
                    metadata = json.load(f)
                step = int(metadata.get("step", -1))
                has_data = fs.exists(f"{subdir}/manifest.ocdbt") or fs.exists(f"{subdir}/d")
                if not has_data:
                    continue
                if bucket == local_bucket:
                    local_step = max(local_step, step)
                if step > best_step:
                    best_step = step
                    best_path = f"gs://{candidate}"
            except Exception:
                continue

    # Only return cross-region path if it's strictly better than local.
    # Otherwise let the trainer discover its own checkpoints (including
    # temporary ones that may not have metadata.json).
    if best_step > local_step and best_path and local_bucket not in best_path:
        return best_path
    return None


_5319_STAGING_PREFIX = "gs://marin-us-central1/larry/5319_staging"


def _stage_checkpoint_for_5319_workaround(checkpoint_base: str, output_path: str) -> str | None:
    """Workaround for issue #5319: stage the latest checkpoint to a separate
    GCS prefix so the trainer's load_path is OUTSIDE ``output_path``.

    Empirically, multi-host resumes hit a ``scheckne`` halt at the first
    broadcast collective after resume whenever the load source lives anywhere
    under the same ``output_path`` as the save target. Even with the save base
    empty and load_path != save_path, a sibling load (e.g. load from
    ``<output_path>/_load_staging_5319/`` and save to
    ``<output_path>/checkpoints/``) still trips the bug. The fix is to stage
    the loaded checkpoint to a global prefix entirely outside ``output_path``.

    Suspected (not yet pinpointed) mechanism: tensorstore / OCDBT / fsspec keep
    some state (connection pool, manifest cache, gcsfs listing cache) keyed at
    a path prefix that includes ``output_path``. Sharing that prefix between
    load and save corrupts the multi-host launch sequence.

    Robust to retries: looks for the latest step-* checkpoint in BOTH
    ``checkpoint_base`` and the staging prefix and uses whichever has the
    highest step. So if a prior attempt was interrupted at any point —
    immediately after copy, after delete-original, after partial new save — the
    next invocation picks up the latest available checkpoint and re-stages.

    Returns the staging prefix to pass as ``TrainerConfig.load_checkpoint_path``,
    or None if there's no checkpoint anywhere to stage (fresh run).
    """
    import json
    import logging

    import gcsfs
    import jax

    from levanter.utils.jax_utils import multihost_broadcast_sync

    logger = logging.getLogger(__name__)

    if not checkpoint_base.startswith("gs://"):
        return None  # local-fs case can't hit the multi-host bug

    fs = gcsfs.GCSFileSystem()

    is_source = jax.process_index() == 0
    # Stage outside output_path: under a global staging prefix, namespaced by
    # the basename of output_path so concurrent runs don't collide.
    output_basename = os.path.basename(output_path.rstrip("/"))
    staging_root = f"{_5319_STAGING_PREFIX}/{output_basename}"

    def _strip_scheme(path: str) -> str:
        return path[len("gs://") :] if path.startswith("gs://") else path

    base_no_scheme = _strip_scheme(checkpoint_base)
    staging_no_scheme = _strip_scheme(staging_root)

    def _find_latest(parent_no_scheme: str) -> tuple[int, str] | None:
        """Find the highest-step step-* subdir under ``parent`` with valid metadata.

        Returns (step, subdir_no_scheme) or None.
        """
        try:
            subdirs = fs.ls(parent_no_scheme, detail=False)
        except FileNotFoundError:
            return None
        best: tuple[int, str] | None = None
        for subdir in subdirs:
            if "/step-" not in subdir:
                continue
            try:
                with fs.open(f"{subdir}/metadata.json") as f:
                    metadata = json.load(f)
                step = int(metadata.get("step", -1))
                if step < 0:
                    continue
                if best is None or step > best[0]:
                    best = (step, subdir)
            except Exception:
                continue
        return best

    if is_source:
        base_latest = _find_latest(base_no_scheme)
        staging_latest = _find_latest(staging_no_scheme)

        if base_latest is None and staging_latest is None:
            payload = {"staged": False}  # fresh run
        else:
            # Pick the higher-step source. Tie-break to base (newer save attempt).
            use_base = base_latest is not None and (staging_latest is None or base_latest[0] >= staging_latest[0])

            if use_base:
                src_step, src_subdir = base_latest  # type: ignore[misc]
                dst_subdir = os.path.join(staging_no_scheme, f"step-{src_step}")
                # Always clear staging before re-copying so we don't mix step dirs.
                try:
                    if fs.exists(staging_no_scheme):
                        logger.info("issue #5319 workaround: clearing staging at %s", staging_root)
                        fs.rm(staging_no_scheme, recursive=True)
                except Exception as exc:
                    logger.warning("issue #5319 workaround: failed to clear staging: %s", exc)

                logger.info("issue #5319 workaround: server-side copy %s -> gs://%s", src_subdir, dst_subdir)
                fs.cp(src_subdir, dst_subdir, recursive=True)
                logger.info("issue #5319 workaround: staging copy complete")
            else:
                src_step, _ = staging_latest  # type: ignore[misc]
                logger.info("issue #5319 workaround: staging already has step-%d; skipping re-copy", src_step)

            # Always empty checkpoint_base before training starts so the trainer's
            # save target is fresh. We have step-{src_step} preserved in staging
            # at this point, so removing checkpoint_base content is safe.
            try:
                if fs.exists(base_no_scheme):
                    logger.info("issue #5319 workaround: clearing save base at %s", checkpoint_base)
                    fs.rm(base_no_scheme, recursive=True)
                    logger.info("issue #5319 workaround: save base cleared")
            except Exception as exc:
                logger.warning("issue #5319 workaround: failed to clear save base: %s", exc)

            payload = {"staged": True}
    else:
        payload = None

    payload = multihost_broadcast_sync(payload, is_source=is_source)
    if not payload.get("staged", False):
        return None
    return staging_root


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    checkpoint_base = os.path.join(config.output_path, "checkpoints")

    # Diagnostic override (issue #5319) takes precedence: load from a different
    # GCS path than where we save.
    if config.load_checkpoint_path_override is not None:
        load_path = config.load_checkpoint_path_override
    else:
        # Search all regions for an existing checkpoint (handles cross-region resume).
        load_path = _find_checkpoint_across_regions(config.output_path) if config.enable_cross_region_ckpt_read else None

    # Issue #5319 workaround: if load_path is None, the trainer would default to
    # loading from checkpoint_base. That same-path resume reliably hits a
    # multi-host TPU `scheckne` halt at the first broadcast after resume. Stage
    # the latest checkpoint to a sibling prefix so load_path != save_path.
    if load_path is None:
        load_path = _stage_checkpoint_for_5319_workaround(checkpoint_base, config.output_path)

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint_path=load_path,
        checkpointer=CheckpointerConfig(
            base_path=checkpoint_base,
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 10000}],
            delete_old_temp_checkpoints=config.delete_old_temp_checkpoints,
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# Compute-optimal sweep: each (dim, budget) pair sits on the N*(C) frontier.
# Budgets derived from inverting N*(C) = 1.09e-2 * C^0.535.
COMPUTE_OPTIMAL_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
    (1280, 2.83e19),
]

compute_optimal_steps: list[ExecutorStep] = []
for _dim, _budget in COMPUTE_OPTIMAL_CONFIGS:
    _model, _optimizer, _batch, _steps = build_from_heuristic(budget=_budget, hidden_dim=_dim)
    _run_id = f"moe-v16-compute-opt-d{_dim}-{_budget:.2e}"

    compute_optimal_steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                steps=versioned(_steps),
                batch_size=versioned(_batch),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["compute-optimal", f"d={_dim}", f"budget={_budget:.0e}"],
                    group="compute-optimal-sweep",
                    name=_run_id,
                ),
                optimizer=versioned(_optimizer),
                grug_trainer=versioned(
                    GrugTrainerConfig(
                        z_loss_weight=1e-4,
                        ema_beta=None,
                        log_every=1,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        eval_batch_size=512,
                        steps_per_eval=1000,
                        max_eval_batches=8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            ),
        )
    )

# Public alias for consumers that import GRUG_MOE_TRIAL_MODEL by name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = compute_optimal_steps[0].config.model


if __name__ == "__main__":
    executor_main(
        steps=compute_optimal_steps,
        description="Compute-optimal sweep: d512/d768/d1024/d1280 at N*(C) frontier.",
    )
