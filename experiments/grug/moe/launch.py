# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
import re
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
from marin.training.training import (
    DEFAULT_CHECKPOINTS_PATH,
    TEMPORARY_CHECKPOINT_TTL_DAYS,
    TEMPORARY_CHECKPOINTS_PATH,
    temporary_checkpoint_base_path,
)

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix


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
    # NOT wrapped in versioned() so flipping it on doesn't change the
    # executor's content hash — flag-only opt-in.
    enable_cross_region_ckpt_read: bool = False


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _find_checkpoint_across_regions(output_path: str) -> str | None:
    """Search all regional marin buckets for the latest checkpoint.

    Scans each region for permanent and temporary checkpoint subdirectories
    with metadata.json, reads the step number. It checks the current output
    path and prior sibling output hashes for the same run id so small launcher
    fixes can still resume existing work. Returns None if the current output
    path already has the best checkpoint, letting the trainer's normal
    checkpoint discovery handle it.
    """
    import json

    import gcsfs
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    if not output_path.startswith("gs://"):
        return None
    parts = output_path.split("/", 3)
    if len(parts) < 4:
        return None
    local_bucket = parts[2]
    suffix = parts[3]
    checkpoint_suffix = os.path.join(suffix, DEFAULT_CHECKPOINTS_PATH)
    run_parent = os.path.dirname(suffix)
    run_dir = os.path.basename(suffix)
    run_match = re.fullmatch(r"(?P<run_id>.+)-[0-9a-f]{6}", run_dir)
    run_id_prefix = run_match.group("run_id") if run_match else None

    fs = gcsfs.GCSFileSystem()
    best_step = -1
    best_path: str | None = None
    current_step = -1
    seen_roots: set[str] = set()

    for bucket in REGION_TO_DATA_BUCKET.values():
        temp_checkpoint_suffix = os.path.join(
            bucket,
            "tmp",
            f"ttl={TEMPORARY_CHECKPOINT_TTL_DAYS}d",
            TEMPORARY_CHECKPOINTS_PATH,
            bucket,
            checkpoint_suffix,
        )

        root_candidates: list[tuple[str, bool]] = [
            (f"{bucket}/{checkpoint_suffix}", True),
            (temp_checkpoint_suffix, True),
        ]
        if run_id_prefix:
            glob_patterns = [
                os.path.join(bucket, run_parent, f"{run_id_prefix}-*", DEFAULT_CHECKPOINTS_PATH),
                os.path.join(
                    bucket,
                    "tmp",
                    f"ttl={TEMPORARY_CHECKPOINT_TTL_DAYS}d",
                    TEMPORARY_CHECKPOINTS_PATH,
                    bucket,
                    run_parent,
                    f"{run_id_prefix}-*",
                    DEFAULT_CHECKPOINTS_PATH,
                ),
            ]
            for pattern in glob_patterns:
                try:
                    root_candidates.extend((root, False) for root in fs.glob(pattern))
                except FileNotFoundError:
                    continue

        for candidate, is_current_root in root_candidates:
            if candidate in seen_roots:
                continue
            seen_roots.add(candidate)
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
                    if is_current_root and bucket == local_bucket:
                        current_step = max(current_step, step)
                    if step > best_step:
                        best_step = step
                        best_path = f"gs://{candidate}"
                except Exception:
                    continue

    # Only return an explicit path if the best checkpoint is outside the
    # current output roots. Otherwise let the trainer discover the current
    # permanent and temporary checkpoints itself.
    if best_step > current_step and best_path:
        return best_path
    return None


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


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Search all regions for an existing checkpoint (handles cross-region
    # resume after the parent CPU coordinator moves regions).
    load_path: str | None = (
        _find_checkpoint_across_regions(config.output_path) if config.enable_cross_region_ckpt_read else None
    )

    # Map template launch knobs onto full Levanter TrainerConfig.
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
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
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


RESOLVED_RUN_ID = _resolve_run_id("4_10_test_moe")


# Baseline: 1e18 compute budget, d1024. Model + optimizer + batch + steps are
# all derived from `MoeAdamHHeuristic`. To override any of these, swap in
# an explicit `GrugModelConfig` / `GrugMoeAdamHConfig` below.
_BASELINE_BUDGET: float = 1e18
_BASELINE_HIDDEN_DIM: int = 1024
_BASELINE_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept
# because consumers (e.g. experiments/ferries/canary_ferry.py) import it by
# name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model


baseline_moe = ExecutorStep(
    name="grug/4_10_baseline_moe",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_baseline_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/moe-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(_baseline_optimizer),
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


if __name__ == "__main__":
    executor_main(
        steps=[baseline_moe],
        description="Baseline grug MoE (QB+GN+XSA+zloss) on Nemotron mix.",
    )
