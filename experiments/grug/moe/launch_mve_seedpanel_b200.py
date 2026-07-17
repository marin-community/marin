# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE seed-repeat noise panel on cw-us-east-08a GB200s (issue #7067).

GPU re-target of ``launch_mve_seedpanel`` after the us-central2 v4 reservation
was found fully held by a 2048-chip hero run. Training math is IDENTICAL to the
verified swarm config (model / optimizer / steps / batch / seq / phase boundary /
target_budget / seeds are imported from ``launch_mve_seedpanel`` unchanged);
only placement and I/O change:

- Training data: the CW mirror of the datakit store, resolved ABSOLUTELY at
  config-build time against ``marin_prefix()`` (cw-us-east-08a task pods carry
  ``MARIN_PREFIX=s3://marin-us-east-02a/marin``).
- Resources: GB200 GPUs instead of v4-8 TPUs (``--gpus-per-run``, default 1).
- No in-training paloma/uncheatable validation: those tokenized caches live in
  GCS us-central2 and are not mirrored on CW. Evals do not touch training
  state, so the training stream is unchanged; the panel readout is the
  post-hoc ``eval_logprob`` suite either way.
- ``attention_implementation="gpu_fa4_cute"``: grug's ``None`` default resolves
  to TPU splash on TPU but to *reference* attention on GPU, which materializes
  the full [B, H, S, S] score matrix (64GiB at 512x4x4096x4096 — OOM-thrashed
  the first smoke). ``gpu_fa4_cute`` is the segmented FA4/CuTe flash kernel the
  CW GPU canary uses, with explicit SM100/B200 tile tuning.
- CAVEAT (accepted by rav 2026-07-16): hardware numerics differ from the v4
  swarm (Blackwell matmuls, FA4 flash attention vs TPU splash, different
  reduction orders), so the measured seed-variance is under B200 numerics.

Modes:
- ``--dry-run``: parity checks + config print, no side effects.
- ``--direct --index N [--steps S] [--run-suffix smoke]``: run ONE panel run
  in-process (no fray dispatch) — the shape used for the sizing smoke inside a
  federated ``iris job run --gpu GB200xN`` job. ``--steps`` only shortens the
  trainer; the data config (budgets, slices, shuffle, boundary) stays exactly
  the panel's, so smoke throughput is measured under panel I/O.
- default: StepRunner parent — dispatches all 10 runs as fray GPU jobs and
  babysits them (run on a cw-us-east-08a CPU node).
"""

import argparse
import dataclasses
import json
import logging
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe.launch import GrugMoeLaunchConfig, _resolve_tracker, run_grug_moe_trial
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

# The swarm model with the GPU flash-attention backend selected (see module
# docstring). Backend selection only — every shape/math field stays identical.
B200_MODEL = dataclasses.replace(tpu_panel.SWARM_MODEL, attention_implementation="gpu_fa4_cute")

DEFAULT_GPUS_PER_RUN = 1
# One gb200-4x node = 4 GPUs / 144 vCPU / 960GB; request a proportional host share.
HOST_CPU_PER_GPU = 32
HOST_RAM_GB_PER_GPU = 200
HOST_DISK = "200g"


def train_resources(gpus_per_run: int) -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "GB200",
        count=gpus_per_run,
        cpu=HOST_CPU_PER_GPU * gpus_per_run,
        ram=f"{HOST_RAM_GB_PER_GPU * gpus_per_run}g",
        disk=HOST_DISK,
        preemptible=False,
    )


def _abs_bucket_component(bucket: str) -> DatasetComponent:
    """Datakit bucket component with the cache dir absolutized against MARIN_PREFIX.

    On cw-us-east-08a this resolves to the CW mirror
    ``s3://marin-us-east-02a/marin/datakit/store_8ac06c74/...`` — training data
    never crosses clouds.
    """
    return DatasetComponent(
        source=None,
        cache_dir=prefix_join(marin_prefix(), datakit_mix._bucket_path(bucket)),
        format=TextLmDatasetFormat(),
        tags=[bucket],
        flat_cache=True,
    )


def _abs_datakit_components() -> dict[str, DatasetComponent | ConcatDatasetComponent]:
    direct = {
        bucket: _abs_bucket_component(bucket) for bucket, _, _ in datakit_mix._BUCKET_PHASE_WEIGHTS if bucket != "tail"
    }
    return {
        **direct,
        "tail": ConcatDatasetComponent(
            children={bucket: _abs_bucket_component(bucket) for bucket in datakit_mix._TAIL_BUCKETS},
            tags=["tail"],
        ),
    }


def _panel_data_config() -> LmDataConfig:
    """The TPU panel's data config with CW-absolute paths and no validation.

    Budgets, boundary, block size, and shuffle policy are ALWAYS the swarm's,
    independent of trainer step count — a short smoke sees exactly the panel's
    dataset slices and phase-0 stream.
    """
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=_abs_datakit_components(),
        train_weights=[
            (0, datakit_mix._phase_weights(0)),
            (tpu_panel.PHASE_1_START_STEP, datakit_mix._phase_weights(1)),
        ],
        auto_build_caches=False,
        mixture_block_size=tpu_panel.MIXTURE_BLOCK_SIZE,
        target_budget=tpu_panel.TARGET_BUDGET_TOKENS,
        experiment_budget=tpu_panel.EXPERIMENT_BUDGET_TOKENS,
    )


def _panel_launch_config(
    *,
    index: int,
    output_path: str,
    total_steps: int,
    resources: ResourceConfig,
    run_suffix: str | None = None,
) -> GrugMoeLaunchConfig:
    run_id = tpu_panel.RUN_ID_TEMPLATE.format(index=index)
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    seed = tpu_panel.SEED_BASE + index
    return GrugMoeLaunchConfig(
        model=B200_MODEL,
        data=_panel_data_config(),
        output_path=output_path,
        run_id=run_id,
        resources=resources,
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "seedpanel", "mve", "d512", "b200", f"seed{seed}"],
            group=tpu_panel.WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,  # no in-training validation on CW (see module docstring)
    )


def _run_direct(config: GrugMoeLaunchConfig) -> None:
    """``run_grug_moe_trial`` minus the fray dispatch: train in this process.

    Mirrors ``run_grug_moe_trial``'s TrainerConfig assembly exactly, then calls
    ``_run_grug_local`` so the federated job's own GPUs do the training.
    """
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=10), keep=None),
            config.output_path,
        ),
    )
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.grug_trainer, trainer=trainer),
        eval=config.eval,
    )
    _run_grug_local(run_config)


def build_panel_step(index: int, *, gpus_per_run: int, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return _panel_launch_config(
            index=index,
            output_path=ctx.output_path,
            total_steps=tpu_panel.STEPS,
            resources=ctx.runtime_arg("train_resources"),
        )

    return ArtifactStep(
        name=tpu_panel.STEP_NAME_TEMPLATE.format(index=index),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": train_resources(gpus_per_run)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE run in-process (smoke / federated GPU job)")
    parser.add_argument("--index", type=int, default=0, help="panel run index for --direct")
    parser.add_argument("--steps", type=int, default=None, help="step override for --direct (smoke sizing)")
    parser.add_argument("--run-suffix", type=str, default=None, help="run-id suffix for --direct (e.g. 'smoke')")
    parser.add_argument("--gpus-per-run", type=int, default=DEFAULT_GPUS_PER_RUN)
    parser.add_argument("--max-concurrent", type=int, default=14)
    args = parser.parse_args()

    summary = tpu_panel.verify_swarm_parity()
    summary["gpus_per_run"] = args.gpus_per_run
    summary["cluster"] = "cw-us-east-08a"
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    summary["in_training_validation"] = False

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity checks passed; nothing launched.")
        return

    if args.direct:
        total_steps = args.steps or tpu_panel.STEPS
        run_id = tpu_panel.RUN_ID_TEMPLATE.format(index=args.index)
        leaf = f"{run_id}-{args.run_suffix}" if args.run_suffix else run_id
        output_path = prefix_join(marin_prefix(), f"users/rav/grug/{leaf}/dev")
        logger.info("direct mode: run %s, %d steps, output %s", leaf, total_steps, output_path)
        config = _panel_launch_config(
            index=args.index,
            output_path=output_path,
            total_steps=total_steps,
            resources=train_resources(args.gpus_per_run),
            run_suffix=args.run_suffix,
        )
        _run_direct(config)
        return

    print(json.dumps(summary, indent=2))
    steps = [build_panel_step(i, gpus_per_run=args.gpus_per_run).lower() for i in range(tpu_panel.NUM_RUNS)]
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
