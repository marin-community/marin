# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE seed-repeat noise panel on cw-rno2a H100s (issue #7067).

H100 re-target of ``launch_mve_seedpanel_b200`` (rav directive 2026-07-17):
the B200 panel was stopped at ~1-2% trained to free the GB200 fleet. Training
math is IDENTICAL (swarm constants imported unchanged via the B200/TPU panel
launchers); differences from the B200 panel:

- Placement: cw-rno2a ``gd-8xh100ib-i128`` nodes, one node per run,
  ``H100x8`` (80GB HBM ⇒ the ~521GiB step footprint needs 8-way batch
  sharding; global batch stays 512x4096, data-axis=8).
- Attention: ``gpu_fa4_cute`` is kept — FA4/CuTe has a dedicated SM90 path
  (``flash4_cute_kernel_config`` arch_family 9 with an SM90 backward
  schedule); it is the CW H100 canary's default implementation.
- FRESH state (numerics homogeneity): new run ids / W&B names
  ``rav_mve_seedpanel_h100_NN`` in group ``rav_mve_seedpanel_h100`` and a new
  checkpoint prefix, so nothing can resume from the killed B200 attempts'
  checkpoints. Seeds stay 1000-1009. The panel measures seed variance under
  H100 numerics.
- Data: same CW mirror reads (rno2a task pods carry the same
  ``MARIN_PREFIX=s3://marin-us-east-02a/marin`` and object-store creds);
  cross-DC within CoreWeave.

Submission is DIRECT to the rno2a controller (``iris --cluster=cw-rno2a``),
not federated.
"""

import argparse
import json
import logging

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugTrainerConfig

logger = logging.getLogger(__name__)

RUN_ID_TEMPLATE = "rav_mve_seedpanel_h100_{index:02d}"
STEP_NAME_TEMPLATE = "users/rav/grug/rav_mve_seedpanel_h100_{index:02d}"
WANDB_GROUP = "rav_mve_seedpanel_h100"

DEFAULT_GPUS_PER_RUN = 8  # one full gd-8xh100ib node; 512x4096 needs 8-way sharding on 80GB HBM
# Host share per node, following launch_cw_scale's proven gd-8xh100ib request.
HOST_CPU = 32
HOST_RAM = "256g"
HOST_DISK = "256g"


def train_resources(gpus_per_run: int) -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_run,
        cpu=HOST_CPU,
        ram=HOST_RAM,
        disk=HOST_DISK,
        preemptible=False,
    )


def _panel_launch_config(
    *,
    index: int,
    output_path: str,
    total_steps: int,
    resources: ResourceConfig,
    run_suffix: str | None = None,
) -> GrugMoeLaunchConfig:
    run_id = RUN_ID_TEMPLATE.format(index=index)
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    seed = tpu_panel.SEED_BASE + index
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # swarm shapes + gpu_fa4_cute; SM90-supported (see docstring)
        data=b200_panel._panel_data_config(),
        output_path=output_path,
        run_id=run_id,
        resources=resources,
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "seedpanel", "mve", "d512", "h100", f"seed{seed}"],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,  # no in-training validation on CW (see the B200 launcher docstring)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE run in-process (smoke / direct GPU job)")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="step override for --direct (smoke sizing)")
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--gpus-per-run", type=int, default=DEFAULT_GPUS_PER_RUN)
    args = parser.parse_args()

    summary = tpu_panel.verify_swarm_parity()
    summary["runs"] = [
        {"run_id": RUN_ID_TEMPLATE.format(index=i), "step_name": STEP_NAME_TEMPLATE.format(index=i), "seed": 1000 + i}
        for i in range(tpu_panel.NUM_RUNS)
    ]
    summary["gpus_per_run"] = args.gpus_per_run
    summary["cluster"] = "cw-rno2a"
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    summary["in_training_validation"] = False

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity checks passed; nothing launched.")
        return

    if not args.direct:
        raise SystemExit("H100 panel jobs are submitted directly (one iris job per --index); use --direct.")

    total_steps = args.steps or tpu_panel.STEPS
    run_id = RUN_ID_TEMPLATE.format(index=args.index)
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
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
