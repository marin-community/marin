#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume the 1e23 EP8 ragged run from an explicit checkpoint.

This is the reconstructed one-off launcher from the deleted worktree. It keeps
the run identity, output root, and ``initialize_from`` path under explicit env
control so we do not silently fall back to stale defaults.
"""

from __future__ import annotations

import dataclasses
import os
import re
import urllib.parse

import fsspec
from experiments.grug.moe.launch import (
    ExecutorStep,
    GrugEvalConfig,
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    WandbConfig,
    _baseline_batch,
    _baseline_model,
    _baseline_optimizer,
    _baseline_steps,
    _resolve_run_id,
    executor_main,
    run_grug_moe_trial,
    this_output_path,
    versioned,
)
from fray.cluster import ResourceConfig
from levanter.trainer import TrainerConfig

SOURCE_CHECKPOINT_BASE = (
    "gs://marin-us-central2/grug/"
    "moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/"
    "checkpoints"
)
COMPLETE_CHECKPOINT_MARKER = "metadata.json"
CHECKPOINT_STEP_RE = re.compile(r"step-(\d+)")


def _checkpoint_step(checkpoint_path: str) -> int:
    match = CHECKPOINT_STEP_RE.search(checkpoint_path)
    if match is None:
        raise ValueError(f"Could not parse step number from checkpoint path: {checkpoint_path}")
    return int(match.group(1))


def _latest_complete_checkpoint(checkpoint_base: str) -> str:
    fs, _, (plain_base,) = fsspec.get_fs_token_paths(checkpoint_base)
    base_scheme = urllib.parse.urlparse(checkpoint_base).scheme

    def maybe_unstrip_protocol(path: str) -> str:
        if base_scheme != "" and urllib.parse.urlparse(path).scheme == "":
            return f"{base_scheme}://{path}"
        return path

    metadata_paths = fs.glob(os.path.join(plain_base, "step-*", COMPLETE_CHECKPOINT_MARKER))
    checkpoints = [
        maybe_unstrip_protocol(metadata_path).rsplit(f"/{COMPLETE_CHECKPOINT_MARKER}", maxsplit=1)[0]
        for metadata_path in metadata_paths
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No complete checkpoints found under {checkpoint_base}")
    return max(checkpoints, key=_checkpoint_step)


INITIALIZE_FROM = os.environ.get("MARIN_GRUG_INITIALIZE_FROM") or _latest_complete_checkpoint(SOURCE_CHECKPOINT_BASE)
DEFAULT_RUN_ID = f"moe_1e23_d5120_bs2048_ep8_ragged_48l_resume{_checkpoint_step(INITIALIZE_FROM)}_clip15_restore"
RUN_ID = _resolve_run_id(DEFAULT_RUN_ID)
STEP_NAME = os.environ.get("MARIN_GRUG_STEP_NAME", f"grug/{RUN_ID}")
DESCRIPTION = os.environ.get(
    "MARIN_GRUG_DESCRIPTION",
    f"Resume the Grug MoE 1e23 ragged EP8 run from initialize_from={INITIALIZE_FROM} "
    "with max_grad_norm=1.5 and permanent checkpoint retention every 1000 steps.",
)


ragged_ep8_fix = ExecutorStep(
    name=STEP_NAME,
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(
            dataclasses.replace(
                _baseline_model,
                moe_implementation="ragged_all_to_all",
                use_array_stacked_blocks=True,
                num_layers=48,
            )
        ),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-2048", regions=["us-central2"])),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        expert_parallel=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["adamh", "qb", "sharded-qb", "gatednorm", "xsa", "zloss", "eq3e3", "ragged-fix", "resume"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(dataclasses.replace(_baseline_optimizer, max_grad_norm=1.5)),
        priority_band="production",
        grug_trainer=versioned(
            GrugTrainerConfig(
                trainer=TrainerConfig(
                    initialize_from=INITIALIZE_FROM,
                ),
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=1024,
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
        steps=[ragged_ep8_fix],
        description=DESCRIPTION,
    )
