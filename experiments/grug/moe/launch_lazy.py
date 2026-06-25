# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-moe baseline, authored as a lazy artifact.

This is the same run as ``baseline_moe`` in ``launch.py``, written in the
artifact model: a zero-arg function returns a typed :class:`Checkpoint` handle
addressed by an explicit ``name@version``. The experiment file contains no
``ExecutorStep``, ``executor_main``, ``versioned()``, or ``this_output_path()`` —
the decisions are stated inline, and the output path is ``ctx.out``.

The data mixture (``nemotron_mix``) is still a legacy ``ExecutorStep`` catalog, so
``run()`` resolves it through the bridge (the existing ``Executor``); migrating the
tokenize catalog to handles is the next step. A golden test
(``tests/experiment/test_grug_moe_lazy_parity.py``) pins this to produce the same
materialized config as ``baseline_moe``.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import BuildContext, Checkpoint, Recipe, run

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    RESOLVED_RUN_ID,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# 1e18 compute budget, d1024 — model/optimizer/batch/steps derived from the heuristic.
_BUDGET = 1e18
_HIDDEN_DIM = 1024
_TARGET_STEPS = 2**14


def grug_moe_baseline(*, version: str = "v1") -> Checkpoint:
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )

    def build_config(ctx: BuildContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=ctx.out,
            run_id=RESOLVED_RUN_ID,
            resources=ResourceConfig.with_tpu("v5p-8"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(project="marin_moe", tags=["moe"], group="moe-iter04", name=None),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return Checkpoint(
        name="grug/4_10_baseline_moe",
        version=version,
        # resources live in the config (run_grug dispatches its own Fray TPU job),
        # so the launcher step itself runs inline — matching baseline_moe.
        recipe=Recipe(fn=run_grug_moe_trial, build_config=build_config, resources=None),
    )


if __name__ == "__main__":
    run(grug_moe_baseline())
