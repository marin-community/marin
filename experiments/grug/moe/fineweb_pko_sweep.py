# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fineweb-edu baseline and PKO comparison.

Trains on fineweb_edu_subcache_10B instead of nemotron_mix.
Tests baseline vs PKO (every 4th + last layer) at gate 1 scales.

GitHub issue: https://github.com/marin-community/marin/issues/TBD
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.pretraining_datasets.nemotron import DEFAULT_NEW_RUN_DATA_SHUFFLE

FINEWEB_DATA = add_validation_sets_to_mixture(
    lm_data_config(
        training_set=fineweb_edu_subcache_10B,
        shuffle=versioned(DEFAULT_NEW_RUN_DATA_SHUFFLE),
    ),
    default_validation_sets(tokenizer=fineweb_edu_subcache_10B.config.tokenizer),
)

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

CONFIGS: list[tuple[str, bool]] = [
    ("baseline", False),
    ("pko", True),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for label, use_pko in CONFIGS:
        for dim, budget in GATE1_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            if use_pko:
                model = dataclasses.replace(model, partial_key_offset="every_4th", last_layer_pko=True)
            run_id = f"fineweb-{label}-d{dim}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(model),
                        data=FINEWEB_DATA,
                        output_path=this_output_path(),
                        run_id=run_id,
                        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                        enable_cross_region_ckpt_read=True,
                        steps=versioned(num_steps),
                        batch_size=versioned(batch),
                        seed=versioned(0),
                        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                        tracker=WandbConfig(
                            project="dial_moe",
                            tags=["fineweb", label, f"d={dim}", f"budget={budget:.2e}"],
                            group="fineweb-pko",
                            name=run_id,
                        ),
                        optimizer=versioned(optimizer),
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
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Fineweb-edu: baseline vs PKO at gate 1 scales.",
    )
