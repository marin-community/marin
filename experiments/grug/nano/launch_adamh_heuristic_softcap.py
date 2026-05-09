# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH variant with the tanh soft-cap (`cap * tanh(x / cap)`)
instead of the modded-nanogpt rsqrt form. Otherwise identical to
`launch_adamh_heuristic.py:nano_adamh_heuristic_trial`.

Driven by `cap_form="tanh"` on the model config. Useful as a controlled A/B
against `nano-adamh-heuristic-rsqrt_cap` to measure the impact of the soft-cap
saturation profile on the heuristic's training trajectory.
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    ADAMH_HEURISTIC_TRAIN_STEPS,
    NANO_124M_HEURISTIC_MODEL,
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_124M_HEURISTIC_SOFTCAP_MODEL = dataclasses.replace(
    NANO_124M_HEURISTIC_MODEL,
    cap_form="tanh",
)


NANO_124M_HEURISTIC_SOFTCAP_OPTIMIZER = build_heuristic_optimizer(
    batch_size=512,
    num_train_steps=ADAMH_HEURISTIC_TRAIN_STEPS,
    seq_len=NANO_124M_HEURISTIC_SOFTCAP_MODEL.max_seq_len,
    hidden_dim=NANO_124M_HEURISTIC_SOFTCAP_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-softcap")


nano_adamh_heuristic_softcap_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-softcap-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_HEURISTIC_SOFTCAP_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(ADAMH_HEURISTIC_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "softcap"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_HEURISTIC_SOFTCAP_OPTIMIZER),
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
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_heuristic_softcap_trial],
        description="Nano heuristic AdamH, tanh soft-cap, 4875 steps on fineweb10B-gpt2.",
    )
