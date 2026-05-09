# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p7: p6 + SwiGLU MLP at intermediate_dim = 3 * hidden_dim."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p6 import P6_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P7_TRAIN_STEPS = 3350
P7_BATCH_SIZE = 128
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

# SwiGLU has 3 matrices (gate, up, down) so we shrink the intermediate dim from
# 4*hidden_dim (ReLU^2 default) to 3*hidden_dim. Per-block FLOPs: 16D^2 -> 18D^2,
# i.e. SwiGLU at 3*D is ~12% more compute than ReLU^2 at 4*D. _init_muon_tuned was
# extended to construct the gate matrix when mlp_type="swiglu".
P7_MODEL = dataclasses.replace(
    P6_MODEL,
    mlp_type="swiglu",
    intermediate_dim=3 * P6_MODEL.hidden_dim,  # 3 * 768 = 2304
)

P7_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p7")


nano_muon_tuned_p7_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p7-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P7_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P7_TRAIN_STEPS),
        batch_size=versioned(P7_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p7", "swiglu"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P7_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P7_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nano_muon_tuned_p7_trial], description="muon-tuned p7: + SwiGLU at 3*D, 3350 steps.")
