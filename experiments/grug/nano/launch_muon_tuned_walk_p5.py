# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p5: step-4 (`nocap-nobias-attngate-gn`) + seq=4096, batch=128.

Tokens-per-batch unchanged (1024x512 = 4096x128 = 524,288), so total tokens
and per-step token count are identical to step 4.
"""

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
from experiments.grug.nano.launch_muon_tuned_nocap_nobias_attngate_gn import (
    NANO_124M_MUON_TUNED_NOCAP_NOBIAS_ATTNGATE_GN_MODEL,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P5_TRAIN_STEPS = 3350
P5_BATCH_SIZE = 128
P5_SEQ_LEN = 4096
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P5_MODEL = dataclasses.replace(
    NANO_124M_MUON_TUNED_NOCAP_NOBIAS_ATTNGATE_GN_MODEL,
    max_seq_len=P5_SEQ_LEN,
)

P5_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p5")


nano_muon_tuned_p5_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p5-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P5_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P5_TRAIN_STEPS),
        batch_size=versioned(P5_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p5", "seq4k"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P5_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P5_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nano_muon_tuned_p5_trial], description="muon-tuned p5: seq=4096, batch=128, 3350 steps.")
