# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p9: p8 + GQA at the moe heuristic ratio.

`MoeAdamHHeuristic._compute_kv_heads` returns the largest divisor of
`num_heads` that is `<= num_heads / 4`. For our `num_heads=6` that's `1` (MQA).
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
from experiments.grug.nano.launch_muon_tuned_walk_p8 import P8_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P9_TRAIN_STEPS = 3350
P9_BATCH_SIZE = 128
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025


def _moe_kv_heads(num_heads: int, gqa_ratio: int = 4) -> int:
    target = max(1, num_heads // gqa_ratio)
    for k in range(target, 0, -1):
        if num_heads % k == 0:
            return k
    return 1


P9_NUM_KV_HEADS = _moe_kv_heads(P8_MODEL.num_heads, gqa_ratio=4)

P9_MODEL = dataclasses.replace(P8_MODEL, num_kv_heads=P9_NUM_KV_HEADS)

P9_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p9")


nano_muon_tuned_p9_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p9-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P9_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P9_TRAIN_STEPS),
        batch_size=versioned(P9_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p9", "gqa"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P9_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P9_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p9_trial],
        description=f"muon-tuned p9: + GQA num_kv_heads={P9_NUM_KV_HEADS}, 3350 steps.",
    )
