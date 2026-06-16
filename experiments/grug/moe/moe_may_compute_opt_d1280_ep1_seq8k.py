# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=1280 EP=1 compute-optimal cell at 2x sequence length.

Same recipe as the d=1280 cell in ``launch.py``'s ``compute_opt_steps``,
modified to run at seq=8192 with bs=128 (half the original 256 to preserve
``tokens_per_batch = 256*4096 = 128*8192 = 1,048,576`` -- MuonH yields
identical peak LR / beta2 / epsilon to the original schedule).

Architecture is identical to the d=1280 EP=1 baseline (``sliding_window=2048``
explicitly preserved instead of the heuristic's ``seq // 2 = 4096`` at seq=8k).
Same ``num_train_steps = 14,325`` so total tokens trained match
(14,325 * 128 * 8192 = 1.50e10 = same as the original).

Permanent checkpoints every 1k steps.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d1280_ep1_seq8k
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 1280
_BS: int = 128  # half of the d=1280 EP=1 baseline (256), preserves tokens_per_batch at seq=8k
_SEQ: int = 8192
_STEPS: int = 14_325  # same as the d=1280 EP=1 baseline; total tokens = 14,325 * 128 * 8192 = 1.50e10
_SLIDING_WINDOW: int = 2048  # explicit override of heuristic default (seq // 2 = 4096 at seq=8k)

_heuristic = MoeMuonHHeuristic()

# build_model_config sets sliding_window = seq_len // 2; override to keep the
# baseline d=1280 EP=1 architecture (sw=2048) unchanged.
_model = dataclasses.replace(
    _heuristic.build_model_config(_DIM, seq_len=_SEQ),
    sliding_window=_SLIDING_WINDOW,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep1_seq8k"
compute_opt_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", "seq8k"],
            group="moe-may-compute-opt",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=128,
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
        steps=[compute_opt_step],
        description=(
            f"d={_DIM} EP=1 compute-optimal at seq={_SEQ} bs={_BS} (tpb preserved). "
            f"sw={_SLIDING_WINDOW}, steps={_STEPS}, keep every 1k. v4-32 us-central2."
        ),
    )
