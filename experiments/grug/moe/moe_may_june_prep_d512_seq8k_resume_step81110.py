# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume ``june_prep_moe_may_d512_ep2_bs128_seq8192_sw2k`` at step-71808, run to step-81110.

The source run targets 90,122 total steps. Its ~80% checkpoint sits at step-71808
(closest to 90,122 * 0.8 = 72,098). This launcher resumes from that checkpoint
and trains the additional 9,302 steps needed to reach step-81110 (= 90% of the
original 90,122-step schedule). A permanent checkpoint is saved at step-81110.

Model / data / batch / seq_len / sliding_window / EP are all unchanged from
the source — this is a vanilla continuation, no context extension.

LR schedule notes (matches source exactly):
- Peak LR / beta2 / epsilon are matched to the source by feeding the heuristic
  the *original* total-token count (90,122 * 128 * 8192).
- Data permutation is matched by passing ``total_steps=90_122`` to the data
  config, so the iterator reads the same tokens at steps 71808-81110 that the
  source would have read at those steps.
- The lr_scheduler's warmup + decay are *decoupled* from the trainer's stop
  step via ``GrugMoeMuonHConfig.schedule_num_train_steps_override = 90_122``.
  The scheduler thinks the schedule is 90,122 steps long (matching source);
  the trainer stops at 81,110. So at every step from 71808-81110 the LR is
  identical to what the source would have produced at the same step. This
  makes the step-81110 checkpoint a faithful 90% snapshot of the source's
  trajectory — suitable as a base for downstream forks.

Submit (us-central2, v4-32, production)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_may_june_prep_d512_seq8k_resume_step81110
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_SOURCE_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/june_prep_moe_may_d512_ep2_bs128_seq8192_sw2k-bc7059/checkpoints/step-71808"
)
_DIM: int = 512
_BS: int = 128
_SEQ: int = 8192
_SLIDING_WINDOW: int = 2048
_ORIG_STEPS: int = 90_122  # source's total schedule length (preserved for LR + data)
_STEPS: int = 81_110  # 90% of _ORIG_STEPS; trainer stops here
_EP: int = 2

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(_model_base, disable_pko=True, sliding_window=_SLIDING_WINDOW)
# Peak LR is matched to the source by using the original token count (90,122 * BS * SEQ),
# not the truncated 81,110-step count.
_orig_tokens = float(_ORIG_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _orig_tokens, _DIM, seq_len=_SEQ)
# Decouple the LR scheduler's num_train_steps from the trainer's stop step,
# so the LR at every resumed step exactly matches the original 90,122-step
# schedule's value at the same step (rather than getting compressed into the
# shorter 81,110-step run).
_optimizer = dataclasses.replace(_optimizer, schedule_num_train_steps_override=_ORIG_STEPS)

_run_id = f"june_prep_moe_may_d{_DIM}_ep{_EP}_seq8k_sw2k_resume_to_step{_STEPS}"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        # Pass the *original* total_steps so the data permutation matches what the
        # source iterator would have produced at steps 71808-81110.
        data=_datakit_data_config(
            total_steps=_ORIG_STEPS,
            batch_size=_BS,
            max_seq_len=_SEQ,
            enable_simulated_epoching=True,
        ),
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "june_prep", f"d{_DIM}", f"ep{_EP}", "seq8k", "resume", "step81110"],
            group="june-prep-runs",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        # Mark the step-81110 checkpoint permanent so it survives the
        # default retention policy.
        checkpoint_keep=[{"every": _STEPS, "until": _STEPS}],
        load_checkpoint_path=versioned(_SOURCE_CKPT_PATH),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                # Match source's 4.2M eval tokens: 128 * 8192 * 4 = 4_194_304.
                eval_batch_size=128,
                steps_per_eval=1000,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[step],
        description=(
            f"Resume d={_DIM} EP={_EP} bs128 seq8k_sw2k from step-71808 (~80%) and train to "
            f"step-{_STEPS} (~90% of {_ORIG_STEPS}). Saves a permanent checkpoint at step-{_STEPS}. "
            "Peak LR matched to source via original token count; decay span = new num_train_steps. "
            "v4-32 us-central2."
        ),
    )
