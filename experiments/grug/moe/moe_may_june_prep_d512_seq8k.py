# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""June MoE prep run d=512 at seq_len=8192 (half batch keeps TPB=1M).

Mirror of ``moe_may_june_prep_d512`` with seq_len doubled and batch_size
halved. sliding_window auto-scales to seq_len/2 via the heuristic.

Submit (us-central2, v4-32, production)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_may_june_prep_d512_seq8k
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config, _phase_1_start_step
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 512
_BS: int = 128  # TPB = 128 * 8192 = 1,048,576 (~1M tokens/batch, same as seq4k)
_SEQ: int = 8192
_STEPS: int = 90122
_EP: int = 2

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)  # sliding_window auto = _SEQ // 2 = 4096
_model = dataclasses.replace(_model_base, disable_pko=True)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_phase_step = _phase_1_start_step(_STEPS, _BS)

_run_id = f"june_prep_moe_may_d{_DIM}_ep{_EP}_bs{_BS}_seq{_SEQ}"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=_datakit_data_config(
            total_steps=_STEPS,
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
            tags=["moe", "june_prep", f"d{_DIM}", f"ep{_EP}", "datakit_mix", "disable_pko", "seq8k"],
            group="june-prep-runs",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        checkpoint_keep=[{"every": _phase_step, "until": _phase_step}],
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
            f"June MoE prep d={_DIM} seq={_SEQ}. "
            f"steps={_STEPS}, tokens={_tokens:.2e}, bs={_BS} (TPB=1M), EP={_EP}, "
            f"disable_pko, ckpt at phase-1 step {_phase_step}. v4-32 us-central2."
        ),
    )
