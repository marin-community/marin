# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""June MoE prep run d=1280: seq=8192, no_long_rope, 2x batch, stacked blocks.

Higher-MFU variant of ``moe_may_june_prep_d1280_stacked``:

- ``seq_len=8192`` (2x), ``batch_size=512`` (2x) -- 4x tokens per step, so
  steps drop from 1,178,741 to 294,685 for the same ~1.24T-token budget.
- ``disable_long_rope=True``: long layers (every 4th + last) skip rotary
  entirely (short layers keep half-RoPE).
- ``sliding_window=2048`` (pinned -- heuristic would auto-scale to 4096
  with seq_len=8192; we want the same window as the seq=4096 baselines).
- ``use_array_stacked_blocks=True`` + ``disable_pko=True`` keep the scan
  path that fits d=1280 in v4-512 HBM.
- ``eval_batch_size=256, max_eval_batches=2`` halves eval count to keep
  eval token volume flat now that each example is 2x longer.

Submit (us-central2, v4-512, production)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_may_june_prep_d1280_no_long_rope_seq8k_stacked
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

_DIM: int = 1280
_BS: int = 1024  # TPB = 1024 * 8192 = 8,388,608 (~8M tokens/batch, 2x the bs512 sibling)
_SEQ: int = 8192
_STEPS: int = 147343  # 294,685 / 2 -- same ~1.24T-token budget as the bs512 sibling
_EP: int = 2

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_phase_step = _phase_1_start_step(_STEPS, _BS)

_run_id = f"june_prep_moe_may_d{_DIM}_ep{_EP}_bs{_BS}_no_long_rope_seq{_SEQ}_sw2k_stacked"
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
        resources=versioned(ResourceConfig.with_tpu("v4-512")),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=[
                "moe",
                "june_prep",
                f"d{_DIM}",
                f"ep{_EP}",
                "datakit_mix",
                "disable_pko",
                "no_long_rope",
                "seq8k",
                "stacked",
            ],
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
                eval_batch_size=256,
                steps_per_eval=1000,
                max_eval_batches=2,
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
            f"June MoE prep d={_DIM} seq={_SEQ} BS={_BS} no_long_rope stacked. "
            f"steps={_STEPS}, tokens={_tokens:.2e}, TPB=8M, EP={_EP}, "
            f"disable_pko, sliding_window=2048, ckpt at phase-1 step {_phase_step}."
        ),
    )
