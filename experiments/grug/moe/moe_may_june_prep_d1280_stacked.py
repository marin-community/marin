# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""June MoE prep run d=1280 with use_array_stacked_blocks=True + eval enabled.

Companion to ``moe_may_june_prep_d1280`` (which runs eval-off). This version
re-enables eval and switches to the haliax ``ArrayStacked`` / ``lax.scan``
block layout. The scan reuses gradient all-reduce buffers across iterations,
which previously dropped program-HBM from 32.36G to 10.04G at d=1280.

Risk: the prior d=1280 scan attempt OOM'd on optimizer state (23.21G of
arguments — opt-state tensors that came out only ~4-way sharded instead
of the expected ~256-way). This run intentionally re-tries the same path
to (a) confirm reproduction and (b) capture a fresh HBM dump for the
opt-state debug.

Submit (us-central2, v4-512, production)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_may_june_prep_d1280_stacked
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
_BS: int = 256  # TPB = 256 * 4096 = 1,048,576 (~1M tokens/batch)
_SEQ: int = 4096
_STEPS: int = 1178741
_EP: int = 2

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
# disable_pko=True is required by use_array_stacked_blocks (CausalSelfAttention
# reads use_pko at trace time -- the scan body needs a Python-constant value).
_model = dataclasses.replace(_model_base, disable_pko=True, use_array_stacked_blocks=True)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_phase_step = _phase_1_start_step(_STEPS, _BS)

_run_id = f"june_prep_moe_may_d{_DIM}_ep{_EP}_bs{_BS}_stacked_eval"
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
            tags=["moe", "june_prep", f"d{_DIM}", f"ep{_EP}", "datakit_mix", "disable_pko", "stacked"],
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
            f"June MoE prep d={_DIM} stacked+eval: 5000-TPP overtraining on datakit_moe_mix. "
            f"steps={_STEPS}, tokens={_tokens:.2e}, bs={_BS} (TPB=1M), EP={_EP}, "
            f"disable_pko, use_array_stacked_blocks=True, ckpt at phase-1 step {_phase_step}. "
            f"v4-512 us-central2."
        ),
    )
