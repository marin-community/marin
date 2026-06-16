# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 EP=2 5000-T/N, 4x compute-opt batch, heuristic_v1 + AdamH, **warmup=1%**.

Sibling of ``moe_may_5000tn_4x_d512_ep2_v1_adamh`` -- same optimizer recipe and
heuristic, but with ``warmup=0.01`` (1pct, matching the May Recipe / MuonH default)
instead of the heuristic_v1 default of ``warmup=0.1`` (10pct).

This isolates the "warmup fraction" axis from the "optimizer choice" axis when
comparing against the v2 + MuonH d=512 4x base (which also uses 1% warmup).

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_5000tn_4x_d512_ep2_v1_adamh_warmup1pct
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import MoeAdamHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 512
_BS: int = 128
_SEQ: int = 4096
_STEPS: int = 180_054
_MIN_LR_RATIO: float = 0.1
_WARMUP: float = 0.01  # 1pct, matches the May Recipe / MuonH path
_EP: int = 2

_heuristic = MoeAdamHHeuristic(min_lr_ratio=_MIN_LR_RATIO, warmup=_WARMUP)
_model = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_optimizer_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_may_5000tn_4x_d{_DIM}_ep{_EP}_v1_adamh_warmup1pct"
overtrained_step = ExecutorStep(
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
            tags=["moe", "moe_may_5000tn", f"d{_DIM}", "4x_batch", "min_lr_0.1", f"ep{_EP}", "v1_adamh", "warmup_1pct"],
            group="moe-may-5000tn-4x",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=10,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=128,
                steps_per_eval=2000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
        expert_parallel=versioned(_EP),
        keep_permanent_checkpoints=versioned(False),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[overtrained_step],
        description=(
            f"d={_DIM} EP={_EP} 5000-T/N overtrained, 4x compute-opt batch ({_BS}), "
            f"min_lr_ratio={_MIN_LR_RATIO}, heuristic_v1 + AdamH, warmup={_WARMUP} (1pct), "
            f"steps={_STEPS}, tokens={_tokens:.2e}. v4-32 us-central2."
        ),
    )
