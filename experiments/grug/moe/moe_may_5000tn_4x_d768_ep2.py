# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=768 EP=2 at 5000 tokens/active-param overtrained, 4x compute-opt batch, min_lr_ratio=0.1.

EP=2 sibling of ``moe_may_5000tn_4x_d768_ep1``.  ``expert_parallel=2`` puts the
v4-32 mesh at (data=8, expert=2); bs=256 stays divisible by data=8.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_5000tn_4x_d768_ep2
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 768
_BS: int = 256  # 4x of the May Recipe compute-opt baseline (64) at d=768
_SEQ: int = 4096
_STEPS: int = 262_594  # 55.1M active * 5000 / (256 * 4096) -- yields 275.4B tokens
_MIN_LR_RATIO: float = 0.1
_EP: int = 2

_heuristic = MoeMuonHHeuristic(min_lr_ratio=_MIN_LR_RATIO)
_model = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_may_5000tn_4x_d{_DIM}_ep{_EP}"
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
            tags=["moe", "moe_may_5000tn", f"d{_DIM}", "4x_batch", "min_lr_0.1", f"ep{_EP}"],
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
            f"min_lr_ratio={_MIN_LR_RATIO}, steps={_STEPS}, tokens={_tokens:.2e}. v4-32 us-central2."
        ),
    )
