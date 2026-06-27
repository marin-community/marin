# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 BS=1024 baselines on v5p-8 us-east5-a.

Two ExecutorSteps share the same model / data / hardware and differ only
in step count (200 vs 1000). Re-creates the high-BS d=512 baselines that
the earlier ``launch_datakit_moe_mix.py`` submissions on the cherry-picked
``moe_may_pr_d512_datakit_test`` branch failed to produce, this time on
``june_tpu_67b_a2b`` where the stacked-block port is healthy.

Submit (us-east5-a, production)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a \\
        --priority production -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_d512_baseline_v5p_8
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 512
_BS: int = 1024  # 1024 * 8192 = 8,388,608 tokens/step
_SEQ: int = 8192
_EP: int = 1
_REPLICA_AXIS: int = 1
_SLICE: str = "v5p-8"
_ZONE: str = "us-east5-a"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)

steps: list[ExecutorStep] = []
for _variant, _rmsnorm_to_adam in (("baseline", False), ("rmsadam", True)):
    for _step_count in (200, 1000):
        _tokens = float(_step_count * _BS * _SEQ)
        _optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
        _optimizer = dataclasses.replace(_optimizer, rmsnorm_to_adam=_rmsnorm_to_adam)
        _run_id = f"moe_d{_DIM}_{_variant}_bs{_BS}_seq{_SEQ}_v5p_8_{_step_count}steps"
        steps.append(
            ExecutorStep(
                name=f"grug/{_run_id}",
                fn=run_grug_moe_trial,
                config=GrugMoeLaunchConfig(
                    model=versioned(_model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=_run_id,
                    resources=versioned(ResourceConfig.with_tpu(_SLICE, zone=_ZONE)),
                    steps=versioned(_step_count),
                    batch_size=versioned(_BS),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="marin_moe",
                        tags=[
                            "moe",
                            "d512_baseline",
                            f"d{_DIM}",
                            f"ep{_EP}",
                            f"bs{_BS}",
                            "nemotron_mix",
                            "disable_pko",
                            "no_long_rope",
                            "stacked",
                            "v5p_8",
                            _variant,
                            f"{_step_count}steps",
                        ],
                        group="moe-d512-baseline-v5p-8",
                        name=None,
                    ),
                    optimizer=versioned(_optimizer),
                    expert_parallel=_EP,
                    grug_trainer=versioned(
                        GrugTrainerConfig(
                            z_loss_weight=_LOGIT_Z_LOSS_WEIGHT,
                            ema_beta=None,
                            log_every=1,
                            replica_axis_size=_REPLICA_AXIS,
                        )
                    ),
                    eval=versioned(
                        GrugEvalConfig(
                            # v5p-8 = 4 devices, replica=1, EP=1 → data=4.
                            # eval_batch_size must be divisible by 4.
                            eval_batch_size=256,
                            steps_per_eval=1000,
                            max_eval_batches=8,
                            eval_current=True,
                            eval_ema=False,
                        )
                    ),
                ),
            )
        )


if __name__ == "__main__":
    executor_main(
        steps=steps,
        description=(
            f"d={_DIM} BS={_BS} seq={_SEQ} baselines on {_SLICE} {_ZONE} "
            f"(200-step and 1000-step variants), datakit mix, pure FSDP."
        ),
    )
