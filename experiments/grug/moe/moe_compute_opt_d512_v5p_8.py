# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 compute-optimal A/B on v5p-8 us-east5-a.

Two ExecutorSteps share the same model / data / hardware, and differ only
in whether the stacked ``rms_attn.weight`` / ``rms_mlp.weight`` leaves
route to muonh (baseline) or to plain adam (``rmsnorm_to_adam=True``).
Used to measure the impact of the muonh routing on stacked RMSNorm scales.

Config:
- d=512 May Recipe compute-optimal cell: BS=32, seq=4096, 10,980 steps.
- ``NEMOTRON_MIX_WITH_DEFAULT_VALIDATION`` data.
- MuonH with ``min_lr_ratio=0`` (final LR decays all the way to 0).
- Stacked blocks + ``disable_pko`` + ``disable_long_rope`` for parity with
  the production 67B layout.
- v5p-8 in us-east5-a, preemptible allowed.

Submit (us-east5-a, production, preemptible allowed)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a \\
        --priority production -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_compute_opt_d512_v5p_8
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
_BS: int = 32  # May Recipe d=512 compute-opt cell
_SEQ: int = 4096
_STEPS: int = 10_980
_EP: int = 1
_SLICE: str = "v5p-8"
_ZONE: str = "us-east5-a"

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.0)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer_base = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

steps: list[ExecutorStep] = []
for _variant, _opt in (
    ("baseline", _optimizer_base),
    ("rmsadam", dataclasses.replace(_optimizer_base, rmsnorm_to_adam=True)),
):
    _run_id = f"moe_compute_opt_d{_DIM}_stacked_{_variant}_v5p_8"
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
                steps=versioned(_STEPS),
                batch_size=versioned(_BS),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=[
                        "moe",
                        "moe_compute_opt",
                        f"d{_DIM}",
                        f"ep{_EP}",
                        "nemotron_mix",
                        "disable_pko",
                        "no_long_rope",
                        "stacked",
                        "v5p_8",
                        _variant,
                    ],
                    group="moe-compute-opt-d512-stacked-rmsab",
                    name=None,
                ),
                optimizer=versioned(_opt),
                expert_parallel=_EP,
                grug_trainer=versioned(
                    GrugTrainerConfig(
                        z_loss_weight=0.0,
                        ema_beta=None,
                        log_every=1,
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
            f"d={_DIM} compute-opt A/B: baseline (stacked rms→muonh) vs rmsadam "
            f"(stacked rms→adam) on {_SLICE} {_ZONE}. BS={_BS} seq={_SEQ} steps={_STEPS}, "
            f"NEMOTRON mix, MuonH min_lr_ratio=0."
        ),
    )
