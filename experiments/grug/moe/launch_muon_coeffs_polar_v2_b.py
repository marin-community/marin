# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 evaluation of Muon NS coefficient set ``polar_v2_b``.

Two cells (d=512 and d=768, the May Recipe gate-1 scales per agent.md), May Recipe
compute-optimal defaults (MuonH on heuristic_v2, EP=1, 1% warmup, linear decay to 0,
fp32 params + bf16 compute), only the Newton-Schulz coefficient set on the muonh
group is changed from the ``quintic`` default to ``polar_v2_b``:

  (8.1566, -22.4833, 15.8788)
  (4.0429,  -2.8089,  0.5000)
  (3.8917,  -2.7725,  0.5061)
  (3.2858,  -2.3681,  0.4645)
  (2.3005,  -1.6112,  0.3833)

Coefficients are registered in ``levanter/optim/util.py:NEWTON_SCHULZ_COEFFICIENTS``.

Submit (us-central2, v5p-8, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --reserve v5p-8 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.launch_muon_coeffs_polar_v2_b
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_COEFFICIENT_TYPE: str = "polar_v2_b"
_SEQ_LEN: int = 4096
# (dim, bs, steps) -- May Recipe compute-optimal cells for d=512 and d=768.
_GATE1_CELLS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10_980),  # 3.82e17 FLOPs, 1.44 B tokens
    (768, 64, 16_875),  # 2.81e18 FLOPs, 4.42 B tokens
)

_heuristic = MoeMuonHHeuristic()

gate1_steps: list[ExecutorStep] = []
for _dim, _bs, _steps in _GATE1_CELLS:
    _model = _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN)
    _tokens = float(_steps * _bs * _SEQ_LEN)
    _optimizer = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)
    # Override the coefficient_type field on the MuonH optimizer config.
    _optimizer = dataclasses.replace(_optimizer, coefficient_type=_COEFFICIENT_TYPE)
    _run_id = f"moe_may_muon_coeffs_polar_v2_b_d{_dim}"
    gate1_steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                steps=versioned(_steps),
                batch_size=versioned(_bs),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=["moe", "moe_may_compute_opt", f"d{_dim}", "muon_coeffs", _COEFFICIENT_TYPE],
                    group="moe-may-muon-coeffs-gate1",
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
                        eval_batch_size=512,
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
        steps=gate1_steps,
        description=(f"Gate-1 (d=512, d=768) baselines, Muon NS coefficient_type='{_COEFFICIENT_TYPE}'. v5p-8 EP=1."),
    )
