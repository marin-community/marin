# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MPI-router experiment launcher.

Identical to ``launch.py``'s May-Recipe compute-optimal cells, except the MoE router is
replaced with the Manifold Power Iteration router (arXiv 2606.12397) via ``use_mpi_router``.
Everything else (optimizer, data, trainer, eval, resources) matches the baseline so the
agent.md effective-speedup comparison is apples-to-apples.

Gate 1 = d512 + d768. Submit with:

    .venv/bin/iris --cluster=marin job run --no-wait --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.launch_mpi
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.launch import (
    _SEQ_LEN,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    _heuristic,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_MPI_C_PRIME: float = 3.0
# (dim, batch_size, steps) — same cells as launch.py's compute-optimal baseline.
_GATE1_CELLS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10_980),
    (768, 64, 16_875),
)

mpi_steps: list[ExecutorStep] = []
for _dim, _bs, _steps in _GATE1_CELLS:
    _model = dataclasses.replace(
        _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN),
        use_mpi_router=True,
        mpi_c_prime=_MPI_C_PRIME,
    )
    _tokens = float(_steps * _bs * _SEQ_LEN)
    _optimizer = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)
    _run_id = f"moe_may_mpi_d{_dim}"
    mpi_steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-32", regions=["us-central2"])),
                steps=versioned(_steps),
                batch_size=versioned(_bs),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=["moe", "moe_may_mpi", f"d{_dim}"],
                    group="moe-may-mpi",
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
        steps=mpi_steps,
        description="MPI router (arXiv 2606.12397) gate-1 cells d512+d768, May Recipe.",
    )
