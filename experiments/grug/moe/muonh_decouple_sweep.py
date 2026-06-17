# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH magnitude-direction decoupling sweep launcher.

Tests Häggström's per-row/per-column learnable gains on top of the May-Recipe MuonH baseline:
each muonh matrix is factorized as W = diag(gamma_row) W_hat diag(gamma_col) with W_hat on a
fixed-norm sphere (the existing Frobenius hyperball) and gains stepped by Adam at their own LR.

One ExecutorStep per invocation; all knobs come from env vars. Baseline LR / warmup / schedule
come from the MuonH heuristic so the comparison vs the d512 MuonH anchor (paloma 3.5438) is
apples-to-apples. Example (decoupled, both gains):

    iris --cluster=marin job run --no-wait --extra tpu --region us-east5 \
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-east5 \
      -e MUONH_TAG decouple-both -e MUONH_DECOUPLE 1 -e MUONH_GAIN_MODE both \
      -e MUONH_TPU v5p-8 -e MUONH_REGION us-east5 -e MUONH_PREEMPTIBLE 1 \
      -- python -m experiments.grug.moe.muonh_decouple_sweep
"""

import dataclasses
import os

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

_CELLS: dict[int, tuple[int, int, int]] = {  # dim -> (dim, batch_size, steps)
    512: (512, 32, 10_980),
    768: (768, 64, 16_875),
}


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


_DIM = int(os.environ.get("MUONH_DIM", "512"))
_TAG = os.environ.get("MUONH_TAG", "decouple-both")
_LR_MULT = _f("MUONH_LR_MULT", 1.0)
_TPU = os.environ.get("MUONH_TPU", "v5p-8")
_REGION = os.environ.get("MUONH_REGION", "us-east5")
_PREEMPTIBLE = _flag("MUONH_PREEMPTIBLE", "1")
_dim, _bs, _steps = _CELLS[_DIM]
_tokens = float(_steps * _bs * _SEQ_LEN)

_model = _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN)
_muonh = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)

# Decoupling knobs layered onto the heuristic MuonH config (everything else stays at the anchor).
_optimizer = dataclasses.replace(
    _muonh,
    learning_rate=_muonh.learning_rate * _LR_MULT,
    decouple_gains=_flag("MUONH_DECOUPLE", "1"),
    gain_lr=_f("MUONH_GAIN_LR", 1e-3),
    gain_mode=os.environ.get("MUONH_GAIN_MODE", "both"),
    decouple_lm_head=_flag("MUONH_DECOUPLE_LMHEAD", "0"),
)

_run_id = f"muonh_d{_dim}_{_TAG}"
muonh_steps: list[ExecutorStep] = [
    ExecutorStep(
        name=f"grug/{_run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(_model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU, regions=[_REGION], preemptible=_PREEMPTIBLE)),
            steps=versioned(_steps),
            batch_size=versioned(_bs),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "muonh", "decouple", f"d{_dim}", _TAG],
                group="muonh-decouple-d512",
                name=None,
            ),
            optimizer=versioned(_optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512, steps_per_eval=1000, max_eval_batches=8, eval_current=True, eval_ema=False
                )
            ),
            expert_parallel=2,
        ),
    )
]


if __name__ == "__main__":
    executor_main(steps=muonh_steps, description=f"MuonH decouple d{_DIM} {_TAG}")
