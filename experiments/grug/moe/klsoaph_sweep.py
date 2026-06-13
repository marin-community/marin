# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KLSOAPH d512 coordinate-descent sweep launcher (issue #5728).

One ExecutorStep per invocation (one iris coordinator), so points fan out in
parallel. All HPs come from env vars so the same module serves every sweep
point. Optimizer = GrugMoeKLSoapHConfig (full-matrix SOAP + hyperball); base
LR / warmup / schedule are taken from the may-recipe MuonH heuristic so the
comparison vs MuonH (d512 paloma 3.5438) is apples-to-apples.

Runs on RESERVED v4-32 in us-central2 (baseline hardware, co-located data);
submit with MARIN_PREFIX=gs://marin-us-central2. Example:

    iris --cluster=marin job run --no-wait \
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-central2 \
      -e KLSOAPH_TAG center -e KLSOAPH_BETA2 0.95 \
      -- python -m experiments.grug.moe.klsoaph_sweep
"""

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
from experiments.grug.moe.optimizer import GrugMoeKLSoapHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_CELLS: dict[int, tuple[int, int, int]] = {  # dim -> (dim, batch_size, steps)
    512: (512, 32, 10_980),
    768: (768, 64, 16_875),
}


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _maybe_none_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    if v is None:
        return default
    return None if v.lower() in ("none", "0", "off") else float(v)


_DIM = int(os.environ.get("KLSOAPH_DIM", "512"))
_TAG = os.environ.get("KLSOAPH_TAG", "center")
_LR_MULT = _f("KLSOAPH_LR_MULT", 1.0)
# Hardware (env-parametrized so the same sweep point can run on reserved v4-32 us-central2
# OR preemptible v5p-8 us-east5 when the reservation is contended). MARIN_PREFIX must match
# the region (gs://marin-us-central2 / gs://marin-us-east5) — both have the grug data mirrored.
_TPU = os.environ.get("KLSOAPH_TPU", "v4-32")
_REGION = os.environ.get("KLSOAPH_REGION", "us-central2")
_PREEMPTIBLE = os.environ.get("KLSOAPH_PREEMPTIBLE", "0").lower() in ("1", "true", "yes")
_dim, _bs, _steps = _CELLS[_DIM]
_tokens = float(_steps * _bs * _SEQ_LEN)

_model = _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN)
# Take base LR / warmup / schedule from the MuonH heuristic (apples-to-apples vs MuonH).
_muonh = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)

_optimizer = GrugMoeKLSoapHConfig(
    learning_rate=_muonh.learning_rate * _LR_MULT,
    # SOAP eigenbasis HPs (the variable under test):
    beta1=_f("KLSOAPH_BETA1", 0.95),
    beta2=_f("KLSOAPH_BETA2", 0.9),
    shampoo_beta=_f("KLSOAPH_SHAMPOO", 0.9),
    epsilon=_f("KLSOAPH_EPS", 1e-8),
    precond_freq=int(os.environ.get("KLSOAPH_PRECOND_FREQ", "1")),
    init_factor=_f("KLSOAPH_INITF", 0.1),
    identity_init=os.environ.get("KLSOAPH_IDENTITY_INIT", "0").lower() in ("1", "true", "yes"),
    reparam_eig=os.environ.get("KLSOAPH_REPARAM_EIG", "0").lower() in ("1", "true", "yes"),
    # Non-SOAP groups pinned to the d512 MuonH baseline (apples-to-apples) — env-overridable:
    adam_lr=_f("KLSOAPH_ADAM_LR", _muonh.adam_lr),
    adam_beta1=_f("KLSOAPH_ADAM_BETA1", _muonh.beta1),
    adam_beta2=_f("KLSOAPH_ADAM_BETA2", _muonh.beta2),
    adam_epsilon=_f("KLSOAPH_ADAM_EPS", _muonh.epsilon),
    max_grad_norm=_maybe_none_float("KLSOAPH_MAXGN", _muonh.max_grad_norm),
    # Separate warmups: adam groups keep the MuonH warmup; the SOAP group can warm up longer
    # (it has an early preconditioner-estimation lag). Both default to the MuonH warmup.
    warmup=_f("KLSOAPH_ADAM_WARMUP", _muonh.warmup),
    klsoaph_warmup=_f("KLSOAPH_SOAP_WARMUP", _muonh.warmup),
    # LR-schedule axis: KLSOAPH loses its mid-run lead in the decay-to-zero phase, so allow an
    # independent nonzero floor (min_lr_ratio) to keep it improving through the end.
    min_lr_ratio=_f("KLSOAPH_MIN_LR_RATIO", _muonh.min_lr_ratio),
    lr_schedule=os.environ.get("KLSOAPH_LR_SCHEDULE", _muonh.lr_schedule),  # linear (MuonH) | cosine | inv_sqrt
    decay=_muonh.decay,
)

_run_id = f"klsoaph_d{_dim}_{_TAG}"
klsoaph_steps: list[ExecutorStep] = [
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
                tags=["moe", "klsoaph", "may_recipe", f"d{_dim}", _TAG],
                group="klsoaph-d512-maypr",
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
    executor_main(steps=klsoaph_steps, description=f"KLSOAPH d{_DIM} sweep point {_TAG}")
