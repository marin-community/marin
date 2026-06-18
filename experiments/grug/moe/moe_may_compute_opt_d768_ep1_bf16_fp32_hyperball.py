# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=768 EP=1 May Recipe compute-opt: bf16 everywhere except the Frobenius hyperball.

Same compute-optimal cell as ``launch.py``'s d=768 entry (bs=32, steps=10_980,
tokens=1.44e9, MuonH on heuristic_v2, EP=1). Precision split:

- Master weights, Muon momentum, Newton-Schulz iterations, AdamH mu / nu: bf16
  (mp policy ``params=bfloat16,compute=bfloat16,output=bfloat16``).
- Frobenius hyperball math: fp32 internally, returns fp32 delta. The trainer's
  ``optax.apply_updates`` promotes the add to fp32 and truncates exactly once
  at the cast back to bf16, so the hyperball's norm invariant is preserved.

This sits between ``bf16_all`` (pure bf16, hyperball norm jumps around) and
``bf16_residual_v2`` (fp32 everywhere in opt-state + fp32 residual). It tests
whether the hyperball is the only bf16-sensitive piece: if this closes the gap
to baseline, NS / mu / nu don't need fp32; if it doesn't, we know we need more.

Submit (us-central2, v4-32, production)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d768_ep1_bf16_fp32_hyperball
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 768
_BS: int = 64  # May Recipe compute-opt for d=768
_SEQ: int = 4096
_STEPS: int = 16_875  # May Recipe compute-opt for d=768 -- yields 4.42 B tokens
_EP: int = 1

_heuristic = MoeHeuristicV2()
_model = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = dataclasses.replace(
    _heuristic.build_optimizer_config(_BS, _tokens, _DIM, seq_len=_SEQ),
    hyperball_fp32=True,
)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EP}_bf16_fp32_hyperball"
compute_opt_step = ExecutorStep(
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
        mp=versioned("params=bfloat16,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"ep{_EP}", "bf16_fp32_hyperball"],
            group="moe-may-compute-opt",
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
                eval_batch_size=128,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[compute_opt_step],
        description=(
            f"d={_DIM} EP={_EP} May Recipe compute-opt with bf16 master + bf16 opt-state "
            f"BUT Frobenius hyperball in fp32 (returns fp32 delta). "
            f"bs={_BS}, steps={_STEPS}, tokens={_tokens:.2e}. v4-32 us-central2."
        ),
    )
