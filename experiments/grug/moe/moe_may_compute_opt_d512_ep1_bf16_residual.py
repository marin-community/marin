# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 EP=1 May Recipe compute-opt with bf16 model weights + fp32 residual optimizer.

Keller Jordan modded-nanogpt precision pattern:
- Model weights are bf16 (fast matmul, half the activation/transient cost).
- The optimizer wraps the inner MuonH transform in
  ``bf16_master_with_fp32_residual``: each step reconstructs an fp32 view of
  the param as ``bf16_param + fp32_residual``, runs momentum + Newton-Schulz
  + Frobenius hyperball in fp32, applies the fp32 step, truncates back to
  bf16, and stores the truncation error in the next residual.

End state: forward and backward are bf16 (matmul speed); momentum, NS, AdamH
mu/nu are fp32 (precision); the mantissa lost on each bf16 round-trip is
preserved in the optimizer state. Compare against:
- ``..._bf16_all``: same compute, no residual recovery -- pure bf16 everywhere.
- baseline: params=fp32, compute=bf16, output=bf16 (current default).

Same compute-opt cell as the d=512 entry of ``launch.py``'s ``_COMPUTE_OPT_CELLS``
(bs=32, steps=10_980, tokens=1.44e9, MuonH on heuristic_v2, EP=1).

Submit (us-central2, v4-32, production)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d512_ep1_bf16_residual
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 512
_BS: int = 32  # May Recipe compute-opt for d=512
_SEQ: int = 4096
_STEPS: int = 10_980  # May Recipe compute-opt for d=512 -- yields 1.44 B tokens
_EP: int = 1

_heuristic = MoeHeuristicV2()
_model = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = dataclasses.replace(
    _heuristic.build_optimizer_config(_BS, _tokens, _DIM, seq_len=_SEQ),
    bf16_master=True,
)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EP}_bf16_residual_v2"
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
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"ep{_EP}", "bf16_residual"],
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
            f"d={_DIM} EP={_EP} May Recipe compute-opt with bf16 model + fp32 residual optimizer "
            f"(Keller Jordan modded-nanogpt pattern: bf16 weights, fp32 momentum/NS/mu/nu, "
            f"fp32 mantissa stored in opt-state across bf16 round-trips). "
            f"bs={_BS}, steps={_STEPS}, tokens={_tokens:.2e}. v4-32 us-central2."
        ),
    )
