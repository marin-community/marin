# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 EP=1 May Recipe compute-opt: bf16_residual_v2 + RMSNorm weights stay fp32.

Builds on ``bf16_residual_v2``: bf16 model weights, fp32 master reconstructed
as ``bf16_param + fp32_residual`` inside the optimizer wrapper, fp32 delta
returned to apply_updates so the truncation happens exactly once.

Additional change: every RMSNorm.weight (post-embed, per-block ``rms_attn`` /
``rms_mlp``, final pre-head) is recast back to fp32 after mp.cast_to_param.
The wrapper is now dtype-aware: fp32 leaves bypass the bf16 truncation step,
so their delta application is fp32-exact and they accumulate no residual.

Motivation: RMSNorm weights init to 1.0 and the per-step Adam delta is small
(~6e-4). At scale 1.0 the bf16 ULP is 1/128 ≈ 0.0078, so many updates round
to 0 and the bf16 view stays pinned at 1.0 even though the fp32 residual is
moving. ``params/norm/...`` stays flat, the model has fewer training-time
degrees of freedom. Keeping these specific weights in fp32 (~tiny memory
cost -- one (hidden_dim,) vector per RMSNorm) avoids the ULP floor.

Same compute-opt cell as the d=512 entry of ``launch.py``'s ``_COMPUTE_OPT_CELLS``
(bs=32, steps=10_980, tokens=1.44e9, MuonH on heuristic_v2, EP=1).

Submit (us-central2, v4-32, production)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d512_ep1_bf16_residual_fp32_rmsnorm
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

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EP}_bf16_residual_fp32_rmsnorm"
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
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"ep{_EP}", "bf16_residual", "fp32_rmsnorm"],
            group="moe-may-compute-opt",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
                rms_norm_fp32=True,
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
            f"d={_DIM} EP={_EP} May Recipe compute-opt with bf16 master + fp32 residual + RMSNorm fp32. "
            f"bs={_BS}, steps={_STEPS}, tokens={_tokens:.2e}. v4-32 us-central2."
        ),
    )
