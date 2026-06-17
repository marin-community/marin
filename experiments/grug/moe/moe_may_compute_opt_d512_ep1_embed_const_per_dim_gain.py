# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 EP=1 May Recipe compute-opt with const rescale + learnable per-dim embed gain.

Both embed_norm (RMSNorm) and embed_gated_norm (GatedNorm) are dropped.
Replaced by ``hidden * (1/initializer_std) * embed_gain`` where
``embed_gain`` is a learnable ``(hidden_dim,)`` vector initialized to ones.
Same dtype safety as the scalar sibling (Python-float weak-type promotion).

Same compute-optimal cell as the d=512 entry of ``launch.py``'s
``_COMPUTE_OPT_CELLS`` (bs=32, steps=10_980, tokens=1.44e9, MuonH on
heuristic_v2, EP=1).

Submit (us-east5-a, v5p-8, interactive priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a \\
        --priority interactive \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d512_ep1_embed_const_per_dim_gain
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
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(_model_base, embed_learnable_gain="per_dim")
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_optimizer_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EP}_embed_const_per_dim_gain"
compute_opt_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"ep{_EP}", "embed_const_per_dim_gain"],
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
            f"d={_DIM} EP={_EP} May Recipe compute-opt with embed_learnable_gain='per_dim' "
            f"(drop RMSNorm + GatedNorm, const rescale + learnable (hidden,) gain init to ones). "
            f"bs={_BS}, steps={_STEPS}, tokens={_tokens:.2e}. v5p-8 us-east5-a."
        ),
    )
