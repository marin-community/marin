# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=768 EP=1 May Recipe compute-optimal cell, alternating pure-dense and pure-MoE blocks.

Same compute-optimal cell as the d=768 entry of ``launch.py``'s ``_COMPUTE_OPT_CELLS``
(bs=64, steps=16_875, tokens=4.42e9, MuonH on heuristic_v2, EP=1). Architecture:
alternating dense (3*hidden_dim FFN) and MoE (512 experts, top-6, hidden_dim/2
FFN) blocks. See d=512 sibling for hypothesis.

Submit (us-east5-a, v5p-8, interactive priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a \\
        --priority interactive \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_compute_opt_d768_ep1_alternate_dense_moe
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 768
_BS: int = 64  # May Recipe compute-opt for d=768
_SEQ: int = 4096
_STEPS: int = 16_875  # May Recipe compute-opt for d=768 -- yields 4.42 B tokens
_EP: int = 1

_heuristic = MoeMuonHHeuristic()
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    alternate_dense_moe=True,
    dense_intermediate_dim=3 * _DIM,
    num_experts=512,
    num_experts_per_token=6,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = f"moe_may_compute_opt_d{_DIM}_ep{_EP}_alternate_dense_moe"
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
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"ep{_EP}", "alternate_dense_moe"],
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
            f"d={_DIM} EP={_EP} May Recipe compute-opt with alternating dense (3*d FFN) "
            f"and MoE (512 experts, top-6, d/2 FFN) blocks. "
            f"bs={_BS}, steps={_STEPS}, tokens={_tokens:.2e}. v5p-8 us-east5-a."
        ),
    )
