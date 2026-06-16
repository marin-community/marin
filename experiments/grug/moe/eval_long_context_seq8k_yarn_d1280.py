# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL sweep at seq=8k with YaRN over every checkpoint of d=1280 EP=1 baseline.

YaRN extension from yarn_old_seq_len=4096 to seq_len=8192 on the long
(full-attention) layers. Short layers keep ``sliding_window=2048``. The
m-scale is computed for the 4k -> 8k extension (~1.069x). No 2-stage
composition -- this is a single-stage zero-shot YaRN extension of the
4k-trained baseline.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq8k_yarn_d1280
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.eval_long_context_seq32k_yarn import (
    LongContextYarnSweepEvalConfig,
    run_long_context_seq32k_yarn_eval,
)
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION

_DIM: int = 1280
_CKPT_DIR: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep1-b9a7ad/checkpoints"
_CHECKPOINT_STEPS: tuple[int, ...] = (
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    14325,
)
_SEQ: int = 8192
_BS: int = 16
_EP: int = 1
_MAX_DOCS: int = 32
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq8k_yarn_d{_DIM}_ep1_paloma_sweep"
eval_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_long_context_seq32k_yarn_eval,
    config=LongContextYarnSweepEvalConfig(
        dim=versioned(_DIM),
        run_id=_run_id,
        output_path=this_output_path(),
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        checkpoint_steps=versioned(_CHECKPOINT_STEPS),
        checkpoint_dir=versioned(_CKPT_DIR),
        seq_len=versioned(_SEQ),
        batch_size=versioned(_BS),
        expert_parallel=versioned(_EP),
        tokenizer_id=versioned(_TOKENIZER_ID),
        max_docs_per_dataset=versioned(_MAX_DOCS),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        yarn_old_seq_len=versioned(4096),
        yarn_alpha=versioned(1),
        yarn_beta=versioned(32),
        yarn_mscale_coef=versioned(0.1),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL sweep at seq={_SEQ} with YaRN over {len(_CHECKPOINT_STEPS)} "
            f"checkpoints of d={_DIM} EP=1 baseline. v4-32 us-central2."
        ),
    )
