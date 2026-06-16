# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL sweep at seq=4k over every checkpoint of the d=512 EP=1 baseline.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq4k_d512
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.eval_long_context_seq4k import (
    LongContextSweepEvalConfig,
    run_long_context_seq4k_eval,
)

_DIM: int = 512
_CKPT_DIR: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints"
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
    10980,
)
_SEQ: int = 4096
_BS: int = 16
_EP: int = 1
_MAX_DOCS: int = 32
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq4k_d{_DIM}_ep1_sweep"
eval_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_long_context_seq4k_eval,
    config=LongContextSweepEvalConfig(
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
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL sweep at seq={_SEQ} over {len(_CHECKPOINT_STEPS)} checkpoints "
            f"of d={_DIM} EP=1 baseline. v4-32 us-central2."
        ),
    )
