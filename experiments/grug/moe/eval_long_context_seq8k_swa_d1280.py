# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL sweep at seq=8k with long_sliding_window=4k, no YaRN, for d=1280 EP=1 baseline.

Eval at seq=8192 with the long (full-attention) layers capped at
``long_sliding_window=4096``. Short layers keep ``sliding_window=2048``.
No YaRN -- base RoPE inv_freq. Companion to ``eval_long_context_seq8k_yarn_d1280``
for the YaRN-vs-SWA-only comparison at the 4k -> 8k extension factor.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq8k_swa_d1280
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.eval_long_context_seq32k_swa import (
    LongContextSwaSweepEvalConfig,
    run_long_context_seq32k_swa_eval,
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
_LONG_SWA: int = 4096
_MAX_DOCS: int = 32
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq8k_swa_d{_DIM}_ep1_paloma_sweep"
eval_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_long_context_seq32k_swa_eval,
    config=LongContextSwaSweepEvalConfig(
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
        long_sliding_window=versioned(_LONG_SWA),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL sweep at seq={_SEQ}, long_sliding_window={_LONG_SWA}, no YaRN. "
            f"d={_DIM} EP=1 baseline, {len(_CHECKPOINT_STEPS)} checkpoints. v4-32 us-central2."
        ),
    )
