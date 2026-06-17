# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""seq=8k SWA-only full-eval sweep for d=1280 EP=1 baseline with paloma eval_batch_size=64.

Companion to ``eval_long_context_seq8k_yarn_d1280_paloma_bs64`` for the SWA-only
extension path. paloma TaggedEvaluator at ``eval_batch_size=64`` -- 8 x 64 x 8192 = 4.19M
tokens/slice, matching the seq=4k baseline's per-slice training-time token count.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq8k_swa_d1280_paloma_bs64
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.evals.long_context_ppl import long_context_supervised_validation_sets
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
_PALOMA_BS: int = 64
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq8k_swa_d{_DIM}_ep1_paloma_bs{_PALOMA_BS}_sweep"
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
        supervised_datasets=long_context_supervised_validation_sets(),
        paloma_eval_batch_size=versioned(_PALOMA_BS),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context full-eval sweep at seq={_SEQ}, long_sliding_window={_LONG_SWA}, no YaRN, "
            f"paloma_bs={_PALOMA_BS}. {len(_CHECKPOINT_STEPS)} checkpoints. v4-32 us-central2."
        ),
    )
