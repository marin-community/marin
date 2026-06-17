# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""seq=8k YaRN full-eval sweep for d=1280 EP=1 baseline with paloma eval_batch_size=64.

Identical to ``eval_long_context_seq8k_yarn_d1280_paloma_bs128`` except the paloma
TaggedEvaluator runs at ``eval_batch_size=64`` -- 8 batches x 64 sequences x 8192 tokens
= 4.19M tokens/slice, matching the training-time token count of the seq=4k baseline
(``moe_may_compute_opt_d1280_ep1`` at bs=128, max=8, seq=4096 -> 4.19M tokens/slice).

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq8k_yarn_d1280_paloma_bs64
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.evals.long_context_ppl import long_context_supervised_validation_sets
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
_PALOMA_BS: int = 64
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq8k_yarn_d{_DIM}_ep1_paloma_bs{_PALOMA_BS}_sweep"
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
        supervised_datasets=long_context_supervised_validation_sets(),
        paloma_eval_batch_size=versioned(_PALOMA_BS),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context full-eval sweep at seq={_SEQ} with YaRN, paloma_bs={_PALOMA_BS} "
            f"(matches baseline_4k token count), {len(_CHECKPOINT_STEPS)} checkpoints. v4-32 us-central2."
        ),
    )
