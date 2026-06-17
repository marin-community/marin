# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline d=1280 EP=1 ckpts evaluated at seq=4k with paloma eval_batch_size=256.

Provides the seq=4k arm of the paloma-token-coverage sweep: same baseline checkpoints
as the seq=8k yarn/swa sweeps, evaluated at the model's trained seq_len (4096), with
paloma TaggedEvaluator at ``eval_batch_size=256`` -- 8 batches x 256 sequences x 4096
tokens = 8.39M tokens/slice. That matches the seq=8k bs=128 token count exactly, so:

  baseline_4k @ seq=4k @ bs=256  ->  8.39M tokens   (this run)
  yarn / swa @ seq=8k @ bs=128   ->  8.39M tokens   (the matched seq=8k arm)

isolates the effect of eval seq_len at fixed token coverage. Also useful vs the
training-time baseline_4k log (which scored 4.19M tokens/slice at bs=128) to confirm
that doubling token coverage barely moves paloma macro.

Uses the swa runner with ``long_sliding_window=4096`` at seq=4k -- mathematically
equivalent to the baseline architecture (long layers see all 4k positions, the
window cap never activates). No YaRN.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq4k_baseline_d1280_paloma_bs256
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
_SEQ: int = 4096
_BS: int = 16
_EP: int = 1
_LONG_SWA: int = 4096
_MAX_DOCS: int = 32
_PALOMA_BS: int = 256
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq4k_baseline_d{_DIM}_ep1_paloma_bs{_PALOMA_BS}_sweep"
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
            f"Baseline d={_DIM} EP=1 ckpts evaluated at seq={_SEQ} (=training seq_len), "
            f"long_sliding_window={_LONG_SWA} (no-op at seq=4k), paloma_bs={_PALOMA_BS} "
            f"(8.39M tokens/slice, matches seq=8k bs=128 token count). "
            f"{len(_CHECKPOINT_STEPS)} checkpoints. v4-32 us-central2."
        ),
    )
