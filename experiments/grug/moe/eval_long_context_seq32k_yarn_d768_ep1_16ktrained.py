# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL at seq=32k of the d=768 EP=1 16k-context-extended baseline.

The 16kctx training run (``moe_may_compute_opt_d768_ep1_16kctx_long_yarn_mscale01_from15k``)
resumed from the d=768 EP=1 baseline at step ~15k and trained the last
~1,875 steps at seq=16384 with YaRN (``long_yarn_old_seq_len=4096``,
``mscale_coef=0.1``).  Here we evaluate that 16k-trained checkpoint at
seq=32768 with a two-stage YaRN: the inv_freq is first rescaled
(4096 -> 16384) to reproduce the training-time rescaling, then further
rescaled (16384 -> 32768) starting from that already-rescaled inv_freq.
``long_qk_mult`` uses the total compression m-scale (32k/4k = 8x),
``~1.208x``.

Both saved checkpoints are scored (step 16000 + final 16875).

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq32k_yarn_d768_ep1_16ktrained
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.eval_long_context_seq32k_yarn import (
    LongContextYarnSweepEvalConfig,
    run_long_context_seq32k_yarn_eval,
)

_DIM: int = 768
_CKPT_DIR: str = (
    "gs://marin-us-central2/grug/moe_may_compute_opt_d768_ep1_16kctx_long_yarn_mscale01_from15k-1f6cd3/checkpoints"
)
_CHECKPOINT_STEPS: tuple[int, ...] = (16000, 16875)
_SEQ: int = 32_768
_BS: int = 16
_EP: int = 1
_MAX_DOCS: int = 32
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq32k_yarn_d{_DIM}_ep1_16ktrained_sweep"
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
        yarn_prior_seq_len=versioned(16384),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL at seq={_SEQ} for d={_DIM} EP=1 16k-trained baseline, "
            f"YaRN-extended from 16k. {len(_CHECKPOINT_STEPS)} checkpoints. v4-32 us-central2."
        ),
    )
