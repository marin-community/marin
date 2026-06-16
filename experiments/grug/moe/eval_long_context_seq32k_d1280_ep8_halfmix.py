# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL at seq=32k of the d=1280 EP=8 32kctx halfmix checkpoint.

The 32kctx training run (``moe_may_compute_opt_d1280_ep8_32kctx_long_yarn_mscale01_halfmix_from13k``)
resumed from the d=1280 baseline at step ~13k and trained the last ~1,325
steps at seq=32768 on a 50/50 nemotron+longmino mix, with YaRN already
applied as ``long_yarn_old_seq_len=4096`` (matching the existing 32kctx
eval).  Here we evaluate that checkpoint at seq=32k using the same
training-time YaRN config -- no additional rescaling, since the model
was trained at 32k.

Routes through the same sweep runner as the other long-context evals
for methodology consistency.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_seq32k_d1280_ep8_halfmix
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.eval_long_context_seq32k_yarn import (
    LongContextYarnSweepEvalConfig,
    run_long_context_seq32k_yarn_eval,
)

_DIM: int = 1280
_CKPT_DIR: str = (
    "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep8_32kctx_long_yarn_mscale01_halfmix_from13k-e55766/"
    "checkpoints"
)
_CHECKPOINT_STEPS: tuple[int, ...] = (14325,)
_SEQ: int = 32_768
_BS: int = 16
_EP: int = 8
_MAX_DOCS: int = 32
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"

_run_id = f"moe_may_eval_long_context_seq32k_d{_DIM}_ep8_halfmix_sweep"
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
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL at seq={_SEQ} for d={_DIM} EP=8 32kctx halfmix final ckpt. "
            f"Original training YaRN config (yarn_old_seq_len=4096). v4-32 us-central2."
        ),
    )
