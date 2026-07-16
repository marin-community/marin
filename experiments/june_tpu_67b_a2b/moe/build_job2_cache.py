# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off CPU job: build the Stage-2 (``nemotron_science_think``) tokenized cache and
report the packed 1-epoch step count from its ledger.

Runs the EXACT production data path (``build_grug_chat_data_config`` -> ``build_caches``)
so the cache is byte-identical to what Job 2's trainer would build -- and lands it on the
CW object store so Job 2 reuses it instead of re-tokenizing. Then reads the cache ledger's
``field_counts.input_ids`` and computes ``ceil(input_ids / (seq_len * batch))`` = 1 packed
epoch, the SAME procedure that finalized Job 1 to 257.

Submit (cw-us-east-02a, CPU coordinator; MARIN_PREFIX must be s3://marin-us-east-02a/marin)::

    cd ~/Documents/marin && export KUBECONFIG=~/.kube/coreweave-iris-gpu
    uv run iris --cluster=cw-us-east-02a job run --job-name grug-67b-job2-cache \\
      --cpu 32 --memory 128G --extra cpu --priority interactive --max-retries 2 --no-wait \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HF_TOKEN "$HF_TOKEN" \\
      -- python -m experiments.june_tpu_67b_a2b.moe.build_job2_cache
"""

import math

from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _BATCH, _JOB2_DATASET, _SEQ
from experiments.june_tpu_67b_a2b.moe.sft_launch import build_grug_chat_data_config
from experiments.marin_tokenizer import marin_tokenizer
from experiments.sft_launcher.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE


def main() -> None:
    data = build_grug_chat_data_config(
        datasets=[_JOB2_DATASET],
        tokenizer=marin_tokenizer,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        mixture_block_size=2048,
    )
    caches = data.build_caches("train")
    cache = caches[_JOB2_DATASET.slug]
    total_tokens = cache.ledger.field_counts["input_ids"]
    total_rows = cache.ledger.total_num_rows
    steps = math.ceil(total_tokens / (_SEQ * _BATCH))
    print(
        "JOB2_CACHE_RESULT "
        f"slug={_JOB2_DATASET.slug} rows={total_rows} input_ids={total_tokens} "
        f"seq={_SEQ} batch={_BATCH} steps={steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
