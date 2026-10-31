# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen validation prompts against the published uniform token budgets."""

import json
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages
from transformers import AutoTokenizer

from experiments.evaluation.mini_policy_runtime.artifacts import publish
from experiments.evaluation.mini_policy_runtime.profile import (
    DATASET,
    MAX_INPUT_TOKENS,
    MODEL,
    MODEL_REVISION,
    REVISION,
    TOTAL_CONTEXT_LENGTH,
)


def main():
    data = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    rows = []
    counts = defaultdict(list)
    for row in pq.read_table(data).to_pylist():
        task = TaskSpec.model_validate_json(row["task_spec"])
        messages = conversation_messages(task.context.events)
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False
        )
        length = len(tokens)
        counts[row["benchmark"]].append(length)
        rows.append({"task_id": row["task_id"], "benchmark": row["benchmark"], "prompt_tokens": length})
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "tokenizer": MODEL,
        "tokenizer_revision": MODEL_REVISION,
        "thinking": True,
        "summary": {
            b: {
                "count": len(v),
                "max_prompt_tokens": max(v),
                "over_49152": sum(n > MAX_INPUT_TOKENS for n in v),
                "over_65536": sum(n > TOTAL_CONTEXT_LENGTH for n in v),
            }
            for b, v in counts.items()
        },
        "rows": rows,
    }
    publish("prompt-inventory.json", report)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
