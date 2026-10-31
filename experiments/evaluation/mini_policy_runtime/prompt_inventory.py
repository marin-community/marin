# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen validation prompts against the published uniform token budgets."""

import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages
from transformers import AutoTokenizer

from experiments.evaluation.mini_policy_runtime.profile import DATASET, DESTINATION, MODEL, MODEL_REVISION, REVISION


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
                "over_49152": sum(n > 49152 for n in v),
                "over_65536": sum(n > 65536 for n in v),
            }
            for b, v in counts.items()
        },
        "rows": rows,
    }
    encoded = json.dumps(report, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "prompt-inventory.json").write_bytes(encoded)
    fs, key = filesystem_for(str(StoragePath(DESTINATION) / "prompt-inventory.json"))
    fs.pipe_file(key, encoded)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
