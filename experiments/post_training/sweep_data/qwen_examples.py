# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one #202 training trajectory per family, rendered by qwen_render, for qwen_vllm_check.py (task #230).

TMax: the first training trajectory with one user message, reasoning on every turn and under 12 messages. SWE-smith
(Kimi, terminus-2): the first training trajectory with 6 to 11 messages.

    python -m experiments.post_training.sweep_data.qwen_examples <tokenizer dir> <#202 train-index.jsonl> <out.json>
"""

import glob
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from experiments.post_training.sweep_data.prepare_agentic import KIMI, TMAX
from experiments.post_training.sweep_data.qwen_render import every_turn_thinks, render
from experiments.post_training.sweep_data.render import kimi_messages, tmax_messages


def main() -> None:
    tokenizer_dir, index_path, output = sys.argv[1:4]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    template = every_turn_thinks(tokenizer.chat_template)
    with open(index_path) as stream:
        train = {entry["trajectory"] for entry in map(json.loads, stream) if "trajectory" in entry}
    tmax_dir = snapshot_download(TMAX[0], repo_type="dataset", revision=TMAX[1], allow_patterns=[f"{TMAX[2]}/*"])
    kimi_dir = snapshot_download(KIMI[0], repo_type="dataset", revision=KIMI[1], allow_patterns=["data/*"])
    tmax = next(
        row
        for row in pq.read_table(glob.glob(f"{tmax_dir}/{TMAX[2]}/*.parquet")[0]).to_pylist()
        if row["metadata"]["trial_name"] in train
        and sum(m["role"] == "user" for m in row["messages"]) == 1
        and all((m["reasoning_content"] or "").strip() for m in row["messages"] if m["role"] == "assistant")
        and len(row["messages"]) < 12
    )
    kimi = next(
        row
        for path in sorted(glob.glob(f"{kimi_dir}/data/*.parquet"))
        for row in pq.read_table(path).to_pylist()
        if row["trial_name"] in train and 6 <= len(row["conversations"]) < 12
    )
    examples = []
    for family, name, (messages, tools) in (
        ("tmax", tmax["metadata"]["trial_name"], tmax_messages(tmax)),
        ("swesmith", kimi["trial_name"], (kimi_messages(kimi), None)),
    ):
        rendered = render(tokenizer, template, messages, tools)
        examples.append(
            {
                "family": family,
                "trajectory": name,
                "messages": messages,
                "tools": tools,
                "ids": rendered.ids,
                "spans": rendered.turn_spans,
            }
        )
    with open(output, "w") as stream:
        json.dump(examples, stream)


if __name__ == "__main__":
    main()
