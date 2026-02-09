# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Given a directory of evaluation files, convert them to Dolma format.

The Dolma format is a JSONL file with a "text" field.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --cluster=us-central2 \
    lib/marin/src/marin/transform/evaluation/eval_to_dolma.py \
    --input_path gs://marin-us-central2/raw/evaluation/ \
    --output_path gs://marin-data/processed/evaluation/dolma
"""

from dataclasses import dataclass

import draccus
from zephyr import Dataset, ZephyrContext, load_jsonl


@dataclass
class ConvertEvalToDolmaConfig:
    input_path: str
    output_path: str


def map_row(row: dict):
    """Transform evaluation record to Dolma format.

    Args:
        row: Record with "prompt" and "response" fields

    Returns:
        Record with "text" field containing prompt + response
    """
    row["text"] = row["prompt"] + "\n" + row["response"]
    return row


@draccus.wrap()
def convert_eval_to_dolma(cfg: ConvertEvalToDolmaConfig):
    pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map(map_row)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    with ZephyrContext(name="eval-to-dolma") as ctx:
        ctx.execute(pipeline)


if __name__ == "__main__":
    convert_eval_to_dolma()
