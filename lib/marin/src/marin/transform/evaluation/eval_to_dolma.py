# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from zephyr import Dataset, flow_backend, load_jsonl


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
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(cfg.input_path, "**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map(map_row)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))


if __name__ == "__main__":
    convert_eval_to_dolma()
