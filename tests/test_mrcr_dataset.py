# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datasets.mrcr import MrcrTransformConfig, transform_mrcr


def test_transform_mrcr_builds_paired_prompts_with_identical_targets(tmp_path):
    input_path = tmp_path / "input" / "2needle"
    input_path.mkdir(parents=True)
    messages = [
        {"role": "user", "content": "old context " * 2_100},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "return the remembered answer"},
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": json.dumps(messages),
                    "answer": "prefix-old answer",
                    "n_needles": 2,
                }
            ]
        ),
        input_path / "2needle_0.parquet",
    )

    output_path = tmp_path / "output"
    transform_mrcr(MrcrTransformConfig(input_path=str(tmp_path / "input"), output_path=str(output_path)))

    def read(condition: str) -> dict[str, str]:
        with gzip.open(output_path / "2needle" / "4k-8k" / f"{condition}.jsonl.gz", "rt") as f:
            return json.loads(f.read())

    full_context = read("full_context")
    final_user_only = read("final_user_only")

    assert full_context["input"].startswith("User: old context")
    assert full_context["input"].endswith(final_user_only["input"])
    assert final_user_only["input"] == "User: return the remembered answer\nAssistant: "
    assert full_context["target"] == final_user_only["target"] == "prefix-old answer"
