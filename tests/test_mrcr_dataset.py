# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.text.formats import ChatLmDatasetFormat
from marin.execution.artifact import ArtifactRecord, write_record

from experiments.datasets.mrcr import MrcrTokenizedCache, MrcrTransformConfig, transform_mrcr


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

    def read(condition: str) -> dict[str, list[dict[str, str]]]:
        with gzip.open(output_path / "2needle" / f"{condition}.jsonl.gz", "rt") as f:
            return json.loads(f.read())

    full_context = read("full_context")
    final_user_only = read("final_user_only")

    full_messages = full_context["messages"]
    final_messages = final_user_only["messages"]
    assert full_messages[0]["content"].startswith("User: old context")
    assert full_messages[0]["content"].endswith(final_messages[0]["content"])
    assert final_messages[0] == {
        "role": "user",
        "content": "User: return the remembered answer\nAssistant: ",
    }
    assert full_messages[1] == final_messages[1] == {"role": "assistant", "content": "prefix-old answer"}


def test_mrcr_cache_uses_packed_right_sliced_chat_examples(tmp_path):
    write_record(
        ArtifactRecord(
            output_path=str(tmp_path),
            config={"tokenizer": "passthrough", "tags": ["mrcr/full_context"]},
        )
    )

    component = MrcrTokenizedCache.raw_load(str(tmp_path)).as_component()

    assert isinstance(component.format, ChatLmDatasetFormat)
    assert component.format.pack is True
    assert component.format.slice_strategy == "right"
    assert component.tags == ["mrcr/full_context"]
