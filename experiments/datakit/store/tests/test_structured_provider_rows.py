# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash
from marin.datakit import normalize

from experiments.datakit.store.reconstruct_selected_source import provider_rows
from experiments.datakit.store.structured_provider_rows import source_adapter, swe_openhands_provider_paths


@pytest.mark.parametrize(
    "name,raw,expected",
    [
        (
            "molmo2-cap",
            {"merged_caption": " A scene ", "frame_timestamps": [1.234, 3.0], "frame_captions": [" Moving ", "Done"]},
            "A scene\n\n[Frame at 1.23s] Moving\n\n[Frame at 3.00s] Done",
        ),
        (
            "numinamath-1.5",
            {
                "problem": " 1+1? ",
                "solution": " 2 ",
                "problem_is_valid": "Yes",
                "solution_is_valid": "Yes",
                "source": "original-contest",
                "answer": "2",
                "problem_type": "math",
                "question_type": "answer",
                "synthetic": False,
            },
            "<user>\n1+1?\n</user>\n\n<assistant>\n2\n</assistant>",
        ),
        (
            "numinamath-tir",
            {"messages": [{"role": "user", "content": " Compute "}, {"role": "tool", "content": " 2 "}]},
            "<user>\nCompute\n</user>\n\n<tool>\n2\n</tool>",
        ),
        (
            "superior-reasoning",
            {"input": "Compute", "output": "<think>Reason</think>\n2"},
            "<user>\nCompute\n</user>\n\n<assistant>\nReason\n2\n</assistant>",
        ),
        (
            "swe-rebench-openhands",
            {"trajectory": json.dumps([{"role": "user", "content": "Fix"}]), "resolved": 0},
            "This trajectory failed to solve the task.\n\n<user>\nFix\n</user>",
        ),
    ],
)
def test_original_parquet_projection_reconstructs_production_text_and_canonical_hash(tmp_path, name, raw, expected):
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist([{**raw, "text": "incorrect staged-text shortcut"}]), path)
    adapter = source_adapter(name)
    projected = list(provider_rows(str(path), list(adapter.columns), 8))
    documents = adapter.row_to_docs(projected[0])
    assert len(documents) == 1
    document = normalize._make_normalize_fn("text", "id")(documents[0])
    document = normalize._make_whitespace_compactor(128)(document)
    assert document["text"] == expected
    assert document["id"] == xxhash.xxh3_128_hexdigest(expected.encode())
    if name == "numinamath-1.5":
        assert document["numina_source"] == "original-contest"
        assert adapter.row_to_docs({**raw, "solution_is_valid": "No"}) == []


def test_swe_inventory_matches_production_parquet_glob_and_never_treats_tools_json_as_rows():
    assert swe_openhands_provider_paths(["raw/tools.json", "raw/nested/trajectory.parquet"]) == [
        "raw/nested/trajectory.parquet"
    ]
    with pytest.raises(ValueError, match="no Parquet"):
        swe_openhands_provider_paths(["raw/tools.json"])
