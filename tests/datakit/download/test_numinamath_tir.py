# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.download.numinamath_tir import (
    HF_DATASET_ID,
    HF_REVISION,
    TRAIN_PARQUET_GLOB,
    numinamath_tir_normalize_steps,
    row_to_doc,
)
from marin.datakit.sources import all_sources


def test_row_to_doc_renders_messages_as_tagged_transcript():
    row = {
        "messages": [
            {"role": "user", "content": "Solve $x + 1 = 3$."},
            {
                "role": "assistant",
                "content": "We can verify with code.\n```python\nprint(3 - 1)\n```\n```output\n2\n```\nThus x = 2.",
            },
        ]
    }

    [doc] = row_to_doc(row)

    assert doc["source"] == HF_DATASET_ID
    assert len(doc["id"]) == 64
    assert doc["text"] == (
        "<user>\n"
        "Solve $x + 1 = 3$.\n"
        "</user>\n\n"
        "<assistant>\n"
        "We can verify with code.\n"
        "```python\n"
        "print(3 - 1)\n"
        "```\n"
        "```output\n"
        "2\n"
        "```\n"
        "Thus x = 2.\n"
        "</assistant>"
    )


def test_row_to_doc_preserves_supported_message_order():
    row = {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Compute 2 + 2."},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
    }

    [doc] = row_to_doc(row)

    assert doc["text"] == (
        "<system>\nYou are a math tutor.\n</system>\n\n"
        "<user>\nCompute 2 + 2.\n</user>\n\n"
        "<tool>\n4\n</tool>\n\n"
        "<assistant>\nThe answer is 4.\n</assistant>"
    )


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": "not-a-list"},
        {"messages": []},
        {"messages": [{"role": "critic", "content": "Nope."}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": None}]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"content": "Missing role."}]},
        {"messages": [{"role": "user", "content": "Hi"}, "bad-message"]},
    ],
)
def test_row_to_doc_drops_invalid_rows(row):
    assert row_to_doc(row) == []


def test_numinamath_tir_normalize_steps_use_train_split_and_stable_names():
    processed, normalized = numinamath_tir_normalize_steps()
    download = processed.deps[0]

    assert download.name == "raw/numinamath-tir"
    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == [TRAIN_PARQUET_GLOB]
    assert processed.name == "processed/numinamath-tir"
    assert processed.deps == [download]
    assert normalized.name == "normalized/numinamath-tir"
    assert normalized.deps == [processed]


def test_numinamath_tir_is_registered_as_datakit_source():
    source = all_sources()["numinamath-tir"]

    assert source.rough_token_count_b == 0.08
    assert source.normalize_steps[0].name == "processed/numinamath-tir"
    assert source.normalized.name == "normalized/numinamath-tir"
