# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "bench" / "sample_packed_doc_lengths.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("sample_packed_doc_lengths", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
sample_packed_doc_lengths = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(sample_packed_doc_lengths)

_texts_from_jsonl = sample_packed_doc_lengths._texts_from_jsonl
_texts_from_text_file = sample_packed_doc_lengths._texts_from_text_file
_pack_token_lengths = sample_packed_doc_lengths._pack_token_lengths


@pytest.mark.parametrize(
    ("token_lengths", "num_packs", "max_docs_per_pack", "expected"),
    [
        ([5, 7, 11], 1, None, [(5, 7, 4)]),
        ([3, 4, 5, 6, 7], 2, 2, [(3, 4, 9), (5, 6, 5)]),
        ([20, 3], 1, None, [(16,)]),
    ],
)
def test_pack_token_lengths(
    token_lengths: list[int],
    num_packs: int,
    max_docs_per_pack: int | None,
    expected: list[tuple[int, ...]],
):
    packs = _pack_token_lengths(token_lengths, seq_len=16, num_packs=num_packs, max_docs_per_pack=max_docs_per_pack)

    assert packs == expected


def test_pack_token_lengths_requires_enough_input():
    with pytest.raises(ValueError):
        _pack_token_lengths([3, 4], seq_len=16, num_packs=2, max_docs_per_pack=None)


def test_texts_from_jsonl_supports_nested_keys(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    rows = [
        {"conversation": {"text": "hello"}},
        {"conversation": {"text": ["a", "b"]}},
        {"conversation": {"text": ""}},
    ]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    assert list(_texts_from_jsonl(str(path), text_key="conversation.text")) == ["hello", "a\nb"]


def test_texts_from_text_file_skips_empty_lines(tmp_path: Path):
    path = tmp_path / "docs.txt"
    path.write_text("first\n\nsecond\n")

    assert list(_texts_from_text_file(str(path))) == ["first", "second"]
