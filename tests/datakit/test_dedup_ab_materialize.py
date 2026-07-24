# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.dedup_ab_materialize import _join_requested_texts, _review_key


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_review_key_includes_complete_member_and_canonical_location() -> None:
    score = {
        "variant": "baseline",
        "source_main_dir": "s3://normalized/member",
        "basename": "part-00001.parquet",
        "id": "member",
        "canonical_source_main_dir": "s3://normalized/canonical",
        "canonical_basename": "part-00002.parquet",
        "canonical_id": "canonical",
    }

    assert _review_key(score) == (
        '["baseline","s3://normalized/member","part-00001.parquet","member",'
        '"s3://normalized/canonical","part-00002.parquet","canonical"]'
    )


def test_join_requested_texts_reads_full_text_and_verifies_hash(tmp_path) -> None:
    normalized_path = tmp_path / "part-00000.parquet"
    text = "complete text, including the part beyond a MinHash cap"
    pq.write_table(
        pa.table({"id": ["a", "b"], "text": ["first", text]}),
        normalized_path,
    )
    request = {
        "review_key": "pair",
        "side": "member",
        "doc_id": "b",
        "expected_sha256": _sha256(text),
        "score_json": "{}",
    }

    records = list(_join_requested_texts(str(normalized_path), iter([request])))

    assert records == [
        {
            "review_key": "pair",
            "side": "member",
            "score_json": "{}",
            "text": text,
        }
    ]


def test_join_requested_texts_rejects_hash_mismatch(tmp_path) -> None:
    normalized_path = tmp_path / "part-00000.parquet"
    pq.write_table(pa.table({"id": ["a"], "text": ["actual"]}), normalized_path)
    request = {
        "review_key": "pair",
        "side": "member",
        "doc_id": "a",
        "expected_sha256": _sha256("different"),
        "score_json": "{}",
    }

    with pytest.raises(AssertionError, match="hash changed"):
        list(_join_requested_texts(str(normalized_path), iter([request])))
