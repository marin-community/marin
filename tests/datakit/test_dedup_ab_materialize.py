# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.dedup_ab_materialize import _join_requested_texts, _pair_texts, _review_key


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


def test_pair_texts_retains_full_audit_metrics() -> None:
    member_text = "member"
    canonical_text = "canonical"
    score = {
        "variant": "treatment",
        "source_main_dir": "s3://normalized/member",
        "basename": "part-00001.parquet",
        "id": "member",
        "canonical_source_main_dir": "s3://normalized/canonical",
        "canonical_basename": "part-00002.parquet",
        "canonical_id": "canonical",
        "evidence_class": "ambiguous",
        "cross_source": True,
        "raw_chars": len(member_text),
        "canonical_raw_chars": len(canonical_text),
        "clean_chars": len(member_text),
        "canonical_clean_chars": len(canonical_text),
        "length_ratio": len(member_text) / len(canonical_text),
        "member_is_longer": False,
        "member_text_truncated_for_minhash": False,
        "canonical_text_truncated_for_minhash": True,
        "exact_raw_text": False,
        "exact_clean_text": False,
        "member_clean_text_contained": False,
        "char_5gram_jaccard": 0.1,
        "char_5gram_canonical_containment": 0.2,
        "char_5gram_member_containment": 0.3,
        "word_5gram_jaccard": 0.4,
        "word_5gram_canonical_containment": 0.5,
        "word_5gram_member_containment": 0.6,
        "baseline_shared_buckets": 1,
        "treatment_shared_buckets": 2,
        "raw_sha256": _sha256(member_text),
        "canonical_raw_sha256": _sha256(canonical_text),
    }
    review_key = _review_key(score)
    records = [
        {
            "review_key": review_key,
            "side": "member",
            "score_json": json.dumps(score),
            "text": member_text,
        },
        {
            "review_key": review_key,
            "side": "canonical",
            "score_json": "",
            "text": canonical_text,
        },
    ]

    pair = _pair_texts(review_key, iter(records))

    assert pair["canonical_text_truncated_for_minhash"] is True
    assert pair["word_5gram_member_containment"] == 0.6
    assert pair["member_text"] == member_text
    assert pair["canonical_text"] == canonical_text
