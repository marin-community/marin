# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.dedup_ab_machine_labels import decision_for_pair
from experiments.datakit.scripts.dedup_ab_semantic_batch import load_semantic_cases
from experiments.datakit.scripts.dedup_ab_semantic_workload import merge_summaries, summarize_pairs


def _pair(index: int) -> dict:
    member_text = f"member document {index}"
    canonical_text = f"canonical document {index}"
    return {
        "review_key": f"pair-{index}",
        "variant": "baseline",
        "member_source_main_dir": "member-source",
        "member_basename": "part.parquet",
        "member_id": f"member-{index}",
        "canonical_source_main_dir": "canonical-source",
        "canonical_basename": "part.parquet",
        "canonical_id": f"canonical-{index}",
        "raw_sha256": hashlib.sha256(member_text.encode()).hexdigest(),
        "canonical_raw_sha256": hashlib.sha256(canonical_text.encode()).hexdigest(),
        "member_text": member_text,
        "canonical_text": canonical_text,
        "exact_raw_text": False,
        "evidence_class": "ambiguous",
        "cross_source": True,
        "raw_chars": len(member_text),
        "canonical_raw_chars": len(canonical_text),
        "length_ratio": len(member_text) / len(canonical_text),
        "member_is_longer": False,
        "member_text_truncated_for_minhash": False,
        "canonical_text_truncated_for_minhash": False,
        "exact_clean_text": False,
        "member_clean_text_contained": False,
        "char_5gram_jaccard": 0.1,
        "char_5gram_canonical_containment": 0.2,
        "char_5gram_member_containment": 0.2,
        "word_5gram_jaccard": 0.1,
        "word_5gram_canonical_containment": 0.2,
        "word_5gram_member_containment": 0.2,
        "baseline_shared_buckets": 1,
        "treatment_shared_buckets": 0,
    }


def _decision(pair: dict, path: str, row_index: int) -> dict:
    return {
        **decision_for_pair(pair),
        "pair_path": path,
        "pair_row_index": row_index,
    }


def test_semantic_batch_reads_exact_rows_across_parquet_row_groups(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pairs = [_pair(index) for index in range(5)]
    pq.write_table(pa.Table.from_pylist(pairs), path, row_group_size=2)
    decisions = [_decision(pair, str(path), index) for index, pair in enumerate(pairs)]

    cases, total = load_semantic_cases(decisions, semantic_offset=1, limit=3)

    assert total == 5
    assert [case["review_key"] for case in cases] == ["pair-1", "pair-2", "pair-3"]
    assert [case["member_text"] for case in cases] == [pairs[index]["member_text"] for index in (1, 2, 3)]


def test_semantic_batch_rejects_tampered_row_reference(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pairs = [_pair(0), _pair(1)]
    pq.write_table(pa.Table.from_pylist(pairs), path, row_group_size=1)
    decision = _decision(pairs[0], str(path), 1)

    with pytest.raises(AssertionError, match="differs from referenced pair"):
        load_semantic_cases([decision], semantic_offset=0, limit=1)


def test_semantic_batch_rejects_missing_row(tmp_path) -> None:
    path = tmp_path / "pairs.parquet"
    pair = _pair(0)
    pq.write_table(pa.Table.from_pylist([pair]), path)
    decision = _decision(pair, str(path), 2)

    with pytest.raises(IndexError, match="outside"):
        load_semantic_cases([decision], semantic_offset=0, limit=1)


def test_semantic_workload_accounts_only_routed_full_text_pairs() -> None:
    baseline = _pair(0)
    treatment = {**_pair(1), "variant": "treatment"}
    exact = _pair(2)
    exact["canonical_text"] = exact["member_text"]
    exact["canonical_raw_sha256"] = exact["raw_sha256"]
    exact["canonical_raw_chars"] = exact["raw_chars"]
    exact["exact_raw_text"] = True

    first = summarize_pairs([baseline, exact])
    second = summarize_pairs([treatment])
    summary = merge_summaries([first, second])

    expected_raw_chars = sum(len(pair["member_text"]) + len(pair["canonical_text"]) for pair in (baseline, treatment))
    assert summary["pairs"] == 3
    assert summary["semantic_pairs"] == 2
    assert summary["semantic_raw_chars"] == expected_raw_chars
    assert summary["semantic_review_units"] == 2
    assert summary["minimum_model_requests"] == 4
    assert summary["maximum_model_requests"] == 6
    assert summary["direct_pairs"] == 2
    assert summary["chunked_pairs"] == 0
    assert summary["counts"]["baseline/pairs"] == 1
    assert summary["counts"]["treatment/pairs"] == 1
    assert summary["counts"]["baseline/review_units"] == 1
    assert summary["counts"]["treatment/review_units"] == 1
    assert summary["counts"]["baseline/cross_source/True"] == 1
