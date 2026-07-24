# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from experiments.datakit.scripts import dedup_ab_audit
from experiments.datakit.scripts.dedup_ab_audit import (
    _cc_distance_entries,
    _evidence_class,
    _graph_distance_records,
    _set_metrics,
    _validate_score_counts,
)


def test_set_metrics_preserves_containment_direction() -> None:
    canonical = {"shared", "canonical-only"}
    member = {"shared"}

    jaccard, canonical_containment, member_containment = _set_metrics(canonical, member)

    assert jaccard == 0.5
    assert canonical_containment == 0.5
    assert member_containment == 1.0


def test_longer_dropped_member_with_unique_content_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.49,
        canonical_word_containment=0.99,
        member_word_containment=0.49,
    )

    assert evidence == "ambiguous"


def test_clean_member_containment_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.4,
        canonical_word_containment=0.4,
        member_word_containment=1.0,
    )

    assert evidence == "ambiguous"


def test_exact_raw_text_confirms_redundancy() -> None:
    evidence = _evidence_class(
        exact_raw_text=True,
        word_jaccard=1.0,
        canonical_word_containment=1.0,
        member_word_containment=1.0,
    )

    assert evidence == "strong_duplicate"


def test_low_bidirectional_overlap_confirms_false_positive() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.03,
        canonical_word_containment=0.08,
        member_word_containment=0.12,
    )

    assert evidence == "strong_false_positive"


def test_low_jaccard_with_containment_requires_review() -> None:
    evidence = _evidence_class(
        exact_raw_text=False,
        word_jaccard=0.03,
        canonical_word_containment=0.03,
        member_word_containment=0.95,
    )

    assert evidence == "ambiguous"


def test_graph_distance_discovers_iterations_beyond_original_run_cap(monkeypatch) -> None:
    iteration_count = 58

    def fake_cc_shards(directory: str) -> dict[int, str]:
        iteration = int(directory.rsplit("it_", 1)[1])
        if iteration >= iteration_count:
            return {}
        return {0: f"{directory}/part-00000.parquet"}

    monkeypatch.setattr(dedup_ab_audit, "_cc_shards", fake_cc_shards)

    entries = _cc_distance_entries("s3://bucket/dedup", {"s3://bucket/source"})

    assert len(entries) == 1
    assert len(entries[0]["iteration_paths"]) == iteration_count
    assert entries[0]["iteration_paths"][-1].endswith("/it_57/part-00000.parquet")


def test_graph_distance_rejects_a_capped_nonconverged_shard(monkeypatch) -> None:
    final_table = pa.Table.from_pylist(
        [
            {
                "record_id": "source_000|doc",
                "id_norm": "1",
                "adjacency_list": ["2"],
                "component_id": "1",
                "changed": True,
            }
        ]
    )
    monkeypatch.setattr(dedup_ab_audit, "_read_table", lambda path, columns: final_table.select(columns))

    with pytest.raises(AssertionError, match="still has 1 changed nodes"):
        list(
            _graph_distance_records(
                {
                    "shard_index": 0,
                    "iteration_paths": ["s3://bucket/dedup/metadata/cc/it_50/part-00000.parquet"],
                    "source_by_tag": {"source_000": "s3://bucket/source"},
                }
            )
        )


def test_score_counts_require_every_artifact_marker_and_drop() -> None:
    baseline = {
        "counters": {
            "dedup/fuzzy/document/cluster_members": 13,
            "dedup/fuzzy/document/canonicals": 5,
        }
    }
    treatment = {
        "counters": {
            "dedup/fuzzy/document/cluster_members": 4,
            "dedup/fuzzy/document/canonicals": 2,
        }
    }
    counters = {
        "audit/markers/baseline": 13,
        "audit/drops/baseline": 8,
        "audit/markers/treatment": 4,
        "audit/drops/treatment": 2,
    }

    _validate_score_counts(counters, baseline, treatment)

    counters["audit/drops/treatment"] = 1
    with pytest.raises(AssertionError, match="treatment score coverage mismatch"):
        _validate_score_counts(counters, baseline, treatment)
