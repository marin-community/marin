# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.datakit.scripts.dedup_ab_marker_relation_review import (
    _kept_by_datakit,
    canonical_inventory,
    canonical_locations,
    relationship_pair,
)


def _marker_record(cluster_id: str, canonical: bool, doc_id: str) -> dict:
    return {
        "source_main_dir": "s3://normalized/source",
        "basename": "part.parquet",
        "id": doc_id,
        "marker": {
            "dup_cluster_id": cluster_id,
            "is_cluster_canonical": canonical,
        },
    }


def test_canonical_locations_requires_exactly_one_canonical_per_cluster() -> None:
    records = [
        _marker_record("a", False, "a-member"),
        _marker_record("a", True, "a-canonical"),
        _marker_record("b", True, "b-canonical"),
    ]

    locations = canonical_locations(records, {"a", "b"})

    assert locations == {
        "a": {
            "source_main_dir": "s3://normalized/source",
            "basename": "part.parquet",
            "id": "a-canonical",
        },
        "b": {
            "source_main_dir": "s3://normalized/source",
            "basename": "part.parquet",
            "id": "b-canonical",
        },
    }


def test_canonical_locations_rejects_missing_and_duplicate_canonicals() -> None:
    with pytest.raises(AssertionError, match="absent"):
        canonical_locations([_marker_record("a", False, "member")], {"a"})

    with pytest.raises(AssertionError, match="multiple canonicals"):
        canonical_locations(
            [
                _marker_record("a", True, "canonical-1"),
                _marker_record("a", True, "canonical-2"),
            ],
            {"a"},
        )


def test_canonical_inventory_records_nonconverged_orphan_labels() -> None:
    found, orphaned = canonical_inventory(
        [
            _marker_record("a", False, "a-member"),
            _marker_record("b", True, "b-canonical"),
        ],
        {"a", "b"},
    )

    assert found == {
        "b": {
            "source_main_dir": "s3://normalized/source",
            "basename": "part.parquet",
            "id": "b-canonical",
        }
    }
    assert orphaned == ["a"]


def test_datakit_keep_decision_ignores_cluster_id_changes() -> None:
    difference = {
        "capped_is_canonical": False,
        "converged_is_canonical": False,
        "capped_cluster_id": "old",
        "converged_cluster_id": "new",
    }

    assert _kept_by_datakit(difference, "capped") is False
    assert _kept_by_datakit(difference, "converged") is False


def test_relationship_pair_preserves_direction_and_routes_semantic_review() -> None:
    difference = {
        "source_main_dir": "s3://normalized/member",
        "basename": "member.parquet",
        "id": "member",
        "change_kind": "attributes_changed",
        "capped_cluster_id": "old",
        "converged_cluster_id": "new",
    }
    canonical = {
        "source_main_dir": "s3://normalized/canonical",
        "basename": "canonical.parquet",
        "id": "canonical",
    }

    pair = relationship_pair(
        difference=difference,
        variant="baseline_cap50",
        cluster_id="old",
        canonical=canonical,
        member_text="unique member payload",
        canonical_text="different canonical payload",
    )

    assert pair["variant"] == "baseline_cap50"
    assert pair["relationship_cluster_id"] == "old"
    assert pair["member_id"] == "member"
    assert pair["canonical_id"] == "canonical"
    assert pair["evidence_class"] == "ambiguous"
    assert pair["metric_evidence_class"] == "strong_false_positive"
    assert pair["exact_raw_text"] is False
    assert pair["cross_source"] is True
