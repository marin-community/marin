# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.datakit.scripts.dedup_ab_upgrade_smoke_labels import upgrade_smoke_labels


def _historical_label(member_id: str = "member") -> dict:
    return {
        "version": "v1",
        "labels": [
            {
                "variant": "baseline",
                "source": "dataset",
                "member_id": member_id,
                "canonical_id": "canonical",
                "label": "false_positive",
                "basis": "The complete documents contain different substantive content.",
            }
        ],
    }


def _pair(member_id: str = "member") -> dict:
    return {
        "variant": "baseline",
        "member_source_main_dir": "s3://normalized/dataset/outputs/main",
        "member_basename": "part-00001.parquet",
        "member_id": member_id,
        "canonical_source_main_dir": "s3://normalized/canonical",
        "canonical_basename": "part-00002.parquet",
        "canonical_id": "canonical",
    }


def test_upgrade_binds_historical_label_to_materialized_locations() -> None:
    labels = upgrade_smoke_labels(
        historical=_historical_label(),
        pair_records=[_pair()],
        scores_dir="s3://audit/scores",
        pairs_dir="s3://review/pairs",
    )

    assert labels.version == "v2"
    assert labels.labels[0].member_basename == "part-00001.parquet"
    assert labels.labels[0].method == "semantic"


def test_upgrade_requires_exact_historical_and_pair_coverage() -> None:
    with pytest.raises(AssertionError, match="no historical label"):
        upgrade_smoke_labels(
            historical=_historical_label("different"),
            pair_records=[_pair()],
            scores_dir="s3://audit/scores",
            pairs_dir="s3://review/pairs",
        )

    with pytest.raises(AssertionError, match="no materialized pair"):
        upgrade_smoke_labels(
            historical=_historical_label(),
            pair_records=[],
            scores_dir="s3://audit/scores",
            pairs_dir="s3://review/pairs",
        )
