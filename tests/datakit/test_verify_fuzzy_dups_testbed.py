# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams

from experiments.datakit.scripts.verify_fuzzy_dups_testbed import _normalize_candidate_source_keys


def _candidates(sources: dict[str, FuzzyDupsPerSource]) -> FuzzyDupsAttrData:
    return FuzzyDupsAttrData(
        version="v1",
        params=MinHashParams(num_perms=286, num_bands=26, ngram_size=5, seed=42),
        sources=sources,
        counters={},
    )


def test_imported_candidates_normalize_legacy_source_paths(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    source_path = "s3://marin-us-east-02a/marin/datakit/sample/source/outputs/main"
    source = FuzzyDupsPerSource(attr_dir="s3://marin-us-east-02a/marin/attrs/source")

    imported = _normalize_candidate_source_keys(_candidates({source_path: source}))

    assert imported.sources == {"datakit/sample/source/outputs/main": source}


def test_imported_candidates_reject_colliding_source_keys(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    source_key = "datakit/sample/source/outputs/main"
    source_path = f"s3://marin-us-east-02a/marin/{source_key}"

    with pytest.raises(ValueError, match="normalize to the same key"):
        _normalize_candidate_source_keys(
            _candidates(
                {
                    source_key: FuzzyDupsPerSource(attr_dir="/attrs/relative"),
                    source_path: FuzzyDupsPerSource(attr_dir="/attrs/absolute"),
                }
            )
        )
