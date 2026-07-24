# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
)
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams, NgramKind

from experiments.datakit.reports.dedup import dedup_report


def test_dedup_report_counts_transitive_members_as_kept_documents(tmp_path):
    attr_dir = tmp_path / "attrs"
    attr_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "canonical",
                    "attributes": {"dup_cluster_id": "cluster", "is_cluster_canonical": True},
                },
                {
                    "id": "duplicate-a",
                    "attributes": {"dup_cluster_id": "cluster", "is_cluster_canonical": False},
                },
                {
                    "id": "duplicate-b",
                    "attributes": {"dup_cluster_id": "cluster", "is_cluster_canonical": False},
                },
            ]
        ),
        attr_dir / "part-00000.parquet",
    )
    dedup = FuzzyDupsAttrData(
        params=MinHashParams(
            num_perms=286,
            num_bands=26,
            ngram_size=5,
            ngram_kind=NgramKind.WORD,
            seed=42,
        ),
        sources={
            str(tmp_path / "normalized" / "outputs" / "main"): FuzzyDupsPerSource(attr_dir=str(attr_dir)),
        },
        counters={
            "dedup/fuzzy/document/cluster_members": 3,
            "dedup/fuzzy/document/canonicals": 1,
            "dedup/fuzzy/document/singletons_skipped": 4,
            "dedup/fuzzy/document/transitive_members_kept": 2,
        },
    )
    output_dir = tmp_path / "report"
    output_dir.mkdir()

    report = dedup_report(str(output_dir), dedup)

    assert report.stats["duplicates_to_drop"] == 2
    assert report.stats["transitive_members_kept"] == 2
    assert report.stats["dup_rate"] == pytest.approx(2 / 9)
    html = (output_dir / "report.html").read_text()
    assert "transitive members kept" in html
    assert '"transitive_members_kept": 2' in html
