# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams, NgramKind
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams

from experiments.datakit.reports.dedup import dedup_report


def test_dedup_report_surfaces_exact_verification_evidence(tmp_path):
    counter = "dedup/fuzzy/verification"
    source_main_dir = "s3://bucket/corpus/source/outputs/main"
    artifact = FuzzyDupsAttrData(
        params=MinHashParams(
            num_perms=286,
            num_bands=26,
            ngram_size=5,
            ngram_kind=NgramKind.WORD,
            seed=42,
        ),
        verification=FuzzyVerificationParams(),
        decisions_dir="s3://bucket/dedup/metadata/decisions",
        sources={
            source_main_dir: FuzzyDupsPerSource(
                attr_dir="s3://bucket/dedup/outputs/source_000",
            )
        },
        counters={
            f"{counter}/candidates": 100,
            f"{counter}/decision/accepted": 20,
            f"{counter}/decision/member_longer": 50,
            f"{counter}/decision/containment_below_threshold": 30,
            f"{counter}/source/source_000/decision/accepted": 20,
            f"{counter}/source/source_000/decision/member_longer": 50,
            f"{counter}/source/source_000/decision/containment_below_threshold": 30,
            f"{counter}/histogram/member_containment/100": 20,
            f"{counter}/histogram/jaccard/099": 20,
            f"{counter}/histogram/member_unique/0": 20,
            f"{counter}/histogram/shared_buckets/26": 20,
            "dedup/fuzzy/document/transitive_members_kept": 12,
        },
    )

    report = dedup_report(str(tmp_path), artifact)

    assert report.stats == {
        "candidates": 100,
        "verified_duplicates": 20,
        "rejected_candidates": 80,
        "acceptance_rate": 0.2,
        "transitive_members_kept": 12,
        "n_sources": 1,
    }
    html = Path(report.html_path).read_text()
    assert "candidate evidence:" in html
    assert "s3://bucket/dedup/metadata/decisions" in html
    assert "containment_below_threshold" in html
    assert "0.7184" in html
    assert "0.903206" in html
