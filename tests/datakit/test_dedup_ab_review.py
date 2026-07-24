# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.datakit.scripts.dedup_ab_review import DedupLabels, validate_label_coverage


def _score(
    *,
    variant: str,
    role: str,
    doc_id: str,
    canonical_id: str,
    exact_raw_text: bool = False,
    evidence_class: str = "ambiguous",
) -> dict:
    return {
        "variant": variant,
        "role": role,
        "source_main_dir": f"s3://normalized/{variant}",
        "basename": "part-00000.parquet",
        "id": doc_id,
        "canonical_source_main_dir": f"s3://normalized/{variant}",
        "canonical_basename": "part-00000.parquet",
        "canonical_id": canonical_id,
        "exact_raw_text": exact_raw_text,
        "evidence_class": evidence_class,
    }


def _labels(*rows: dict) -> DedupLabels:
    return DedupLabels(
        scores_dir="s3://audit/scores",
        method="Every ambiguous pair was read in full.",
        labels=list(rows),
    )


def _label(
    *,
    variant: str = "baseline",
    member_id: str = "member",
    canonical_id: str = "canonical",
    label: str = "false_positive",
    method: str = "semantic",
) -> dict:
    return {
        "variant": variant,
        "member_source_main_dir": f"s3://normalized/{variant}",
        "member_basename": "part-00000.parquet",
        "member_id": member_id,
        "canonical_source_main_dir": f"s3://normalized/{variant}",
        "canonical_basename": "part-00000.parquet",
        "canonical_id": canonical_id,
        "label": label,
        "method": method,
        "basis": "Full member and canonical text were compared.",
    }


def test_complete_labels_cover_every_marker_and_drop() -> None:
    scores = [
        _score(variant="baseline", role="canonical", doc_id="canonical", canonical_id="canonical"),
        _score(variant="baseline", role="drop", doc_id="member", canonical_id="canonical"),
        _score(variant="treatment", role="canonical", doc_id="t-canonical", canonical_id="t-canonical"),
        _score(
            variant="treatment",
            role="drop",
            doc_id="t-member",
            canonical_id="t-canonical",
            exact_raw_text=True,
            evidence_class="strong_duplicate",
        ),
    ]
    labels = _labels(
        _label(),
        _label(
            variant="treatment",
            member_id="t-member",
            canonical_id="t-canonical",
            label="true_duplicate",
            method="raw_identity",
        ),
    )

    result = validate_label_coverage(scores, labels)

    assert result["score_markers"] == 4
    assert result["labeled_drops"] == 2
    assert result["covered_markers"] == 4
    assert result["variants"]["baseline"]["candidate_precision"] == 0.0
    assert result["variants"]["treatment"]["candidate_precision"] == 1.0


def test_missing_drop_label_is_rejected() -> None:
    scores = [
        _score(variant="baseline", role="canonical", doc_id="canonical", canonical_id="canonical"),
        _score(variant="baseline", role="drop", doc_id="member", canonical_id="canonical"),
    ]

    with pytest.raises(AssertionError, match="missing=1"):
        validate_label_coverage(scores, _labels())


def test_unreferenced_marker_is_rejected() -> None:
    scores = [
        _score(variant="baseline", role="canonical", doc_id="canonical", canonical_id="canonical"),
        _score(variant="baseline", role="drop", doc_id="member", canonical_id="canonical"),
        _score(variant="baseline", role="canonical", doc_id="orphan", canonical_id="orphan"),
    ]

    with pytest.raises(AssertionError, match="uncovered=1"):
        validate_label_coverage(scores, _labels(_label()))


def test_raw_identity_requires_exact_raw_text() -> None:
    scores = [
        _score(variant="baseline", role="canonical", doc_id="canonical", canonical_id="canonical"),
        _score(variant="baseline", role="drop", doc_id="member", canonical_id="canonical"),
    ]
    label = _label(label="true_duplicate", method="raw_identity")

    with pytest.raises(AssertionError, match="lacks exact raw identity"):
        validate_label_coverage(scores, _labels(label))


def test_low_overlap_requires_strong_false_positive_evidence() -> None:
    scores = [
        _score(variant="baseline", role="canonical", doc_id="canonical", canonical_id="canonical"),
        _score(
            variant="baseline",
            role="drop",
            doc_id="member",
            canonical_id="canonical",
            evidence_class="ambiguous",
        ),
    ]
    label = _label(method="low_overlap")

    with pytest.raises(AssertionError, match="lacks strong false-positive evidence"):
        validate_label_coverage(scores, _labels(label))
