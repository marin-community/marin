# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality explanations must describe receipts, without inventing campaign intent."""

from copy import deepcopy
from typing import Any

import pytest

from infra.marina.apps.rl_data_catalog.server.quality import Quality
from infra.marina.apps.rl_data_catalog.server.quality_evidence import FullReviewReason, quality_explanation


def declaration() -> dict[str, Any]:
    return {
        "id": "Marin:example",
        "dataset_id": "org/data",
        "dataset_revision": "a" * 40,
        "pipeline": {"version": "1"},
        "actor_census_input": {"task_specs_sha256": "b" * 64, "measured_tasks": 1000},
        "within_source_dedup": {"unique_count": 1000},
        "quality_result": {"reviewed": 96, "sample_count": 100},
        "independent_qc": {"reviewed": 96, "evidence_url": "s3://bucket/sample"},
    }


def full_record() -> dict[str, Any]:
    return {
        "receipt": {
            "artifact": {"uri": "s3://bucket/full", "sha256": "c" * 64},
            "panel": {
                "dataset_id": "org/data",
                "dataset_revision": "a" * 40,
                "converter_version": "1",
                "task_specs_sha256": "b" * 64,
                "retained_tasks": 1000,
            },
        },
        "summary": {"reviewed": 800},
    }


def test_no_recorded_full_review_keeps_sample_receipt_and_exact_counts():
    source = declaration()
    source.update(build_path="MarinSkyRL", name="openscience", tags=["grandfathered"])
    explanation = quality_explanation(source, Quality.SAMPLE, [])
    assert explanation is not None
    assert explanation["code"] == FullReviewReason.NO_RECORDED_FULL_REVIEW
    assert explanation["sample"] == {
        "reviewed": 96,
        "selected": 100,
        "population": 1000,
        "receipt": "s3://bucket/sample",
    }
    assert explanation["full"] is None
    quick = quality_explanation(source, Quality.QUICK, [])
    assert quick is not None and quick["sample"] is None
    assert quality_explanation(source, Quality.FULL, []) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", "org/other"),
        ("dataset_revision", "d" * 40),
        ("converter_version", "2"),
        ("task_specs_sha256", "e" * 64),
        ("retained_tasks", 999),
    ],
)
def test_full_review_on_another_input_names_only_proven_differences(field, value):
    record = full_record()
    record["receipt"]["panel"][field] = value
    before = deepcopy(record)
    explanation = quality_explanation(declaration(), Quality.SAMPLE, [record])
    assert explanation is not None
    assert explanation["code"] == FullReviewReason.OTHER_INPUT
    assert explanation["full"]["different_fields"] == [field]
    assert explanation["full"]["receipt"] == record["receipt"]["artifact"]
    assert record == before


@pytest.mark.parametrize(
    "reviewed,code",
    [(0, FullReviewReason.INCOMPLETE), (800, FullReviewReason.INCOMPLETE), (1000, FullReviewReason.NOT_FINALIZED)],
)
def test_current_input_full_review_reports_recorded_progress_not_running_job(reviewed, code):
    record = full_record()
    record["summary"]["reviewed"] = reviewed
    old = full_record()
    old["receipt"]["panel"]["dataset_revision"] = "d" * 40
    explanation = quality_explanation(declaration(), Quality.SAMPLE, [old, record])
    assert explanation is not None
    assert explanation["code"] == code
    assert explanation["full"]["reviewed"] == reviewed
    assert explanation["full"]["population"] == 1000


@pytest.mark.parametrize("side", ["current", "recorded"])
def test_missing_identity_cannot_prove_a_repin(side):
    source, record = declaration(), full_record()
    if side == "current":
        del source["actor_census_input"]["task_specs_sha256"]
    else:
        del record["receipt"]["panel"]["task_specs_sha256"]
    explanation = quality_explanation(source, Quality.SAMPLE, [record])
    assert explanation is not None and explanation["code"] == FullReviewReason.INPUT_UNVERIFIED


def test_archived_full_binding_preserves_its_input_and_receipt():
    source = declaration()
    source["full_content_result_binding"] = {
        "source_id": source["id"],
        "input": {
            "dataset_id": "org/data",
            "dataset_revision": "d" * 40,
            "converter_version": "1",
            "task_specs_sha256": "e" * 64,
            "measured_tasks": 999,
        },
        "review_receipt": {"uri": "s3://bucket/archived-full", "sha256": "f" * 64},
        "result": {"reviewed": 999},
    }
    explanation = quality_explanation(source, Quality.SAMPLE, [])
    assert explanation is not None
    assert explanation["code"] == FullReviewReason.OTHER_INPUT
    assert explanation["full"]["input"]["dataset_revision"] == "d" * 40
    assert explanation["full"]["receipt"] == source["full_content_result_binding"]["review_receipt"]
