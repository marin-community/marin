# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The retained grading receipt contract: the legacy shape reads, new writes stay strict."""

import pytest
from pydantic import ValidationError

from experiments.post_training.task_curation.retained_grading import (
    RetainedGradingLegacyReceipt,
    RetainedGradingReceipt,
    parse_retained_grading,
)

# The retained-release receipts the ruling names; the -r3 chain supersedes most of them,
# so the registry families hold only the sources their current revisions still pin.
RULING_SOURCES = (
    "TaskTrove:DCAgent__exp_rpt_curriculum-easy",
    "TaskTrove:DCAgent__exp_rpt_curriculum-medium-v2",
    "TaskTrove:DCAgent__exp_rpt_e2egit-large",
    "TaskTrove:DCAgent__exp_rpt_e2egit-v2",
    "TaskTrove:DCAgent__exp_rpt_multifile-v3",
    "TaskTrove:DCAgent__exp_rpt_stack-pytest-v2",
)


def legacy_grading_record() -> dict:
    """The exact older retained grading shape: the frozen population, not a normalized one."""
    return {
        "source_id": "TaskTrove:DCAgent__exp_rpt_curriculum-easy",
        "input": {
            "dataset_id": "open-athena/dcagent-curriculum-easy-taskcompendium",
            "dataset_revision": "b" * 40,
            "converter_version": "source-script-v1-legacy0000000",
            "converter_revision": "c" * 40,
            "measured_tasks": 3,
            "task_specs_sha256": "d" * 64,
            "evidence_url": "s3://marin-us-east-02a/marin/rl-data/atlas/actor-census/legacy/2026.11.12.1/.artifact.json",
        },
        "population_path": "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/legacy/2026.11.12.1",
        "scanned_task_specs_sha256": "a" * 64,
        "evidence": {
            "assessed_tasks": 3,
            "required_counts": {"shared": 1},
            "declared_counts": {"shared": 1},
            "status_counts": {"shared": 1},
            "reason_counts": {},
            "evidence_url": "s3://marin-us-east-02a/marin/rl-data/atlas/grading-census/legacy/receipt.json",
        },
        "task_evidence_uri": "s3://marin-us-east-02a/marin/rl-data/atlas/grading-census/legacy/tasks.parquet",
    }


def test_legacy_shape_is_accepted_for_reads_only() -> None:
    """The older shape parses through the union but never through the current write contract."""
    parsed = parse_retained_grading(legacy_grading_record())
    assert isinstance(parsed, RetainedGradingLegacyReceipt)
    assert parsed.population_path.endswith("2026.11.12.1")
    # The write contract refuses the legacy shape: new receipts require normalized_path.
    with pytest.raises(ValidationError, match="normalized_path"):
        RetainedGradingReceipt.model_validate(legacy_grading_record())


def test_new_writes_require_normalized_path() -> None:
    """A retained grading write without a real normalized population path is refused."""
    record = legacy_grading_record()
    with pytest.raises(ValidationError, match="normalized_path"):
        RetainedGradingReceipt.model_validate(record)
    record["normalized_path"] = ""
    with pytest.raises(ValidationError):
        RetainedGradingReceipt.model_validate(record)
    record["normalized_path"] = "s3://marin-us-east-02a/marin/rl-data/atlas/normalized/legacy/2026.11.19.5"
    receipt = RetainedGradingReceipt.model_validate(record)
    assert receipt.normalized_path.startswith("s3://")


def test_unknown_shape_still_fails() -> None:
    """A grading record matching neither contract is refused."""
    record = legacy_grading_record()
    del record["population_path"]
    with pytest.raises(ValidationError):
        parse_retained_grading(record)


@pytest.mark.data_integration
def test_every_named_retained_receipt_parses() -> None:
    """The retained receipts for the ruling's sources parse at whatever shape the registry pins."""
    from experiments.post_training.task_curation.datasets.oracle_admitted_releases import (  # noqa: PLC0415
        oracle_admitted_release_receipts,
    )
    from experiments.post_training.task_curation.datasets.retained_releases import (  # noqa: PLC0415
        retained_release_receipts,
    )

    retained = retained_release_receipts()
    oracle = oracle_admitted_release_receipts()
    parsed = 0
    for source_id in RULING_SOURCES:
        receipt = retained.get(source_id)
        if receipt is not None:
            assert receipt.grading.source_id == source_id
            assert receipt.grading.evidence.assessed_tasks == receipt.grading.input.measured_tasks
            parsed += 1
        admitted = oracle.get(source_id)
        if admitted is not None and admitted.archived_original is not None:
            assert admitted.archived_original.retained.grading.source_id == source_id
            parsed += 1
    assert parsed > 0, "the ruling's named sources must still parse from their receipt families"
