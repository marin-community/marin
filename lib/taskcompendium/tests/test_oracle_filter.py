# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle admission rejects missing execution evidence and stale converted states."""

import pytest
from pydantic import ValidationError

from taskcompendium.models import ResourceGroups, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import OracleFilterReceipt, require_oracle_filter

from .oracle_filter_fixtures import oracle_receipt
from .pipeline_stages import SOURCE_FILES
from .test_source_processing import apple_rows, convert_script_graded


def task():
    return convert_script_graded(
        RawRow(
            "oracle",
            Source(dataset=SOURCE_FILES.dataset, revision="1", row="0", importer_revision="1"),
            apple_rows(1)[0],
        ),
        ConversionContext({}, None),
    )


def test_oracle_filter_requires_scored_success_instead_of_reward_only_receipts():
    current = task()
    with pytest.raises(ValueError, match="full-population Harbor oracle filter"):
        require_oracle_filter(current, None)
    assert require_oracle_filter(current, oracle_receipt([current]))
    assert not require_oracle_filter(current, oracle_receipt([current], failed=frozenset({current.id})))
    document = oracle_receipt([current]).model_dump(mode="json")
    document["tasks"][current.id]["trials"][0]["verdict"] = {"status": "infra_error", "reward": 1}
    document["tasks"][current.id]["calibration"]["attempts"][0]["verdict"] = {"status": "infra_error", "reward": 1}
    document["tasks"][current.id]["exclusion_reason"] = "unscored verifier"
    assert not require_oracle_filter(current, OracleFilterReceipt.model_validate(document))


def test_changed_task_or_incomplete_population_cannot_reuse_an_oracle_receipt():
    current = task()
    receipt = oracle_receipt([current])
    changed = current.model_copy(update={"labels": {"changed": "after trial"}})
    with pytest.raises(ValueError, match="changed after oracle filtering"):
        require_oracle_filter(changed, receipt)
    document = receipt.model_dump(mode="json")
    document["input_rows"] += 1
    with pytest.raises(ValidationError, match="full input population"):
        OracleFilterReceipt.model_validate(document)


def test_calibration_only_oracle_cannot_supply_admission_evidence():
    current = task().model_copy(update={"resources": ResourceGroups()})
    with pytest.raises(ValueError, match="full-population Harbor oracle filter"):
        require_oracle_filter(current, None)
    receipt = oracle_receipt([current])
    assert receipt.tasks[current.id].trials == ()
    assert require_oracle_filter(current, receipt)
    document = receipt.model_dump(mode="json")
    document["tasks"][current.id]["calibration"]["oracle_kind"] = "original"
    with pytest.raises(ValidationError, match=r"calibration-only|Calibration-only"):
        OracleFilterReceipt.model_validate(document)
