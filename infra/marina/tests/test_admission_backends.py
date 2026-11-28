# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission counts production backends only; dev-only verdicts never satisfy it."""

import pytest

from experiments.post_training.task_curation.tests.admission_fixtures import (
    with_declaration_admission_census,
)
from infra.marina.apps.rl_data_catalog.server.admission import (
    backend_verification_evidence,
    has_runnable_production_support,
    require_available_lexical_clearance,
)
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import (
    contamination_declarations,
)
from lib.taskcompendium.tests.decontamination_fixtures import (
    decontamination_evidence,
)


def test_verify_evidence_survives_without_becoming_an_admission_gate():
    verdicts = {"iris": "verify", "local": "verify", "docker": "verify"}
    assert has_runnable_production_support(verdicts)
    assert backend_verification_evidence(verdicts) == ["iris", "local"]
    assert backend_verification_evidence({"custom": "verify"}, {"custom": "production"}) == ["custom"]


REV = "b" * 40


def _lexical_declaration(use, evidence=None, *, dataset_id="fixture"):
    """An admitted-census declaration with an explicit use and optional lexical evidence."""

    source = contamination_declarations()[0][1]
    base = source.model_copy(
        update={
            "dataset_id": dataset_id,
            "dataset_revision": REV,
            "intended_use": use,
            "lexical_contamination": evidence,
            "within_source_dedup": source.within_source_dedup.model_copy(
                update={"source_dataset_id": dataset_id, "source_revision": REV}
            ),
        }
    )
    return with_declaration_admission_census(base)


def _eval_receipt_evidence():
    """A benchmark-measured lexical receipt bound to the fixture pin."""

    evidence = decontamination_evidence("fixture", REV, 3)
    payload = evidence.model_dump(mode="json")
    payload["receipt"]["use"] = "eval"
    payload["receipt"]["status"] = "measured (benchmark, not remediated)"
    payload["receipt"]["release"] = None
    return evidence.model_validate(payload)


def test_new_train_admission_requires_the_private_release_receipt():
    with pytest.raises(ValueError, match="Decontamination"):
        require_available_lexical_clearance(_lexical_declaration("train"))


def test_new_train_admission_passes_with_private_release_evidence():
    evidence = decontamination_evidence("open-athena/fixture", REV, 3)
    require_available_lexical_clearance(_lexical_declaration("train", evidence, dataset_id="open-athena/fixture"))


def test_new_evaluation_admission_requires_a_measured_receipt():
    with pytest.raises(ValueError, match="measured lexical receipt"):
        require_available_lexical_clearance(_lexical_declaration("eval"))


def test_new_evaluation_admission_passes_with_a_benchmark_measured_receipt():
    require_available_lexical_clearance(_lexical_declaration("eval", _eval_receipt_evidence()))


def test_null_intended_use_is_refused_before_any_lexical_check():
    with pytest.raises(ValueError, match="explicit intended_use"):
        require_available_lexical_clearance(_lexical_declaration(None))
