# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import repair_projection_training_sample as repair  # noqa: E402
from glm_semantic_labels import SampleDocument  # noqa: E402


def document(index: int) -> SampleDocument:
    return SampleDocument(index, f"sha-{index}", "source", "standard", index, f"text-{index}")


def valid_config(documents: list[SampleDocument]) -> dict:
    return {
        "training_run_id": repair.LABEL_RUN_ID,
        "pilot_run_id": repair.HIERARCHY_RUN_ID,
        "excluded_evaluation_run_id": repair.EXCLUDED_EVALUATION_RUN_ID,
        "document_count": repair.EXPECTED_DOCUMENTS,
        "purpose": "semantic_projection_training",
        "variant": {"name": repair.HIERARCHY_VARIANT},
        "training_identity_sha256": repair.identity_digest(documents),
    }


def test_repair_input_validation_rejects_identity_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repair, "EXPECTED_DOCUMENTS", 2)
    documents = [document(0), document(1)]
    config = valid_config(documents)
    config["training_identity_sha256"] = "different"

    with pytest.raises(ValueError, match="identity differs"):
        repair.validate_repair_inputs(config, documents)
