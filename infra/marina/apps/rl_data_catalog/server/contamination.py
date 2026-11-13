# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive the displayed lexical-decontamination state from catalog evidence."""

from enum import StrEnum
from typing import Any

from finestore.rl.curation import DeconReceiptStatus, LexicalContaminationEvidence

from .quality import intended_use


class Contamination(StrEnum):
    NO_ISSUES = "NO ISSUES"
    REMEDIATED = "REMEDIATED"
    MEASURED_EVAL = "MEASURED, EVAL"
    REMEDIATION_REQUIRED = "REMEDIATION REQUIRED"
    NOT_MEASURED = "NOT MEASURED"


def contamination(declaration: dict[str, Any]) -> Contamination:
    """Classify recorded overlap without treating missing evidence as clean."""
    evidence = declaration.get("lexical_contamination")
    if evidence is None:
        return Contamination.NOT_MEASURED
    measured = LexicalContaminationEvidence.model_validate(evidence)
    if measured.contaminated_tasks == 0:
        current_pin = (declaration["dataset_id"], declaration["dataset_revision"])
        if (measured.input_dataset_id, measured.input_dataset_revision) == current_pin:
            return Contamination.NO_ISSUES
        if measured.admission_blocker(*current_pin) is None:
            return Contamination.NO_ISSUES
        return Contamination.NOT_MEASURED
    if intended_use(declaration) == "eval":
        return Contamination.MEASURED_EVAL
    receipt = measured.receipt
    if (
        receipt is not None
        and receipt.status is DeconReceiptStatus.REMEDIATED
        and measured.admission_blocker(declaration["dataset_id"], declaration["dataset_revision"]) is None
        and receipt.release is not None
        and receipt.release.private
        and receipt.release.format == "taskcompendium"
    ):
        return Contamination.REMEDIATED
    return Contamination.REMEDIATION_REQUIRED
