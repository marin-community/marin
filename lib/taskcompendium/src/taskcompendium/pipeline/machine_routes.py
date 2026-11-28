# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry verified machine-routing evidence into exported task specifications.

Controls record, for every backend each machine role tried, the outcome it saw.
That evidence is the task's routing contract downstream: a deployment may select
a backend for a role only when the task's evidence does not exclude it, so a
backend whose control failed is never reselected at training, evaluation or
curation time. Every exported machine task carries evidence — its own passing
trial's, or, when the source passed on a sample, the aggregated evidence of the
sampled tasks. A task without the label has no verification evidence at all.
"""

import json
from collections.abc import Iterable, Mapping
from typing import Any

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import CheckStatus, MachineRole, ValidationVerdict

ROUTE_VERDICTS_LABEL = "machine.verdicts"
"""The JSON task label holding each machine role's verification verdicts by backend."""

type RouteEvidence = dict[str, dict[str, str]]
"""Per machine role: each backend mapped to the control outcome that role saw on it."""

EXCLUDING_VERDICT_STATUSES = frozenset({CheckStatus.FAIL, CheckStatus.UNSUPPORTED})
"""Verdict outcomes that stop downstream deployments from selecting the backend."""

_VERDICT_SEVERITY = {
    CheckStatus.FAIL: 0,
    CheckStatus.UNSUPPORTED: 1,
    CheckStatus.INFRA_ERROR: 2,
    CheckStatus.DEFECT: 3,
    CheckStatus.PASS: 4,
    CheckStatus.SKIPPED: 5,
}


def route_evidence(verdicts: Iterable[ValidationVerdict]) -> RouteEvidence:
    """Serialize verdicts as role, then backend; the ladder's final outcome wins per pair."""
    evidence: RouteEvidence = {}
    for verdict in verdicts:
        evidence.setdefault(verdict.role.value, {})[verdict.backend.value] = verdict.status.value
    return evidence


def merge_route_evidence(*evidences: RouteEvidence | None) -> RouteEvidence:
    """Aggregate evidence from several tasks, keeping the most severe outcome on disagreement."""
    merged: RouteEvidence = {}
    for evidence in evidences:
        for role, backends in (evidence or {}).items():
            target = merged.setdefault(role, {})
            for backend, status in backends.items():
                current = target.get(backend)
                if current is None or _VERDICT_SEVERITY[CheckStatus(status)] < _VERDICT_SEVERITY[CheckStatus(current)]:
                    target[backend] = status
    return merged


def attach_route_evidence(row: dict[str, Any], evidence: RouteEvidence) -> dict[str, Any]:
    """Retain the original task identity while attaching verified routing evidence."""
    if not evidence:
        return row
    return {
        **row,
        "labels": {**dict(row.get("labels") or ()), ROUTE_VERDICTS_LABEL: json.dumps(evidence, separators=(",", ":"))},
    }


def export_route_evidence(admitted: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Merge routing evidence into the task's own labels after admission checks its identity."""
    labels = dict(row.get("labels") or ())
    evidence = labels.pop(ROUTE_VERDICTS_LABEL, None)
    if evidence is None:
        return admitted
    task = TaskSpec.model_validate_json(admitted["task_json"])
    return {
        **admitted,
        "task_json": (
            task.model_copy(update={"labels": {**task.labels, ROUTE_VERDICTS_LABEL: evidence}}).model_dump_json()
        ),
    }


def task_route_evidence(task: TaskSpec) -> RouteEvidence:
    """Read the task's routing evidence; an absent or empty label means no verification evidence."""
    recorded = task.labels.get(ROUTE_VERDICTS_LABEL)
    if recorded is None:
        return {}
    evidence = json.loads(recorded)
    shaped = isinstance(evidence, dict) and all(
        isinstance(role, str) and isinstance(backends, dict) for role, backends in evidence.items()
    )
    if not shaped:
        raise ValueError(f"Malformed {ROUTE_VERDICTS_LABEL} label: {recorded!r}")
    return evidence


def excluded_backends(evidence: Mapping[str, Mapping[str, str]], role: MachineRole) -> frozenset[str]:
    """The role's backends whose recorded control outcome excludes selection."""
    return frozenset(
        backend
        for backend, status in evidence.get(role.value, {}).items()
        if status in {status.value for status in EXCLUDING_VERDICT_STATUSES}
    )
