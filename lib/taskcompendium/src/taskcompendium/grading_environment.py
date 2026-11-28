# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assess whether a task's declared grader preserves the state its contract reads."""

import base64
import tomllib
from collections import Counter
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field

from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    ArtifactKind,
    Grader,
    GraderMachine,
    NoGrader,
    ScriptGrader,
    TaskSpec,
    VerifyitGrader,
)

VERIFIER_DOCUMENT_PATHS = ("task.toml", "verifier.toml", "taskcompendium-verifier.toml")
NATIVE_CONFIG_DOCUMENT = "task.toml"
"""The verifier document whose declared environment names a native task's grading boundary."""


class GradingEnvironmentMode(StrEnum):
    SHARED = "shared"
    SEPARATE = "separate"


class GradingEnvironmentStatus(StrEnum):
    MATCHED = "matched"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


class GradingEnvironmentAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    required: GradingEnvironmentMode
    declared: GradingEnvironmentMode | None
    reason: str

    @computed_field
    @property
    def status(self) -> GradingEnvironmentStatus:
        if self.declared is None:
            return GradingEnvironmentStatus.UNAVAILABLE
        if self.required is GradingEnvironmentMode.SHARED and self.declared is GradingEnvironmentMode.SEPARATE:
            return GradingEnvironmentStatus.MISMATCH
        return GradingEnvironmentStatus.MATCHED


def _verifier_documents(task: TaskSpec) -> dict[str, dict]:
    """Decode the declared verifier configuration documents a grading boundary may read."""
    documents = {}
    for resource in task.resources.verifier:
        if resource.path not in VERIFIER_DOCUMENT_PATHS:
            continue
        documents[resource.path] = tomllib.loads(base64.b64decode(resource.source.content_base64).decode())
    return documents


def _grader_machine(grader: Grader) -> GraderMachine | None:
    """The fresh machine a grader runs in, or ``None`` when it grades in the agent's machine."""
    if isinstance(grader, ScriptGrader | VerifyitGrader) and isinstance(grader.placement, GraderMachine):
        return grader.placement
    return None


def assess_grading_environment(task: TaskSpec) -> GradingEnvironmentAssessment:
    """Derive the required state boundary from the task, independently of catalog declarations.

    A SHARED result establishes the container boundary: the grader reads
    agent-mutable workspace state, so grading runs where the agent ran.
    """
    documents = _verifier_documents(task)
    native = isinstance(task.grader, NoGrader) and NATIVE_CONFIG_DOCUMENT in documents
    declared = GradingEnvironmentMode.SEPARATE
    if native:
        verifier = documents[NATIVE_CONFIG_DOCUMENT].get("verifier", {})
        declared = GradingEnvironmentMode(
            verifier.get("environment_mode") or ("separate" if verifier.get("environment") is not None else "shared")
        )
    elif isinstance(task.grader, NoGrader):
        return GradingEnvironmentAssessment(
            required=GradingEnvironmentMode.SEPARATE, declared=None, reason="No executable grader is declared"
        )
    elif isinstance(task.grader, ScriptGrader | VerifyitGrader) and isinstance(task.grader.placement, AgentMachine):
        # A grader placed in the agent's machine declares shared grading, whatever its kind.
        declared = GradingEnvironmentMode.SHARED

    grader = task.grader
    machine = _grader_machine(grader)
    if isinstance(grader, ScriptGrader) and machine is not None and (machine.collect or machine.artifacts):
        return GradingEnvironmentAssessment(
            required=GradingEnvironmentMode.SEPARATE,
            declared=declared,
            reason="The script grader consumes declared collected inputs in its separate verifier environment",
        )
    artifacts = machine.artifacts if machine is not None else ()
    repository = (
        task.answer_type is AnswerType.WORKSPACE_STATE
        or "git_repository" in task.environment_requirements.capabilities
        or (isinstance(grader, ScriptGrader) and any(artifact.kind is ArtifactKind.DIRECTORY for artifact in artifacts))
    )
    if repository:
        return GradingEnvironmentAssessment(
            required=GradingEnvironmentMode.SHARED,
            declared=declared,
            reason="Repository or workspace-state grading reads state the agent can mutate",
        )
    mode = (
        grader.mode
        if isinstance(grader, VerifyitGrader)
        else next(
            (
                document["mode"]
                for name, document in documents.items()
                if name != NATIVE_CONFIG_DOCUMENT and "mode" in document
            ),
            None,
        )
    )
    if mode == "stdio":
        return GradingEnvironmentAssessment(
            required=GradingEnvironmentMode.SEPARATE,
            declared=declared,
            reason="Stdio grading reads the submitted program and its stdout in the declared verifier environment",
        )
    if native or (isinstance(grader, ScriptGrader) and grader.answer_path is None and task.output_paths):
        return GradingEnvironmentAssessment(
            required=GradingEnvironmentMode.SHARED,
            declared=declared,
            reason="The opaque workspace verifier does not declare a file-only grading boundary",
        )
    return GradingEnvironmentAssessment(
        required=GradingEnvironmentMode.SEPARATE,
        declared=declared,
        reason="The typed grader reads the declared submitted answer or captured files in its own environment",
    )


def count_grading_environment(counts: Counter[str], assessment: Mapping[str, str | None]) -> None:
    """Count assessed rows and each required mode, declared mode, status and reason."""
    counts["assessed_tasks"] += 1
    for field in ("required", "declared", "status", "reason"):
        counts[f"{field}:{assessment[field] or 'unavailable'}"] += 1


def grading_environment_census(counts: Counter[str]) -> dict:
    return {
        "assessed_tasks": counts["assessed_tasks"],
        **{
            f"{field}_counts": {
                key.removeprefix(field + ":"): count for key, count in counts.items() if key.startswith(field + ":")
            }
            for field in ("required", "declared", "status", "reason")
        },
    }
