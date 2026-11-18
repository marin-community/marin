# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve approved archived grades without rewriting failed teardown Results."""

import hashlib
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.oracle_filter import (
    OracleTrialReceipt,
    PostGradeTeardownEvidence,
    ResourceCalibrationReceipt,
)

from experiments.post_training.task_curation.archive_evidence import verified_archive_members

ORACLE_AUTHORITY_MESSAGE = "7cf3d8d1fd2b9bb4c9998a4b758746d4"
NOP_AUTHORITY_MESSAGE = "d01cf03119d70430139c23c1c1227f4f"
NOP_MACHINE_CLEANUP_AUTHORITY_MESSAGE = "97aee5e15daa6e03344581964ec1416105abf2a409bf052793894c245b390b87"
"""sha256 of the 2026-11-18 coordinator ruling text approving post-grade machine cleanup
bindings (nop-machine-cleanup-ruling.md)."""


def require_post_grade_teardown(evidence: PostGradeTeardownEvidence, trial: OracleTrialReceipt) -> None:
    """Verify the exact recorded grade, original command and completed scoring phase."""
    result = trial.result
    if evidence.result_sha256 != trial.result_sha256:
        raise ValueError("Recorded teardown grade belongs to another production Result")
    OracleTrialReceipt.model_validate(
        {**trial.model_dump(mode="json"), "post_grade_teardown": evidence.model_dump(mode="json")}
    )
    oracle = evidence.kind == "oracle_verdict_before_environment_stop"
    actual = urlparse(result["trial_uri"])
    if actual.scheme != "file":
        raise ValueError("Recorded teardown grade requires an actual production Trial URI")
    path = PurePosixPath(unquote(actual.path))
    if oracle:
        root = path.relative_to("/tmp")
        expected_path = str(path / "result.json")
    else:
        # The NoP archive roots the trial at its control directory, which the recorded
        # result member names (negative_control_from_archive binds trial.path to it).
        root = PurePosixPath(urlparse(trial.path).fragment).parent
        expected_path = evidence.result_artifact.uri
    if trial.path != expected_path:
        raise ValueError("Recorded teardown grade belongs to another physical Trial")
    references = {
        "result.json": evidence.result_artifact,
        "verifier/verdict.json": evidence.verdict_artifact,
        "exception.txt": evidence.teardown_exception_artifact,
    }
    if evidence.reward_artifact is not None:
        references["verifier/reward.txt"] = evidence.reward_artifact
    if oracle:
        assert evidence.oracle_exit_artifact is not None
        assert evidence.reward_artifact is not None
        references["agent/exit-code.txt"] = evidence.oracle_exit_artifact
    for name, reference in references.items():
        if reference.uri != f"{evidence.archive.uri}#{root / name}":
            raise ValueError("Recorded teardown evidence differs from its original archive members")
    selected = verified_archive_members(evidence.archive, {str(root / name) for name in (*references, "config.json")})
    for name, reference in references.items():
        if hashlib.sha256(selected[str(root / name)]).hexdigest() != reference.sha256:
            raise ValueError("Recorded teardown evidence bytes changed")
    verdict = json.loads(selected[str(root / "verifier/verdict.json")])
    if (
        json.loads(selected[str(root / "result.json")]) != result
        or json.loads(selected[str(root / "config.json")]) != result["config"]
        or verdict != trial.verdict
        or (
            evidence.reward_artifact is not None
            and float(selected[str(root / "verifier/reward.txt")]) != verdict["reward"]
        )
        or (oracle and int(selected[str(root / "agent/exit-code.txt")]) != trial.oracle_exit_code)
    ):
        raise ValueError("Archived oracle grade or original exit differs from the recorded Trial")
    phases = [
        (result.get("agent_execution") or {}).get("started_at"),
        (result.get("agent_execution") or {}).get("finished_at"),
        (result.get("verifier") or {}).get("started_at"),
        (result.get("verifier") or {}).get("finished_at"),
        result.get("finished_at"),
    ]
    if any(value is None for value in phases):
        raise ValueError("Recorded teardown grade requires completed actor and verifier phases")
    times = [datetime.fromisoformat(value) for value in phases]
    if any(value.tzinfo is None for value in times) or times != sorted(times):
        raise ValueError("Recorded scoring phases are not ordered before the final Result")
    # Historical ExceptionInfo times are naive. Use the recorded UTC execution
    # intervals and archived cleanup stack instead of assuming its timezone.
    cleanup = selected[str(root / "exception.txt")].decode()
    environment_stop = "in _stop_environment" in cleanup and (
        "EnvironmentStopError: Environment teardown failed" in cleanup
    )
    machine_cleanup = "Machine cleanup failed" in cleanup
    if not (environment_stop or machine_cleanup):
        raise ValueError("Archive does not prove a subsequent environment teardown failure")
    decision = StoragePath(evidence.decision.uri).read_bytes()
    if hashlib.sha256(decision).hexdigest() != evidence.decision.sha256:
        raise ValueError("Recorded-grade decision bytes changed")
    approval = json.loads(decision)
    cases = approval.get("cases", {})
    case = cases.get(trial.result_sha256, {})
    if machine_cleanup:
        if (
            approval.get("authority_message") != NOP_MACHINE_CLEANUP_AUTHORITY_MESSAGE
            or case.get("task_id") != result["task_name"]
            or case.get("reward") != verdict["reward"]
        ):
            raise ValueError("Recorded grade is outside the approved machine-cleanup partition")
        return
    if (
        approval.get("authority_message") != (ORACLE_AUTHORITY_MESSAGE if oracle else NOP_AUTHORITY_MESSAGE)
        or len(cases) != (4 if oracle else 2)
        or case.get("task_id") != result["task_name"]
        or case.get("reward") != verdict["reward"]
        or not result["task_name"].startswith("scale-swe-verified-")
    ):
        raise ValueError("Recorded grade is outside the approved physical Trial partition")


def bind_post_grade_teardowns(task_map: Path, evidence: dict[str, PostGradeTeardownEvidence], output: Path) -> None:
    """Write a new metadata map preserving raw Results, verdicts and every original attempt."""
    if output.exists():
        raise FileExistsError(output)
    mappings = json.loads(task_map.read_text())
    matched = set()
    for mapping in mappings.values():
        calibration = mapping["calibration"]
        for attempt in calibration["attempts"]:
            proof = evidence.get(attempt["result_sha256"])
            if proof is None:
                continue
            require_post_grade_teardown(proof, OracleTrialReceipt.model_validate(attempt))
            existing = attempt.get("post_grade_teardown")
            if existing is not None and existing != proof.model_dump(mode="json"):
                raise ValueError("Task map already binds another recorded teardown grade")
            attempt["post_grade_teardown"] = proof.model_dump(mode="json")
            matched.add(attempt["result_sha256"])
        ResourceCalibrationReceipt.model_validate(calibration)
    if matched != evidence.keys():
        raise ValueError("Recorded teardown grades do not match the exact original Trial partition")
    output.write_text(json.dumps(mappings, indent=2) + "\n")
