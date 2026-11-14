# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind archived phase deadlines without rewriting production Trial results."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.oracle_filter import PriorPhaseFailureReceipt, ResourceCalibrationReceipt

from experiments.post_training.task_curation.archive_evidence import verified_archive_members


def require_prior_phase_failure(evidence: PriorPhaseFailureReceipt, result: dict) -> None:
    """Verify the original exception, archived config, actual Trial and declared phase."""
    if evidence.result_sha256 != canonical_sha256(result):
        raise ValueError("Earlier phase failure belongs to another production Trial result")
    member = PurePosixPath(evidence.archive_member)
    if member.is_absolute() or ".." in member.parts or member.name != "exception.txt":
        raise ValueError("Earlier phase exception requires an exact relative archive member")
    if evidence.exception.uri != f"{evidence.archive.uri}#{member}":
        raise ValueError("Earlier phase exception reference differs from its immutable archive member")
    trial_uri = urlparse(result["trial_uri"])
    actual_path = PurePosixPath(unquote(trial_uri.path)) / "exception.txt"
    if trial_uri.scheme != "file" or PurePosixPath(evidence.archive_root) / member != actual_path:
        raise ValueError("Earlier exception archive belongs to another physical Trial")
    config_member = str(member.with_name("config.json"))
    selected = verified_archive_members(evidence.archive, {str(member), config_member})
    if json.loads(selected[config_member]) != result["config"]:
        raise ValueError("Earlier phase archive config differs from the actual production Trial")
    exception = selected[str(member)]
    if hashlib.sha256(exception).hexdigest() != evidence.exception.sha256:
        raise ValueError("Earlier phase exception bytes changed")
    phase = evidence.phase
    kind = "AgentTimeoutError" if phase == "agent" else "VerifierTimeoutError"
    label = "Agent execution" if phase == "agent" else "Verifier execution"
    text = exception.decode()
    if (
        "/harbor/trial/trial.py" not in text
        or "in _await_phase" not in text
        or "raise timeout_error from None" not in text
        or not text.rstrip().endswith(
            f"harbor.trial.errors.{kind}: {label} timed out after {float(evidence.timeout_sec)} seconds"
        )
        or result["config"]["resolved_timeouts"][phase] != evidence.timeout_sec
    ):
        raise ValueError("Archive does not prove the reported original phase deadline")


def bind_prior_phase_failures(task_map: Path, failures: dict[str, PriorPhaseFailureReceipt], output: Path) -> None:
    """Write a new task-map version retaining exact raw results and observed deadlines."""
    if output.exists():
        raise FileExistsError(output)
    mappings = json.loads(task_map.read_text())
    matched = set()
    for mapping in mappings.values():
        calibration = mapping["calibration"]
        for attempt in calibration["attempts"]:
            evidence = failures.get(attempt["result_sha256"])
            if evidence is None:
                continue
            require_prior_phase_failure(evidence, attempt["result"])
            existing = attempt.get("prior_phase_failure")
            if existing is not None and existing != evidence.model_dump(mode="json"):
                raise ValueError("Task map already binds different earlier phase evidence")
            attempt["prior_phase_failure"] = evidence.model_dump(mode="json")
            matched.add(attempt["result_sha256"])
        ResourceCalibrationReceipt.model_validate(calibration)
    if matched != failures.keys():
        raise ValueError("Earlier phase evidence does not cover its exact persisted Trial partition")
    output.write_text(json.dumps(mappings, indent=2) + "\n")
