"""Store compact diagnostics for native agentic training attempts."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, AttemptReceipt, Phase
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from rolloutengine.models import SETUP_COMPLETED_METRIC, RolloutOperation, TaskRolloutSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.runtime.environment import environment_identity

from skyrl_train.rollouts.training_record import TrainingRollout

_ERROR_CHARS = 1024
_OUTPUT_TAIL_CHARS = 4096


def _diagnostic_tail(value: Any, limit: int) -> str | None:
    return redact_string(value)[-limit:] if isinstance(value, str) and value else None


def attempt_receipt(
    attempt_id: str, spec: TaskRolloutSpec, result: TrainingRollout, *, task_digest: str | None = None
) -> AttemptReceipt | None:
    """Project the result's setup and verifier evidence without retaining the trajectory."""
    task = spec.task
    if "agentic" not in task.tags:
        return None
    image, setup = environment_identity(task.environment_requirements)
    grade, failure = result.grade, result.failure
    diagnostics = failure.diagnostics if failure is not None else {}
    operation = diagnostics.get("operation")
    evidence = grade.diagnostics
    setup_exit = diagnostics.get("setup_exit_code")
    setup_failed = "setup_reason" in diagnostics
    completed = result.metrics.get(SETUP_COMPLETED_METRIC)
    setup_completed = bool(completed) if setup is not None and completed is not None else None
    if setup_failed:
        phase = Phase.TASK_SETUP
        setup_completed = False
    elif grade.status == Outcome.GRADED or grade.failure is not None or operation == RolloutOperation.GRADE:
        phase = Phase.VERIFIER
    elif operation == RolloutOperation.START:
        phase = Phase.ENVIRONMENT
    else:
        phase = Phase.AGENT
    graded = grade.status == Outcome.GRADED
    error = grade.error
    if diagnostics.get("setup_reason") == "exited" and isinstance(setup_exit, int) and 0 < setup_exit < 128:
        error = f"Task setup command exited {setup_exit}"
    return AttemptReceipt(
        attempt_id=attempt_id,
        task_id=task.id,
        task_digest=task_digest if task_digest is not None else canonical_sha256(task.model_dump(mode="json")),
        outcome=grade.status,
        grading_failure=grade.failure,
        exception_type=failure.exception_type if failure is not None else None,
        error=None if graded else _diagnostic_tail(error, _ERROR_CHARS),
        phase=phase,
        docker_image=image,
        setup_sha256=setup,
        setup_completed=setup_completed,
        setup_stdout=None if graded else _diagnostic_tail(diagnostics.get("setup_stdout"), _OUTPUT_TAIL_CHARS),
        setup_stderr=None if graded else _diagnostic_tail(diagnostics.get("setup_stderr"), _OUTPUT_TAIL_CHARS),
        verifier_stdout=None if graded else _diagnostic_tail(evidence.get("stdout"), _OUTPUT_TAIL_CHARS),
        verifier_stderr=None if graded else _diagnostic_tail(evidence.get("stderr"), _OUTPUT_TAIL_CHARS),
    )


def write_attempt_receipts(export_path: str, receipts: Sequence[AttemptReceipt]) -> None:
    """Write one deterministic object per generated batch with health receipts."""
    if not receipts:
        return
    payload = "".join(
        json.dumps(asdict(row), sort_keys=True) + "\n" for row in sorted(receipts, key=lambda row: row.attempt_id)
    )
    directory = StoragePath(export_path) / ATTEMPT_RECEIPTS_DIRECTORY
    directory.mkdirs()
    (directory / (hashlib.sha256(payload.encode()).hexdigest() + ".jsonl")).write_text(payload)
