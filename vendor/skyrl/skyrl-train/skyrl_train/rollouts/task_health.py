"""Store compact diagnostics for native task rollout attempts."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, EVALUATION_RECEIPTS_DIRECTORY, AttemptReceipt, Phase
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from rolloutengine.models import SETUP_COMPLETED_METRIC, RolloutFailure, RolloutOperation, TaskRolloutSpec
from shellbox.machine import ExitReason
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import task_key
from taskcompendium.runtime.environment import environment_identity

from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.trajectory_runners.types import TrainingPhase

_ERROR_CHARS = 1024
_OUTPUT_TAIL_CHARS = 4096


def _diagnostic_tail(value: Any, limit: int) -> str | None:
    return redact_string(value)[-limit:] if isinstance(value, str) and value else None


def attempt_receipt(
    attempt_id: str,
    spec: TaskRolloutSpec,
    result: TrainingRollout | BaseException,
    *,
    rollout_id: str,
    task_digest: str,
    source_id: str | None = None,
) -> AttemptReceipt:
    """Project the result's setup and verifier evidence without retaining the trajectory."""
    task = spec.task
    image, setup = environment_identity(task.environment_requirements)
    if isinstance(result, BaseException):
        grade = GradeResult(Outcome.INFRA_ERROR, None, str(result))
        failure = RolloutFailure(type(result).__name__, {})
        completed = None
    else:
        grade, failure = result.grade, result.failure
        completed = result.metrics.get(SETUP_COMPLETED_METRIC)
    diagnostics = failure.diagnostics if failure is not None else {}
    operation = diagnostics.get("operation")
    evidence = grade.diagnostics
    setup_exit = diagnostics.get("setup_exit_code") if diagnostics.get("setup_reason") == ExitReason.EXITED else None
    setup_failed = "setup_reason" in diagnostics
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
    return AttemptReceipt(
        attempt_id=attempt_id,
        rollout_id=rollout_id,
        source_row=task_key(source_id, task),
        task_id=task.id,
        task_digest=task_digest,
        outcome=Outcome.INFRA_ERROR if setup_failed else grade.status,
        grading_failure=grade.failure,
        exception_type=failure.exception_type if failure is not None else None,
        error=None if graded else _diagnostic_tail(grade.error, _ERROR_CHARS),
        phase=phase,
        docker_image=image,
        setup_sha256=setup,
        setup_completed=setup_completed,
        setup_exit_code=setup_exit,
        setup_stdout=None if graded else _diagnostic_tail(diagnostics.get("setup_stdout"), _OUTPUT_TAIL_CHARS),
        setup_stderr=None if graded else _diagnostic_tail(diagnostics.get("setup_stderr"), _OUTPUT_TAIL_CHARS),
        verifier_stdout=None if graded else _diagnostic_tail(evidence.get("stdout"), _OUTPUT_TAIL_CHARS),
        verifier_stderr=None if graded else _diagnostic_tail(evidence.get("stderr"), _OUTPUT_TAIL_CHARS),
    )


def write_attempt_receipts(export_path: str, receipts: Sequence[AttemptReceipt], *, phase: TrainingPhase) -> None:
    """Write one deterministic object per generated batch with health receipts."""
    if not receipts:
        return
    payload = "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
        for row in sorted(receipts, key=lambda row: row.rollout_id or row.attempt_id)
    )
    directory = StoragePath(export_path) / ATTEMPT_RECEIPTS_DIRECTORY
    if phase == "eval":
        directory = directory / EVALUATION_RECEIPTS_DIRECTORY
    directory.mkdirs()
    (directory / (hashlib.sha256(payload.encode()).hexdigest() + ".jsonl")).write_text(payload)
