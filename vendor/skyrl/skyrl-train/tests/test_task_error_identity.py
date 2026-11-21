# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native and engine timeout producers persist the same task retry contract."""

import json
from dataclasses import replace

import pytest
from harbor.models.trial.result import ExceptionInfo
from harbor.trial.errors import (
    AgentSetupTimeoutError,
    AgentTimeoutError,
    EnvironmentStartTimeoutError,
    TrialTimeoutError,
    VerifierTimeoutError,
)
from omegaconf import OmegaConf
from rigging.task_errors import ErrorCategory, ErrorIdentity, RetryDisposition
from rolloutengine.models import GradedRolloutInterrupted, Rollout, RolloutOperation
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import _interruption_identity
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.task_errors import TIMED_OUT_DIAGNOSTIC
from skyrl_train.trajectory_runners.trajectory_processing import normalize_trajectory_batches
from skyrl_train.trajectory_runners.types import BatchFields, TaskRequestBatch
from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results
from taskcompendium.grading_result import GradeResult, Outcome

from tests.cpu.task_specs import lowered_task, task_input, text_task


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "native_type, operation, disposition",
    [
        (TrialTimeoutError, RolloutOperation.ATTEMPT, RetryDisposition.TRANSIENT),
        (EnvironmentStartTimeoutError, RolloutOperation.START, RetryDisposition.TRANSIENT),
        (AgentSetupTimeoutError, RolloutOperation.PREPARE, RetryDisposition.TRANSIENT),
        (AgentTimeoutError, RolloutOperation.MODEL, RetryDisposition.UNKNOWN),
        (VerifierTimeoutError, RolloutOperation.GRADE, RetryDisposition.UNKNOWN),
    ],
)
def test_timeout_identity_matches_across_native_engine_and_saved_records(native_type, operation, disposition, wrapped):
    native_error = native_type("deadline expired")
    if wrapped:
        native_error.__cause__ = TimeoutError("phase deadline")
    native = ExceptionInfo.from_exception(native_error)
    interruption = GradedRolloutInterrupted(
        Rollout("timeout", (), "error"), operation, GradeResult(Outcome.UNAVAILABLE, None)
    )
    interruption.__cause__ = TimeoutError("deadline expired")
    engine = _interruption_identity(interruption)
    saved = ExceptionInfo(
        exception_type=native_type.__name__,
        exception_message="deadline expired",
        exception_traceback="",
        occurred_at=native.occurred_at,
    )
    for failure in (native, engine, saved):
        persisted = ErrorIdentity.model_validate_json(failure.model_dump_json())
        assert persisted.exception_type == native_type.__name__
        assert persisted.category == native.category
        assert persisted.retry_disposition is disposition


class EvidenceTokenizer:
    eos_token_id = 99

    def decode(self, token_ids):
        return " ".join(str(token) for token in token_ids)


def test_failure_fields_survive_projection_mixed_batches_and_safe_eval_dump(tmp_path):
    failure = ErrorIdentity(
        exception_type="TaskMachineError",
        category=ErrorCategory.INFRASTRUCTURE,
        cause_type="DaytonaConnectionError",
        operation="start",
        provider_status_code=503,
        retry_disposition=RetryDisposition.TRANSIENT,
        diagnostics={TIMED_OUT_DIAGNOSTIC: False, "private_request": "fixture capability"},
    )
    failed = TrainingRollout(
        task_id="fixture",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(),
        loss_mask=(),
        logprobs=None,
        grade=GradeResult(Outcome.INFRA_ERROR, None),
        stop_reason="error",
        failure=failure,
    )
    succeeded = replace(
        failed, failure=None, grade=GradeResult(Outcome.GRADED, 1.0), response_token_ids=(2,), loss_mask=(1,)
    )
    config = OmegaConf.create({"sampling_params": {"logprobs": None}, "apply_overlong_filtering": False})
    tokenizer = EvidenceTokenizer()
    projection = WholeTaskProjection(config, tokenizer)
    request = TaskRequestBatch(rows=(task_input(lowered_task(text_task())),))
    batches = [projection.project([rollout], request) for rollout in (succeeded, failed)]
    combined = normalize_trajectory_batches(
        batches, fields=BatchFields.from_groups([BatchFields.from_batch(batch) for batch in batches])
    )
    assert combined["error_identities"] == [None, failure]
    dump_per_dataset_eval_results(
        str(tmp_path),
        tokenizer,
        combined,
        ["fixture", "fixture"],
        ["fixture", "fixture"],
        {},
        ["success", "failure"],
        attempt_id="attempt",
        step=0,
        ordinals=[0, 1],
    )
    raw = (tmp_path / "fixture.jsonl").read_text()
    succeeded_row, failed_row = [json.loads(line) for line in raw.splitlines()]
    assert succeeded_row["failure"] is None
    stored = ErrorIdentity.model_validate(failed_row["failure"])
    assert stored.cause_type == "DaytonaConnectionError"
    assert stored.operation == "start"
    assert stored.provider_status_code == 503
    assert stored.category is ErrorCategory.INFRASTRUCTURE
    assert stored.retry_disposition is RetryDisposition.TRANSIENT
    assert failed_row["verification"]["diagnostics"] == {
        TIMED_OUT_DIAGNOSTIC: False,
    }
    assert "fixture capability" not in raw
