# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Engine timeout producers and training exports persist task retry identity."""

import json
from dataclasses import replace

import pytest
from omegaconf import OmegaConf
from rigging.task_errors import ErrorCategory, ErrorIdentity, RetryDisposition
from rolloutengine.models import GradedRolloutInterrupted, Rollout, RolloutOperation
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import _interruption_identity
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.task_errors import TIMED_OUT_DIAGNOSTIC
from skyrl_train.trajectory_runners.trajectory_processing import normalize_trajectory_batches
from skyrl_train.trajectory_runners.types import BatchFields, TaskRequestBatch
from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results
from taskcompendium.grading_result import GradeResult, Outcome

from tests.cpu.task_specs import lowered_task, task_input, text_task


@pytest.mark.parametrize("cause", [None, TimeoutError, PermissionError])
@pytest.mark.parametrize(
    "exception_type, operation, disposition, category",
    [
        ("TrialTimeoutError", RolloutOperation.ATTEMPT, RetryDisposition.TRANSIENT, ErrorCategory.INFRASTRUCTURE),
        (
            "EnvironmentStartTimeoutError",
            RolloutOperation.START,
            RetryDisposition.TRANSIENT,
            ErrorCategory.INFRASTRUCTURE,
        ),
        ("AgentSetupTimeoutError", RolloutOperation.PREPARE, RetryDisposition.TRANSIENT, ErrorCategory.INFRASTRUCTURE),
        ("AgentTimeoutError", RolloutOperation.MODEL, RetryDisposition.UNKNOWN, ErrorCategory.AGENT),
        ("VerifierTimeoutError", RolloutOperation.GRADE, RetryDisposition.UNKNOWN, ErrorCategory.UNKNOWN),
    ],
)
def test_engine_timeout_identity_survives_serialization(exception_type, operation, disposition, category, cause):
    interruption = GradedRolloutInterrupted(
        Rollout("timeout", (), "error"), operation, GradeResult(Outcome.UNAVAILABLE, None)
    )
    interruption.__cause__ = TimeoutError("deadline expired")
    if cause is not None:
        interruption.__cause__.__cause__ = cause("phase deadline")
    engine = _interruption_identity(interruption)
    persisted = ErrorIdentity.model_validate_json(engine.model_dump_json())
    assert persisted.exception_type == exception_type
    assert persisted.operation == operation.value
    assert persisted.category is category
    expected = RetryDisposition.PERMANENT if cause is PermissionError else disposition
    assert persisted.retry_disposition is expected


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
    metadata = [
        TrajectoryMetadata("success", "taskcompendium", "fixture"),
        TrajectoryMetadata("failure", "taskcompendium", "fixture"),
    ]
    dump_per_dataset_eval_results(
        str(tmp_path),
        tokenizer,
        combined,
        metadata,
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
