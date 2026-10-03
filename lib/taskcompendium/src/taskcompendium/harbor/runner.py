# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from math import isfinite
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from verifyit.spec import PredictedActionSpec

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.grading_contract import resolve_verifier
from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_environment_config,
)
from taskcompendium.models import VerifierSpec
from taskcompendium.submission import chat_request

GRADE_RESULT = TypeAdapter(GradeResult)


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    request_timeout: float = Field(gt=0, allow_inf_nan=False)
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    parallel_tool_calls: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    reasoning_effort: str | None = Field(default=None, min_length=1, pattern=r"\S")


def validate_launch_parallel_tool_calls(specification: VerifierSpec, parallel_tool_calls: bool | None) -> None:
    verifier = resolve_verifier(specification)
    if parallel_tool_calls is False and isinstance(verifier, PredictedActionSpec) and len(verifier.expected_calls) > 1:
        raise ValueError("Launch disables parallel calls required by the task")


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    launch: ChatLaunch,
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Run a lowered task and return Harbor's trial result."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_environment_config(specification, environment_config)
    convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    request = chat_request(specification, convention)
    if launch.max_tokens is not None:
        request["max_tokens"] = launch.max_tokens
    if launch.reasoning_effort is not None:
        request["reasoning_effort"] = launch.reasoning_effort
    if launch.temperature is not None:
        request["temperature"] = launch.temperature
    if launch.parallel_tool_calls is not None:
        if "parallel_tool_calls" in request and request["parallel_tool_calls"] != launch.parallel_tool_calls:
            raise ValueError("Launch parallel-tool policy conflicts with the submission convention")
        request["parallel_tool_calls"] = launch.parallel_tool_calls
    validate_launch_parallel_tool_calls(specification.verifier, request.get("parallel_tool_calls"))
    agent = {
        "import_path": "taskcompendium.harbor.adapter:ChatAgent",
        "model_name": launch.model,
        "kwargs": {
            **launch.model_dump(
                exclude={"model", "temperature", "parallel_tool_calls", "max_tokens", "reasoning_effort"}
            ),
            "request": request,
        },
    }
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"},
            "agent": agent,
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
        }
    )
    trial = await Trial.create(config)
    return await trial.run()


def read_trial_outcome(result: TrialResult) -> GradeResult:
    """Read the TaskCompendium outcome from a full Harbor result without filesystem access.

    Harbor's slimmed aggregation result drops stdout. Use the full returned or
    persisted TrialResult when the distinction between wrong and invalid matters.
    """
    if result.verifier_result is not None and result.verifier_result.stdout is not None:
        outcome = GRADE_RESULT.validate_json(result.verifier_result.stdout, strict=True)
        if outcome.status == Outcome.INFRA_ERROR or outcome.reward is None or not isfinite(outcome.reward):
            raise ValueError("Scored Harbor result contains an unavailable TaskCompendium outcome")
        if outcome.status == Outcome.SUBMISSION_FAILURE and outcome.reward != 0.0:
            raise ValueError("Submission failure cannot carry a positive reward")
        rewards = result.verifier_result.rewards
        if rewards is None or rewards.get("reward") != outcome.reward:
            raise ValueError("TaskCompendium outcome disagrees with Harbor rewards")
        return outcome
    if result.exception_info is not None:
        return GradeResult(Outcome.INFRA_ERROR, None, result.exception_info.exception_message)
    raise ValueError("Harbor result contains neither a TaskCompendium outcome nor an exception")
