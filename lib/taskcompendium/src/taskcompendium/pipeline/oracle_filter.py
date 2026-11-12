# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-population Harbor oracle receipts bound to the converted task contents."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec, decode_json_value
from taskcompendium.pipeline.fingerprints import recipe_code_identity
from taskcompendium.pipeline.models import SourceRecipe
from taskcompendium.pipeline.resource_calibration import (
    INITIAL_REQUIREMENTS,
    FailureSignal,
    ResourceDimension,
    ResourceUsage,
    failure_signal,
    next_requirements,
)
from taskcompendium.pipeline.sources import source_files_identity


def has_oracle_solution(task: TaskSpec) -> bool:
    resources = task.resources
    return any(
        resource.path.startswith("solution/")
        for resource in (*resources.all, *resources.worker, *resources.verifier, *resources.oracle)
    )


def oracle_recipe_identity(recipe: SourceRecipe) -> str:
    return canonical_sha256(
        {
            "name": recipe.name,
            "version": recipe.version,
            "source": source_files_identity(recipe.source),
            "code": recipe_code_identity(recipe),
            "grader": recipe.grader_environment.model_dump(mode="json") if recipe.grader_environment else None,
            "oracle_solutions": recipe.oracle_solutions,
        }
    )


class OracleTrialReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    result: dict[str, Any]
    verdict: dict[str, Any] | None
    oracle_exit_code: int | None
    result_sha256: str
    execution_requirements: ExecutionRequirements
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    verifier_resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    failure_log: str = ""

    @model_validator(mode="after")
    def validate_result_hash(self):
        if canonical_sha256(self.result) != self.result_sha256:
            raise ValueError("Oracle trial result hash differs from its receipt")
        config = self.result.get("config", {})
        environment = config.get("environment", {})
        if any(environment.get(name) == "ignore" for name in ("cpu_enforcement_policy", "memory_enforcement_policy")):
            raise ValueError("Oracle calibration cannot ignore task-declared resource limits")
        if any(environment.get(f"override_{name}") is not None for name in ("cpus", "memory_mb", "storage_mb")):
            raise ValueError("Oracle calibration cannot override task-declared resources")
        for phase in ("agent", "verifier"):
            if config.get(phase, {}).get("override_timeout_sec") is not None:
                raise ValueError("Oracle calibration cannot override task-declared timeouts")
        if any(
            config.get(name, 1) not in (None, 1)
            for name in ("timeout_multiplier", "agent_timeout_multiplier", "verifier_timeout_multiplier")
        ):
            raise ValueError("Oracle calibration requires unscaled task timeouts")
        resolved = config.get("resolved_timeouts")
        if resolved is None or (
            resolved["agent"] != self.execution_requirements.agent_timeout_sec
            or resolved["verifier"] != self.execution_requirements.verifier_timeout_sec
        ):
            raise ValueError("Oracle trial deadlines differ from declared execution requirements")
        return self

    @property
    def failure_signal(self) -> FailureSignal:
        if (self.verdict or {}).get("status") == "infra_error":
            return FailureSignal.INFRASTRUCTURE
        usage = self.resource_usage.model_copy(
            update={"oom_kills": self.resource_usage.oom_kills + self.verifier_resource_usage.oom_kills}
        )
        return failure_signal(self.result.get("exception_info"), self.failure_log, usage)

    def passed(self, task_checksum: str) -> bool:
        result = self.result
        verdict = self.verdict or {}
        reward = verdict.get("reward")
        return (
            result.get("agent_info", {}).get("name") == "oracle"
            and result.get("config", {}).get("agent", {}).get("name") == "oracle"
            and result.get("config", {}).get("environment", {}).get("type") in {"local", "daytona"}
            and result.get("task_checksum") == task_checksum
            and result.get("exception_info") is None
            and result.get("finished_at") is not None
            and result.get("completion") is not None
            and self.oracle_exit_code == 0
            and verdict.get("status") == "scored"
            and not isinstance(reward, bool)
            and reward == 1
            and ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") == 1
        )


class VerifierResourceProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: GradingEnvironmentMode
    fixed: dict[ResourceDimension, float] = Field(default_factory=dict)


def next_trial_requirements(
    trial: OracleTrialReceipt, preserved: dict[ResourceDimension, float], verifier: VerifierResourceProfile
) -> ExecutionRequirements | None:
    """Stop at a separate grader's declared limit without increasing the actor's allocation."""
    signal = trial.failure_signal
    verifier_failed = (
        not trial.resource_usage.oom_kills
        and signal is not FailureSignal.AGENT_TIMEOUT
        and (
            trial.verifier_resource_usage.oom_kills
            or signal is FailureSignal.VERIFIER_TIMEOUT
            or (trial.result.get("verifier_execution") or {}).get("started_at") is not None
        )
    )
    fixed = frozenset(preserved)
    if verifier.mode is GradingEnvironmentMode.SEPARATE and verifier_failed:
        fixed |= frozenset(verifier.fixed)
    return next_requirements(trial.execution_requirements, signal, trial.resource_usage, fixed)


class ResourceCalibrationReceipt(BaseModel):
    """Every production attempt, including calibration-only solutions kept out of admission proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    declared_requirements: ExecutionRequirements | None
    preserved_requirements: dict[ResourceDimension, float] = Field(default_factory=dict)
    verifier: VerifierResourceProfile
    oracle_kind: Literal["original", "calibration_only"]
    attempts: tuple[OracleTrialReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attempts(self):
        expected = self.declared_requirements or ExecutionRequirements.model_validate(
            {
                **INITIAL_REQUIREMENTS.model_dump(),
                **self.preserved_requirements,
            }
        )
        infrastructure_retries = 0
        for index, trial in enumerate(self.attempts):
            if trial.execution_requirements != expected:
                raise ValueError("Calibration changed an unfailed resource or a task-declared limit")
            checksum = trial.result["task_checksum"]
            if trial.passed(checksum) and index != len(self.attempts) - 1:
                raise ValueError("Calibration continued after a passing oracle")
            if trial.failure_signal is FailureSignal.INFRASTRUCTURE:
                infrastructure_retries += 1
                if infrastructure_retries > 3:
                    raise ValueError("Calibration exceeded the infrastructure retry budget")
                if index == len(self.attempts) - 1 and infrastructure_retries < 3:
                    raise ValueError("Calibration infrastructure retry budget is incomplete")
                continue
            infrastructure_retries = 0
            if self.declared_requirements is None and index != len(self.attempts) - 1:
                expected = next_trial_requirements(trial, self.preserved_requirements, self.verifier)
                if expected is None:
                    raise ValueError("Calibration continued after a non-resource failure or exhausted cap")
        final = self.attempts[-1]
        if self.declared_requirements is None and not final.passed(final.result["task_checksum"]):
            if next_trial_requirements(final, self.preserved_requirements, self.verifier) is not None:
                raise ValueError("Resource calibration stopped before a pass or the applicable cap")
        return self

    @property
    def passed(self) -> bool:
        trial = self.attempts[-1]
        return trial.passed(trial.result["task_checksum"])

    @property
    def execution_requirements(self) -> ExecutionRequirements:
        return self.attempts[-1].execution_requirements


class NoOracleLivenessReceipt(BaseModel):
    """A separate empty-submission verifier run, bound to the admitted task and runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    task_sha256: str
    runtime_sha256: str | None
    submission_kind: Literal["empty", "candidate"]
    outcome: str
    exit_code: int | None
    reward: float | None = Field(allow_inf_nan=False)
    tests_executed: int
    base: str = ""
    snapshot: str = ""
    detail: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.submission_kind == "empty"
            and self.outcome == "negative_verdict"
            and self.exit_code == 0
            and self.reward is not None
            and self.reward < 1
            and self.tests_executed > 0
        )


class OracleTaskReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_sha256: str
    runtime_sha256: str | None = None
    harbor_task_checksum: str
    has_oracle: bool
    trials: tuple[OracleTrialReceipt, ...]
    calibration: ResourceCalibrationReceipt
    no_oracle_liveness: NoOracleLivenessReceipt | None = None
    exclusion_reason: str | None = None

    @property
    def passed(self) -> bool:
        if not self.calibration.passed:
            return False
        if self.has_oracle:
            return self.calibration.oracle_kind == "original" and any(
                trial.passed(self.harbor_task_checksum) for trial in self.trials
            )
        liveness = self.no_oracle_liveness
        return (
            self.calibration.oracle_kind == "calibration_only"
            and liveness is not None
            and liveness.passed
            and liveness.task_sha256 == self.task_sha256
            and liveness.runtime_sha256 == self.runtime_sha256
        )


class OracleFilterReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[2] = 2
    recipe_identity: str | None
    input_cohort_sha256: str
    input_rows: int = Field(ge=0)
    conversion_rejections: int = Field(ge=0)
    tasks: dict[str, OracleTaskReceipt]

    @model_validator(mode="after")
    def validate_population(self):
        if len(self.tasks) + self.conversion_rejections != self.input_rows:
            raise ValueError("Oracle filter receipt does not cover the full input population")
        for task_id, entry in self.tasks.items():
            if entry.no_oracle_liveness is not None and entry.no_oracle_liveness.task_id != task_id:
                raise ValueError(f"No-oracle liveness receipt belongs to another task: {task_id}")
            if not entry.passed and not entry.exclusion_reason:
                raise ValueError(f"Oracle exclusion has no reason: {task_id}")
            if not entry.has_oracle and entry.trials:
                raise ValueError(f"Non-oracle task carries oracle trials: {task_id}")
            if entry.has_oracle != (entry.calibration.oracle_kind == "original"):
                raise ValueError(f"Calibration-only oracle cannot supply admission evidence: {task_id}")
            if entry.has_oracle and entry.trials != (entry.calibration.attempts[-1],):
                raise ValueError(f"Oracle admission trial differs from final calibration: {task_id}")
        return self

    def admits(self, task: TaskSpec, runtime: dict | None = None) -> bool:
        if task.id not in self.tasks:
            raise ValueError(f"Oracle filter receipt omits task {task.id}")
        entry = self.tasks[task.id]
        if entry.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError(f"Task changed after oracle filtering: {task.id}")
        if runtime is not None and entry.runtime_sha256 != canonical_sha256(runtime):
            raise ValueError(f"Runtime changed after oracle filtering: {task.id}")
        if entry.has_oracle != has_oracle_solution(task):
            raise ValueError(f"Oracle presence changed after filtering: {task.id}")
        if task.execution_requirements != entry.calibration.execution_requirements:
            raise ValueError(f"Task execution requirements differ from calibration: {task.id}")
        return entry.passed

    @property
    def uniform_requirements(self) -> ExecutionRequirements | None:
        """Return a source-wide shape only when every converted task passed at that shape."""
        entries = list(self.tasks.values())
        if not entries or self.conversion_rejections or any(not entry.passed for entry in entries):
            return None
        shape = entries[0].calibration.execution_requirements
        return shape if all(entry.calibration.execution_requirements == shape for entry in entries) else None


def load_oracle_filter(path: str) -> OracleFilterReceipt:
    return OracleFilterReceipt.model_validate(decode_json_value(StoragePath(path).read_text()))


def require_oracle_filter(task: TaskSpec, receipt: OracleFilterReceipt | None, runtime: dict | None = None) -> bool:
    """Require calibration and original-oracle or separate empty-submission liveness evidence."""
    if task.execution_requirements is None:
        raise ValueError(f"SAMPLE/FULL admission requires explicit execution requirements: {task.id}")
    if receipt is None:
        raise ValueError(f"SAMPLE/FULL admission requires a full-population Harbor oracle filter receipt: {task.id}")
    return receipt.admits(task, runtime)
