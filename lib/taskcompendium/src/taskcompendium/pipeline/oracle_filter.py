# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-population Harbor oracle receipts bound to the converted task contents."""

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.grading_environment import GradingEnvironmentMode, assess_grading_environment
from taskcompendium.harbor.adapter import actor_machine_required
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding, network_policy_qualified
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, ScriptGrader, TaskSpec, VerifyitGrader, decode_json_value
from taskcompendium.pipeline.fingerprints import recipe_code_identity
from taskcompendium.pipeline.legacy_roles import LegacyRoleQualificationReceipt
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
from taskcompendium.runtime.oracle import oracle_actor_required
from taskcompendium.runtime.resources import has_oracle_solution


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


class InfrastructureRetryContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    parent_result_sha256: str
    decision_ref: str
    backend_fix_revision: str


class SourcePytestEnvironment(BaseModel):
    """Versions frozen with the protected runner before the agent starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pytest_version: str = Field(pattern=r"^\d+\.\d+")
    pytest_metadata_version: str | None


class NativeVerifierGrade(BaseModel):
    """Production-parsed native reward bytes bound to the actual Trial result."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    rewards: dict[str, Any] = Field(min_length=1)
    artifact_path: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_checksum: str


class PriorPhaseFailureReceipt(BaseModel):
    """An archived source deadline observed before a later cleanup failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    phase: Literal["agent", "verifier"]
    timeout_sec: float = Field(gt=0, allow_inf_nan=False)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive: ContentEvidenceReference
    exception: ContentEvidenceReference
    archive_member: str
    archive_root: str


class NativeOracleSuccessPolicy(BaseModel):
    """Source-approved interpretation of unchanged production native reward maps."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    kind: Literal["all_named_maximum"] = "all_named_maximum"
    maximum: float
    required_reward_keys: tuple[str, ...]
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str | None
    source_task_checksum: str
    input_cohort_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumer: ContentEvidenceReference
    source_census: ContentEvidenceReference
    decision: ContentEvidenceReference

    def passes(self, trial: "OracleTrialReceipt", task_checksum: str) -> bool:
        return (
            trial.scored
            and trial.oracle_exit_code == 0
            and trial.result.get("task_checksum") == task_checksum
            and self.rewards_pass(trial)
        )

    def rewards_pass(self, trial: "OracleTrialReceipt") -> bool:
        rewards = (trial.result.get("verifier_result") or {}).get("rewards") or {}
        return (
            trial.verdict is None
            and trial.native_grade is not None
            and trial.graded
            and bool(rewards)
            and set(self.required_reward_keys) <= rewards.keys()
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value == self.maximum
                for value in rewards.values()
            )
        )


def pre_oracle_host_unavailable(result: dict[str, Any], verdict: dict | None, oracle_exit_code: int | None) -> bool:
    """Recognize frozen host failures that occurred before the original command."""
    config = result.get("config", {})
    if (
        verdict is not None
        or oracle_exit_code is not None
        or (result.get("verifier_result") or {}).get("rewards")
        or (result.get("agent_info") or {}).get("name") != "oracle"
        or config.get("agent", {}).get("name") != "oracle"
    ):
        return False
    exception = result.get("exception_info") or {}
    kind = exception.get("exception_type")
    message = exception.get("exception_message", "")
    traceback = exception.get("exception_traceback", "")
    backend = config.get("environment", {}).get("type")
    if backend == "daytona":
        return (
            kind == "RuntimeError"
            and message == "Machine is closed"
            and "/harbor/agents/oracle.py" in traceback
            and "counters = await resource_counters(environment)" in traceback
            and "in resource_counters" in traceback
            and "await environment.exec_control(" in traceback
            and "/shellbox/backends/daytona/machine.py" in traceback
        )
    return (
        backend == "local"
        and result.get("agent_execution") is None
        and kind == "UnsupportedMachineSpec"
        and message.startswith("LOCAL disk enforcement host probe failed:")
        and "failed to setup loop device for " in message
        and "/disk.img" in message
        and "/shellbox/backends/local/disk.py" in traceback
        and "self._probe()" in traceback
        and "in _prepare" in traceback
    )


class OracleTrialReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    result: dict[str, Any]
    verdict: dict[str, Any] | None
    oracle_exit_code: int | None
    result_sha256: str
    execution_requirements: ExecutionRequirements
    verifier_execution_requirements: ExecutionRequirements | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    verifier_resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    failure_log: str = ""
    infrastructure_retry: InfrastructureRetryContext | None = None
    source_environment: SourcePytestEnvironment | None = None
    network_policy: TaskNetworkPolicyReceipt | None = None
    native_grade: NativeVerifierGrade | None = None
    prior_phase_failure: PriorPhaseFailureReceipt | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_result_hash(self):
        if canonical_sha256(self.result) != self.result_sha256:
            raise ValueError("Oracle trial result hash differs from its receipt")
        if self.native_grade is not None and (
            self.native_grade.result_sha256 != self.result_sha256
            or self.native_grade.task_checksum != self.result.get("task_checksum")
        ):
            raise ValueError("Native grade belongs to another production Trial")
        if self.prior_phase_failure is not None and self.prior_phase_failure.result_sha256 != self.result_sha256:
            raise ValueError("Earlier phase failure belongs to another production Trial result")
        if self.prior_phase_failure is not None and (
            (self.result.get("agent_info") or {}).get("name") != "oracle"
            or self.result.get("config", {}).get("agent", {}).get("name") != "oracle"
        ):
            raise ValueError("Earlier source phase evidence requires an original oracle Trial")
        config = self.result.get("config", {})
        verifier_allocation = self.verifier_execution_requirements
        configured_verifier = config.get("task", {}).get("verifier_execution_requirements")
        recorded_verifier = verifier_allocation.model_dump(mode="json") if verifier_allocation is not None else None
        if configured_verifier != recorded_verifier:
            raise ValueError("Verifier allocation differs from the actual production Trial config")
        if verifier_allocation is not None:
            if any(
                getattr(verifier_allocation, name) != getattr(self.execution_requirements, name)
                for name in ("agent_timeout_sec", "verifier_timeout_sec")
            ):
                raise ValueError("Actor and verifier allocations disagree on phase deadlines")
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
        if self.prior_phase_failure is not None:
            return (
                FailureSignal.AGENT_TIMEOUT
                if self.prior_phase_failure.phase == "agent"
                else FailureSignal.VERIFIER_TIMEOUT
            )
        usage = self.resource_usage.model_copy(
            update={"oom_kills": self.resource_usage.oom_kills + self.verifier_resource_usage.oom_kills}
        )
        signal = failure_signal(self.result.get("exception_info"), self.failure_log, usage)
        if signal in {
            FailureSignal.OOM,
            FailureSignal.DISK_FULL,
            FailureSignal.AGENT_TIMEOUT,
            FailureSignal.VERIFIER_TIMEOUT,
        }:
            return signal
        if pre_oracle_host_unavailable(self.result, self.verdict, self.oracle_exit_code):
            return FailureSignal.INFRASTRUCTURE
        if (self.verdict or {}).get("status") == "infra_error":
            return FailureSignal.INFRASTRUCTURE
        if self.infrastructure_retry is not None and not self.scored and signal is FailureSignal.NON_RESOURCE:
            return FailureSignal.INFRASTRUCTURE
        return signal

    @property
    def graded(self) -> bool:
        verdict = self.verdict or {}
        reward = verdict.get("reward")
        actual = (self.result.get("verifier_result") or {}).get("rewards") or {}
        if self.verdict is not None:
            return (
                verdict.get("status") == "scored"
                and isinstance(reward, (int, float))
                and not isinstance(reward, bool)
                and math.isfinite(reward)
                and actual.get("reward") == reward
            )
        native = self.native_grade
        return (
            native is not None
            and canonical_sha256(actual) == canonical_sha256(native.rewards)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in native.rewards.values()
            )
        )

    @property
    def scored(self) -> bool:
        result = self.result
        return (
            (result.get("agent_info") or {}).get("name") == "oracle"
            and result.get("config", {}).get("agent", {}).get("name") == "oracle"
            and result.get("config", {}).get("environment", {}).get("type") in {"local", "daytona"}
            and result.get("exception_info") is None
            and self.prior_phase_failure is None
            and result.get("finished_at") is not None
            and result.get("completion") is not None
            and self.oracle_exit_code is not None
            and self.graded
        )

    def passed(self, task_checksum: str) -> bool:
        return (
            self.scored
            and self.result.get("task_checksum") == task_checksum
            and self.oracle_exit_code == 0
            and ((self.result.get("verifier_result") or {}).get("rewards") or {}).get("reward") == 1
        )


def trial_network_qualified(task: TaskSpec, runtime: dict | None, trial: OracleTrialReceipt) -> bool:
    """Require observed policies for source-declared machine network permissions."""
    binding = (
        TaskSpecRuntimeBinding(
            task_sha256=canonical_sha256(task.model_dump(mode="json")),
            runtime_sha256=canonical_sha256(runtime),
            runtime=runtime,
        )
        if runtime is not None
        else None
    )
    return network_policy_qualified(binding, trial.network_policy)


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


def initial_verifier_requirements(
    actor: ExecutionRequirements, verifier: VerifierResourceProfile
) -> ExecutionRequirements:
    """Resolve the separate grader from canonical defaults and only its own declarations."""
    if verifier.mode is GradingEnvironmentMode.SHARED:
        return actor
    return ExecutionRequirements.model_validate(
        {
            **INITIAL_REQUIREMENTS.model_dump(),
            **verifier.fixed,
            "agent_timeout_sec": actor.agent_timeout_sec,
            "verifier_timeout_sec": actor.verifier_timeout_sec,
        }
    )


def next_trial_allocations(
    trial: OracleTrialReceipt, preserved: dict[ResourceDimension, float], verifier: VerifierResourceProfile
) -> tuple[ExecutionRequirements, ExecutionRequirements | None] | None:
    """Grow only the failing role, retaining one canonical pair of phase deadlines."""
    actor = trial.execution_requirements
    grader = trial.verifier_execution_requirements
    if grader is None:
        updated = next_trial_requirements(trial, preserved, verifier)
        return None if updated is None else (updated, None)
    signal = trial.failure_signal
    actorless_oracle = trial.network_policy is not None and not trial.network_policy.actor_required
    verifier_failed = actorless_oracle or (
        not trial.resource_usage.oom_kills
        and signal is not FailureSignal.AGENT_TIMEOUT
        and (
            trial.verifier_resource_usage.oom_kills
            or signal is FailureSignal.VERIFIER_TIMEOUT
            or (trial.result.get("verifier") or trial.result.get("verifier_execution") or {}).get("started_at")
            is not None
        )
    )
    if verifier.mode is GradingEnvironmentMode.SEPARATE and verifier_failed:
        fixed = frozenset(verifier.fixed) | frozenset(
            dimension
            for dimension in preserved
            if dimension in {ResourceDimension.AGENT_TIMEOUT, ResourceDimension.VERIFIER_TIMEOUT}
        )
        usage = trial.resource_usage if actorless_oracle else trial.verifier_resource_usage
        updated = next_requirements(grader, signal, usage, fixed)
        if updated is None:
            return None
        actor = actor.model_copy(
            update={
                "agent_timeout_sec": updated.agent_timeout_sec,
                "verifier_timeout_sec": updated.verifier_timeout_sec,
            }
        )
        return actor, updated
    updated = next_requirements(actor, signal, trial.resource_usage, frozenset(preserved))
    if updated is None:
        return None
    if verifier.mode is GradingEnvironmentMode.SHARED:
        return updated, updated
    return updated, grader.model_copy(
        update={
            "agent_timeout_sec": updated.agent_timeout_sec,
            "verifier_timeout_sec": updated.verifier_timeout_sec,
        }
    )


class ResourceCalibrationReceipt(BaseModel):
    """Every production attempt, including calibration-only solutions kept out of admission proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    declared_requirements: ExecutionRequirements | None
    preserved_requirements: dict[ResourceDimension, float] = Field(default_factory=dict)
    initial_requirements: ExecutionRequirements | None = Field(default=None, exclude_if=lambda value: value is None)
    verifier: VerifierResourceProfile
    oracle_kind: Literal["original", "calibration_only"]
    attempts: tuple[OracleTrialReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attempts(self):
        expected = (
            self.declared_requirements
            or self.initial_requirements
            or ExecutionRequirements.model_validate(
                {
                    **INITIAL_REQUIREMENTS.model_dump(),
                    **self.preserved_requirements,
                }
            )
        )
        if self.initial_requirements is not None:
            if self.attempts[0].infrastructure_retry is None:
                raise ValueError("An approved retry context is required to resume a calibrated shape")
            if self.declared_requirements is not None and self.initial_requirements != self.declared_requirements:
                raise ValueError("A resumed shape cannot change declared execution requirements")
            if any(
                getattr(self.initial_requirements, name) != value for name, value in self.preserved_requirements.items()
            ):
                raise ValueError("A resumed shape cannot change a preserved source limit")
        fixed_actor = self.preserved_requirements
        if self.declared_requirements is not None:
            fixed_actor = {
                **fixed_actor,
                **{ResourceDimension(name): value for name, value in self.declared_requirements.model_dump().items()},
            }
        first = self.attempts[0]
        expected_verifier = None
        if first.verifier_execution_requirements is not None:
            expected_verifier = (
                first.verifier_execution_requirements
                if first.infrastructure_retry is not None
                else initial_verifier_requirements(expected, self.verifier)
            )
        infrastructure_retries = 0
        for index, trial in enumerate(self.attempts):
            prior_failure = trial.prior_phase_failure
            if prior_failure is not None:
                original = (
                    self.declared_requirements.model_dump()
                    if self.declared_requirements is not None
                    else self.preserved_requirements
                )
                if original.get(ResourceDimension(f"{prior_failure.phase}_timeout_sec")) != prior_failure.timeout_sec:
                    raise ValueError("Earlier phase failure must match the original source-declared deadline")
            if trial.execution_requirements != expected:
                raise ValueError("Calibration changed an unfailed resource or a task-declared limit")
            if trial.verifier_execution_requirements != expected_verifier:
                raise ValueError("Calibration changed an unfailed verifier resource or a fixed source limit")
            if expected_verifier is not None and any(
                getattr(expected_verifier, name) != value for name, value in self.verifier.fixed.items()
            ):
                raise ValueError("Calibration changed a fixed verifier declaration")
            checksum = trial.result["task_checksum"]
            if trial.passed(checksum) and index != len(self.attempts) - 1:
                raise ValueError("Calibration continued after a passing oracle")
            if trial.failure_signal is FailureSignal.INFRASTRUCTURE:
                infrastructure_retries += 1
                if infrastructure_retries > 3:
                    raise ValueError("Calibration exceeded the infrastructure retry budget")
                if (
                    index == len(self.attempts) - 1
                    and infrastructure_retries < 3
                    and not pre_oracle_host_unavailable(trial.result, trial.verdict, trial.oracle_exit_code)
                ):
                    raise ValueError("Calibration infrastructure retry budget is incomplete")
                continue
            infrastructure_retries = 0
            if index != len(self.attempts) - 1 and (self.declared_requirements is None or expected_verifier is not None):
                updated = next_trial_allocations(trial, fixed_actor, self.verifier)
                if updated is None:
                    raise ValueError("Calibration continued after a non-resource failure or exhausted cap")
                expected, expected_verifier = updated
        final = self.attempts[-1]
        if (self.declared_requirements is None or expected_verifier is not None) and not final.passed(
            final.result["task_checksum"]
        ):
            if next_trial_allocations(final, fixed_actor, self.verifier) is not None:
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


class BackendFixIdentity(BaseModel):
    """Pinned source bytes executed by an approved infrastructure retry phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files_sha256: dict[str, str] = Field(min_length=1)


class InfrastructureRetryApproval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    classification: Literal["our_infrastructure"]
    parent_result_sha256: str
    failure_signature: str = Field(min_length=1)
    decision_ref: str = Field(min_length=1)


LOCAL_EXECUTION_HOST_FIELDS = frozenset({"cgroup_parent", "disk_root", "image_cache", "registry_authfile"})


class LocalExecutionHostTransition(BaseModel):
    """Bind replacement worker paths without changing the source execution contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    parent_kwargs: dict[str, str | None]
    current_kwargs: dict[str, str | None]
    scope_evidence: tuple[ContentEvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_host_fields(self):
        for fields in (self.parent_kwargs, self.current_kwargs):
            if fields.keys() - LOCAL_EXECUTION_HOST_FIELDS:
                raise ValueError("Only physical LOCAL worker paths may change across hosts")
        if not self.current_kwargs.get("cgroup_parent") or not self.current_kwargs.get("disk_root"):
            raise ValueError("A replacement LOCAL host requires explicit cgroup and disk paths")
        return self


@dataclass(frozen=True)
class RecoveredRuntimeRequirements:
    execution_requirements: ExecutionRequirements
    preserved_requirements: dict[ResourceDimension, float]
    verifier: VerifierResourceProfile


def recovered_source_runtime_requirements(
    parent: ResourceCalibrationReceipt, binding: TaskSpecRuntimeBinding
) -> RecoveredRuntimeRequirements:
    """Recover only explicit original role budgets missed by unqualified old attempts."""
    actor = {ResourceDimension(name): value for name, value in binding.resource_limits("task_machine").items()}
    grader = {ResourceDimension(name): value for name, value in binding.resource_limits("verifier_machine").items()}
    if not actor and not grader:
        raise ValueError("Source runtime recovery requires explicit original role resources")
    for attempt in parent.attempts:
        recorded_grade = (
            attempt.result.get("exception_info") is None
            and attempt.result.get("finished_at") is not None
            and bool((attempt.result.get("verifier_result") or {}).get("rewards"))
        )
        if attempt.scored or recorded_grade:
            raise ValueError("Source runtime recovery cannot replay a scored old attempt")
    for name, value in actor.items():
        if parent.declared_requirements is not None and getattr(parent.declared_requirements, name.value) != value:
            raise ValueError("Source runtime recovery cannot change a declared actor limit")
        if name in parent.preserved_requirements and parent.preserved_requirements[name] != value:
            raise ValueError("Source runtime recovery cannot change a preserved actor declaration")
    for name, value in grader.items():
        if name in parent.verifier.fixed and parent.verifier.fixed[name] != value:
            raise ValueError("Source runtime recovery cannot change a preserved verifier declaration")
    preserved = {**parent.preserved_requirements, **actor}
    verifier = parent.verifier.model_copy(update={"fixed": {**parent.verifier.fixed, **grader}})
    shape = parent.execution_requirements.model_copy(update={name.value: value for name, value in actor.items()})
    return RecoveredRuntimeRequirements(shape, preserved, verifier)


class InfrastructureRetryPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cohort_path: str
    cohort_sha256: str
    parent_map_path: str
    parent_map_sha256: str
    source_parent_map_path: str
    source_parent_map_sha256: str
    backend_fix: BackendFixIdentity
    environment: Literal["local", "daytona"]
    environment_kwargs: dict
    approvals: dict[str, InfrastructureRetryApproval] = Field(min_length=1)
    local_host: LocalExecutionHostTransition | None = Field(default=None, exclude_if=lambda value: value is None)
    source_runtime_resource_recovery: TaskSpecRuntimeBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_recovery_partition(self):
        if self.source_runtime_resource_recovery is not None and len(self.approvals) != 1:
            raise ValueError("A source runtime recovery plan must bind one exact original task")
        return self


def require_infrastructure_retry_parent(
    calibration: ResourceCalibrationReceipt, approval: InfrastructureRetryApproval
) -> None:
    final = calibration.attempts[-1]
    if calibration.oracle_kind != "original":
        raise ValueError("Infrastructure retries require an original oracle with immutable source limits")
    if any(
        (attempt.result.get("agent_info") or {}).get("name") != "oracle"
        or attempt.result.get("config", {}).get("agent", {}).get("name") != "oracle"
        for attempt in calibration.attempts
    ):
        raise ValueError("Infrastructure retries require production oracle attempts")
    recorded_grade = (
        final.result.get("exception_info") is None
        and final.result.get("finished_at") is not None
        and bool((final.result.get("verifier_result") or {}).get("rewards"))
    )
    if final.scored or recorded_grade:
        raise ValueError("Infrastructure retry approval cannot replace a scored source result")
    if any(attempt.prior_phase_failure is not None for attempt in calibration.attempts):
        raise ValueError("Infrastructure retry cannot replay an observed source-declared phase deadline")
    if approval.parent_result_sha256 != final.result_sha256:
        raise ValueError("Infrastructure approval belongs to another parent attempt")


class InfrastructureRetryTransition(BaseModel):
    """Retain the preceding phase and its immutable inputs before applying a backend fix."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_sha256: str
    runtime_sha256: str | None
    source_task_checksum: str
    plan_path: str
    plan_sha256: str
    parent_map_path: str
    parent_map_sha256: str
    parent_mapping_sha256: str
    parent_history_sha256: str
    parent_calibration: ResourceCalibrationReceipt
    backend_fix: BackendFixIdentity
    approval: InfrastructureRetryApproval
    local_host: LocalExecutionHostTransition | None = Field(default=None, exclude_if=lambda value: value is None)
    source_runtime_resource_recovery: TaskSpecRuntimeBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_parent_failure(self):
        require_infrastructure_retry_parent(self.parent_calibration, self.approval)
        recovery = self.source_runtime_resource_recovery
        if recovery is not None:
            if recovery.task_sha256 != self.task_sha256 or recovery.runtime_sha256 != self.runtime_sha256:
                raise ValueError("Source runtime recovery belongs to another frozen task or runtime")
            recovered_source_runtime_requirements(self.parent_calibration, recovery)
        return self


class OracleExclusionReason(StrEnum):
    NEGATIVE_CONTROL_PASSED = "negative_control_passed"
    NEGATIVE_CONTROL_PENDING = "negative_control_pending"
    ORACLE_FAILED = "oracle_failed"
    OOM = "resource_calibration:oom"
    DISK_FULL = "resource_calibration:disk_full"
    AGENT_TIMEOUT = "resource_calibration:agent_timeout"
    VERIFIER_TIMEOUT = "resource_calibration:verifier_timeout"
    INFRASTRUCTURE = "resource_calibration:infrastructure"
    NO_ORACLE_LIVENESS = "no_oracle_liveness_missing_or_failed"
    SOURCE_ENV_INCOMPATIBLE_WITH_GRADER = "SOURCE_ENV_INCOMPATIBLE_WITH_GRADER"
    TOOL_PROVIDER_REQUIRED = "tool_provider_required"
    SHARED_EXECUTOR_REQUIRED = "shared_executor_required"
    ENVIRONMENT_BUILD_FAILED = "environment_build_failed"
    EXECUTABILITY_PROOF_MISSING = "executability_proof_missing"
    EXECUTABILITY_RUN_FAILED = "executability_run_failed"
    NETWORK_POLICY_UNSUPPORTED = "network_policy_unsupported"
    UNPINNED_ENVIRONMENT = "unpinned_environment"
    ENVIRONMENT_CONTRACT_UNRESOLVED = "environment_contract_unresolved"
    BACKEND_CONTRACT_UNSUPPORTED = "backend_contract_unsupported"


class MechanicalOracleTaskReceipt(BaseModel):
    """Keep an unexecutable original task and its evidence without inventing a trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: Literal["mechanical"] = "mechanical"
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str | None
    has_oracle: bool
    exclusion_reason: OracleExclusionReason
    evidence: tuple[ContentEvidenceReference, ...] = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return False


class ExecutabilityRunReceipt(BaseModel):
    """A bounded strong-model production trial; reward is separate from executability."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model_name: Literal["glm-5.3"] = "glm-5.3"
    temperature: float = Field(default=0.7, ge=0.7, le=0.7)
    max_turns: Literal[3] = 3
    actor_turns: int = Field(ge=1, le=3)
    model_evidence: tuple[ContentEvidenceReference, ...] = Field(min_length=1)
    trial: OracleTrialReceipt

    @property
    def passed(self) -> bool:
        result = self.trial.result
        agent = result.get("config", {}).get("agent", {})
        name = agent.get("name")
        kwargs = agent.get("kwargs", {})
        temperature = (
            kwargs.get("request_parameters", {}).get("temperature")
            if name == "task-spec-chat"
            else kwargs.get("temperature")
        )
        return (
            name in {"task-spec-chat", "terminus-2"}
            and (result.get("agent_info") or {}).get("name") == name
            and str(agent.get("model_name", "")).rsplit("/", 1)[-1] == self.model_name
            and kwargs.get("max_turns") == self.max_turns
            and temperature == self.temperature
            and result.get("config", {}).get("environment", {}).get("type") in {"local", "daytona"}
            and result.get("exception_info") is None
            and result.get("finished_at") is not None
            and result.get("completion") is not None
            and self.trial.graded
        )


class SourceEnvironmentIncompatibility(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    signature: Literal["pytest-6.2.5/pytest-metadata-stashkey", "pytest-before-7/pytest-metadata-stashkey"] = (
        "pytest-6.2.5/pytest-metadata-stashkey"
    )
    source_setup_sha256: str
    verdict_sha256: str
    source_environment_sha256: str | None = None


def source_environment_incompatibility(
    trial: OracleTrialReceipt, source_setup: str
) -> SourceEnvironmentIncompatibility | None:
    """Recognize the measured pre-agent pytest/plugin conflict without changing its environment."""
    verdict = trial.verdict or {}
    error = verdict.get("detail", {}).get("error", "")
    if (
        trial.oracle_exit_code != 0
        or verdict.get("status") != "infra_error"
        or "pytest wrote no json report (exit 1)" not in error
        or "/opt/taskcompendium-shared/runner/pytest_metadata/plugin.py" not in error
        or "AttributeError: module 'pytest' has no attribute 'StashKey'" not in error
    ):
        return None
    observation = trial.source_environment
    if observation is not None:
        if int(observation.pytest_version.partition(".")[0]) >= 7 or observation.pytest_metadata_version is None:
            return None
        return SourceEnvironmentIncompatibility(
            signature="pytest-before-7/pytest-metadata-stashkey",
            source_setup_sha256=hashlib.sha256(source_setup.encode()).hexdigest(),
            verdict_sha256=canonical_sha256(verdict),
            source_environment_sha256=canonical_sha256(observation.model_dump(mode="json")),
        )
    if "pip install -q pytest==6.2.5" not in source_setup.splitlines():
        return None
    return SourceEnvironmentIncompatibility(
        source_setup_sha256=hashlib.sha256(source_setup.encode()).hexdigest(),
        verdict_sha256=canonical_sha256(verdict),
    )


def oracle_failure_reason(
    trial: OracleTrialReceipt, incompatibility: SourceEnvironmentIncompatibility | None
) -> OracleExclusionReason:
    """Preserve the measured final source outcome after any approved infrastructure phase."""
    if incompatibility is not None:
        return OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER
    if trial.failure_signal not in {FailureSignal.INFRASTRUCTURE, FailureSignal.NON_RESOURCE}:
        return OracleExclusionReason(f"resource_calibration:{trial.failure_signal.value}")
    if not trial.scored:
        return OracleExclusionReason.INFRASTRUCTURE
    return OracleExclusionReason.ORACLE_FAILED


class NegativeControlApplicability(StrEnum):
    WORKSPACE = "workspace"
    ANSWER_GRADED_NOT_APPLICABLE = "answer_graded_not_applicable"


ANSWER_GRADED_MODES = frozenset(
    {
        "unsolvable",
        "reasoning_markup",
        "predicted_action",
        "structured_exact",
        "mcq",
        "math",
        "numeric",
        "exact",
        "structure_closure",
        "json_schema",
        "xml_elements",
        "csv_columns",
        "ifeval",
        "reasoning_gym",
        "judge",
    }
)


def negative_control_applicability(task: TaskSpec) -> NegativeControlApplicability:
    """Derive control scope from the state the source grading contract reads."""
    if assess_grading_environment(task).required is GradingEnvironmentMode.SHARED:
        return NegativeControlApplicability.WORKSPACE
    if (
        isinstance(task.grader, ScriptGrader)
        and task.grader.answer_path is not None
        and not task.output_paths
        and not actor_machine_required(task)
    ):
        return NegativeControlApplicability.ANSWER_GRADED_NOT_APPLICABLE
    if isinstance(task.grader, VerifyitGrader) and task.grader.mode in ANSWER_GRADED_MODES:
        return NegativeControlApplicability.ANSWER_GRADED_NOT_APPLICABLE
    return NegativeControlApplicability.WORKSPACE


class OracleNegativeControlReceipt(BaseModel):
    """One unchanged physical nop Trial, anchored to source and consumed identities."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    source_task_sha256: str
    source_task_checksum: str
    runtime_sha256: str | None
    consumed_task_sha256: str
    consumed_task_checksum: str
    archive: ContentEvidenceReference
    control: ContentEvidenceReference
    result_artifact: ContentEvidenceReference
    setup_artifact: ContentEvidenceReference
    verdict_artifact: ContentEvidenceReference | None
    trial: OracleTrialReceipt
    score: float | None
    verifier_exit_code: int | None
    setup_exit_code: int | None
    original_public_setup: bool
    annotation: Literal["negative_control_verifier_error", "negative_control_verifier_timeout"] | None = None
    redrive_parent: "OracleNegativeControlReceipt | None" = Field(default=None, exclude_if=lambda value: value is None)
    redrive_decision: ContentEvidenceReference | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_physical_control(self):
        result = self.trial.result
        agent = result.get("config", {}).get("agent", {})
        if (result.get("agent_info") or {}).get("name") != "nop" or agent.get("name") != "nop":
            raise ValueError("Negative control requires an actual production nop agent")
        if any(agent.get(name) for name in ("model_name", "model_alias", "kwargs", "env", "skills", "mcp_servers")):
            raise ValueError("Negative control cannot carry an actor, model or source-command override")
        if result.get("task_checksum") != self.consumed_task_checksum:
            raise ValueError("Negative control belongs to another consumed native package")
        if self.score != (self.trial.verdict or {}).get("reward"):
            raise ValueError("Negative-control score differs from its physical verdict")
        verifier = result.get("verifier_result")
        if verifier is not None and verifier.get("exit_code") != self.verifier_exit_code:
            raise ValueError("Negative-control verifier exit differs from its production result")
        parent = self.redrive_parent
        if (parent is None) != (self.redrive_decision is None):
            raise ValueError("Negative-control re-drive requires its physical parent and approved decision")
        if parent is not None:
            if (
                parent.redrive_parent is not None
                or (parent.signal_exit_code is None and not parent.source_producer_interrupted)
                or parent.annotation is not None
            ):
                raise ValueError("Only one approved re-drive of an inconclusive native signal is permitted")
            anchors = (
                "source_task_sha256",
                "source_task_checksum",
                "runtime_sha256",
                "consumed_task_sha256",
                "consumed_task_checksum",
            )
            if any(getattr(self, name) != getattr(parent, name) for name in anchors) or (
                self.trial.execution_requirements != parent.trial.execution_requirements
                or result["config"]["agent"] != parent.trial.result["config"]["agent"]
                or result["config"].get("task", {}).get("verifier_execution_requirements")
                != parent.trial.result["config"].get("task", {}).get("verifier_execution_requirements")
                or result.get("id") == parent.trial.result.get("id")
                or result.get("trial_uri") == parent.trial.result.get("trial_uri")
                or self.trial.path == parent.trial.path
            ):
                raise ValueError("Negative-control re-drive changed its contract or reused the physical Trial")
        if self.annotation == "negative_control_verifier_error" and not self.accepted_source_error:
            raise ValueError("Negative-control annotation does not prove the approved source verifier error")
        if self.annotation == "negative_control_verifier_timeout" and not self.accepted_source_deadline:
            raise ValueError("Negative-control annotation does not prove its original verifier deadline")
        return self

    @property
    def completed_public_setup(self) -> bool:
        return (
            self.setup_exit_code == 0
            and self.original_public_setup
            and self.trial.result.get("finished_at") is not None
            and self.trial.result.get("completion") is not None
            and not self.trial.result.get("secondary_exception_info")
            and not self.trial.resource_usage.oom_kills
            and not self.trial.verifier_resource_usage.oom_kills
        )

    @property
    def accepted_source_deadline(self) -> bool:
        exception = self.trial.result.get("exception_info") or {}
        timeout = self.trial.execution_requirements.verifier_timeout_sec
        return (
            self.completed_public_setup
            and exception.get("exception_type") == "VerifierTimeoutError"
            and exception.get("exception_message") == f"Verifier execution timed out after {float(timeout)} seconds"
            and self.trial.result["config"]["resolved_timeouts"]["verifier"] == timeout
            and self.trial.verdict is None
            and self.score is None
        )

    @property
    def signal_exit_code(self) -> int | None:
        error = (self.trial.verdict or {}).get("detail", {}).get("error", "")
        if not self.completed_public_setup or (
            (self.trial.result.get("exception_info") or {}).get("exception_type") != "VerifierRuntimeError"
            or (self.trial.verdict or {}).get("status") != "infra_error"
        ):
            return None
        return next(
            (
                code
                for code in (-9, -6, -11)
                if f"pytest producer failed before a usable json report (exit {code}):" in error
            ),
            None,
        )

    @property
    def source_producer_interrupted(self) -> bool:
        return (
            self.completed_public_setup
            and (self.trial.result.get("exception_info") or {}).get("exception_type") == "VerifierRuntimeError"
            and (self.trial.verdict or {}).get("status") == "infra_error"
            and (self.trial.verdict or {}).get("detail", {}).get("error")
            == "RuntimeError: pytest producer was interrupted"
        )

    @property
    def accepted_source_error(self) -> bool:
        error = (self.trial.verdict or {}).get("detail", {}).get("error", "")
        signatures = (
            (
                "exit 4",
                "src/sqlfluff/core/plugin/host.py:61: in _load_plugin",
                "UnboundLocalError: local variable 'plugin' referenced before assignment",
            ),
            (
                "exit 4",
                "src/sqlfluff/core/plugin/host.py:88: in get_plugin_manager",
                "UnboundLocalError: local variable 'plugin_manager' referenced before assignment",
            ),
            (
                "exit 4",
                "src/sqlfluff/core/plugin/host.py:80: in get_plugin_manager",
                "UnboundLocalError: local variable 'plugin_manager' referenced before assignment",
            ),
            ("exit 4", "src/sqlfluff/cli/helpers.py:92: in _sequence", "TypeError: 'list' object is not callable"),
            ("exit 3", '/testbed/patsy/eval.py", line 56, in __getitem__', "INTERNALERROR> KeyError: 0"),
        )
        return (
            self.completed_public_setup
            and (self.trial.result.get("exception_info") or {}).get("exception_type") == "VerifierRuntimeError"
            and (self.trial.verdict or {}).get("status") == "infra_error"
            and (
                (
                    "pytest producer failed before a usable json report" in error
                    and any(all(fragment in error for fragment in signature) for signature in signatures)
                )
                or error == "RuntimeError: pytest report contains unsupported test outcomes"
                or (
                    self.redrive_parent is not None
                    and (
                        (
                            self.signal_exit_code is not None
                            and self.signal_exit_code == self.redrive_parent.signal_exit_code
                        )
                        or (self.source_producer_interrupted and self.redrive_parent.source_producer_interrupted)
                    )
                )
            )
        )

    @property
    def conclusive(self) -> bool:
        return (
            self.setup_exit_code == 0
            and self.original_public_setup
            and self.trial.result.get("exception_info") is None
            and self.trial.result.get("finished_at") is not None
            and self.trial.result.get("completion") is not None
            and self.verifier_exit_code in {None, 0}
            and (self.trial.verdict or {}).get("detail", {}).get("reason") != "setup_failed"
            and (
                "total" not in (self.trial.verdict or {}).get("detail", {})
                or (self.trial.verdict or {})["detail"]["total"] > 0
            )
            and self.trial.graded
        )

    def passed(self, policy: NativeOracleSuccessPolicy | None) -> bool:
        if not self.conclusive:
            return False
        if policy is not None:
            return policy.rewards_pass(self.trial)
        return ((self.trial.result.get("verifier_result") or {}).get("rewards") or {}).get("reward") == 1


class OracleTaskReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: Literal["trial"] = "trial"
    task_sha256: str
    runtime_sha256: str | None = None
    harbor_task_checksum: str
    source_task_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_task_checksum: str | None = Field(default=None, exclude_if=lambda value: value is None)
    legacy_role_qualification: LegacyRoleQualificationReceipt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    native_success_policy: NativeOracleSuccessPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    negative_control: OracleNegativeControlReceipt | None = Field(default=None, exclude_if=lambda value: value is None)
    negative_control_applicability: NegativeControlApplicability | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    has_oracle: bool
    trials: tuple[OracleTrialReceipt, ...]
    calibration: ResourceCalibrationReceipt
    no_oracle_liveness: NoOracleLivenessReceipt | None = None
    executability_run: ExecutabilityRunReceipt | None = None
    exclusion_reason: OracleExclusionReason | None = None
    source_environment_incompatibility: SourceEnvironmentIncompatibility | None = None
    infrastructure_retries: tuple[InfrastructureRetryTransition, ...] = ()

    @model_validator(mode="after")
    def validate_negative_control(self):
        control = self.negative_control
        if control is None:
            return self
        if self.negative_control_applicability is not NegativeControlApplicability.WORKSPACE or (
            control.source_task_sha256 != self.source_task_sha256
            or control.source_task_checksum != self.source_task_checksum
            or control.runtime_sha256 != self.runtime_sha256
            or control.consumed_task_sha256 != self.task_sha256
            or control.consumed_task_checksum != self.harbor_task_checksum
            or control.trial.execution_requirements != self.calibration.execution_requirements
        ):
            raise ValueError("Negative control differs from the source or consumed task contract")
        source = self.calibration.attempts[-1]
        if control.trial.verifier_execution_requirements != self.verifier_execution_requirements:
            raise ValueError("Negative control differs from the original verifier allocation")
        if control.conclusive and (source.verdict is None) != (control.trial.verdict is None):
            raise ValueError("Negative control uses a different production grading contract")
        return self

    @model_validator(mode="after")
    def validate_legacy_role_qualification(self):
        qualification = self.legacy_role_qualification
        if qualification is None:
            return self
        final = self.calibration.attempts[-1]
        if (
            not self.has_oracle
            or self.calibration.oracle_kind != "original"
            or not self.calibration.passed
            or len(self.calibration.attempts) != 1
            or self.infrastructure_retries
            or final.verifier_execution_requirements is not None
            or qualification.source_task_sha256 != self.source_task_sha256
            or qualification.source_task_checksum != self.source_task_checksum
            or qualification.consumed_task_checksum != self.harbor_task_checksum
            or qualification.published_task_sha256 != self.task_sha256
            or qualification.source_runtime.runtime_sha256 != self.runtime_sha256
            or qualification.source_runtime.model_dump(mode="json")
            != final.result["config"]["task"].get("source_runtime")
            or qualification.result_sha256 != final.result_sha256
            or qualification.calibration_sha256 != canonical_sha256(self.calibration.model_dump(mode="json"))
        ):
            raise ValueError("Legacy role qualification differs from its unchanged completed source pass")
        for allocation in (qualification.actor, qualification.oracle_helper, qualification.verifier):
            if allocation is not None and allocation.execution_requirements != final.execution_requirements:
                raise ValueError("Legacy allocation differs from the frozen producer's actual final budget")
        return self

    @property
    def verifier_execution_requirements(self) -> ExecutionRequirements | None:
        """Return measured allocation or the separately qualified frozen allocation."""
        if self.legacy_role_qualification is not None:
            return self.legacy_role_qualification.verifier.execution_requirements
        return self.calibration.attempts[-1].verifier_execution_requirements

    @model_validator(mode="after")
    def validate_native_success_policy(self):
        policy = self.native_success_policy
        if policy is None:
            return self
        if (
            policy.source_task_sha256 != self.source_task_sha256
            or policy.source_task_checksum != self.source_task_checksum
            or policy.runtime_sha256 != self.runtime_sha256
            or not self.has_oracle
            or any(trial.verdict is not None for trial in self.trials)
        ):
            raise ValueError("Native success policy differs from the original source contract")
        return self

    @model_validator(mode="after")
    def validate_source_environment_exclusion(self):
        evidence = self.source_environment_incompatibility
        if (self.exclusion_reason is OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER) != (
            evidence is not None
        ):
            raise ValueError("Source environment exclusions require their measured incompatibility evidence")
        if evidence is not None:
            trial = self.calibration.attempts[-1]
            verdict = trial.verdict
            if verdict is None or evidence.verdict_sha256 != canonical_sha256(verdict):
                raise ValueError("Source environment evidence belongs to another verdict")
            observation_hash = (
                canonical_sha256(trial.source_environment.model_dump(mode="json"))
                if trial.source_environment is not None
                else None
            )
            if evidence.source_environment_sha256 != observation_hash:
                raise ValueError("Source environment evidence differs from the frozen pre-agent versions")
            if evidence.signature == "pytest-before-7/pytest-metadata-stashkey" and observation_hash is None:
                raise ValueError("General pytest incompatibility evidence requires frozen pre-agent versions")
        return self

    @model_validator(mode="after")
    def validate_infrastructure_retry_chain(self):
        if (self.source_task_sha256 is None) != (self.source_task_checksum is None):
            raise ValueError("Original source task and package anchors must be supplied together")
        source_task_sha256 = self.source_task_sha256 or self.task_sha256
        source_task_checksum = self.source_task_checksum or self.harbor_task_checksum
        history = self.infrastructure_retries
        if not history:
            if self.calibration.initial_requirements is not None:
                raise ValueError("A resumed allocation requires its approved parent phase")
            if any(attempt.infrastructure_retry is not None for attempt in self.calibration.attempts):
                raise ValueError("Infrastructure attempt context requires its approved phase history")
            return self
        phases = [*(transition.parent_calibration for transition in history), self.calibration]
        seen_paths = set()
        seen_results = set()
        backend = phases[0].attempts[0].result["config"]["environment"]
        for index, transition in enumerate(history):
            if (
                transition.task_sha256 != source_task_sha256
                or transition.runtime_sha256 != self.runtime_sha256
                or transition.source_task_checksum != source_task_checksum
            ):
                raise ValueError("Infrastructure retry changed the task or its native environment")
            if transition.parent_history_sha256 != canonical_sha256(
                {"phases": [previous.model_dump(mode="json") for previous in history[:index]]}
            ):
                raise ValueError("Infrastructure retry history does not extend its exact parent")
            previous, current = phases[index : index + 2]
            start = (
                current.declared_requirements
                or current.initial_requirements
                or ExecutionRequirements.model_validate(
                    {**INITIAL_REQUIREMENTS.model_dump(), **current.preserved_requirements}
                )
            )
            expected_start = previous.execution_requirements
            expected_preserved = previous.preserved_requirements
            expected_verifier = previous.verifier
            if transition.source_runtime_resource_recovery is not None:
                recovered = recovered_source_runtime_requirements(previous, transition.source_runtime_resource_recovery)
                expected_start = recovered.execution_requirements
                expected_preserved = recovered.preserved_requirements
                expected_verifier = recovered.verifier
            expected_verifier_start = previous.attempts[-1].verifier_execution_requirements
            if transition.source_runtime_resource_recovery is not None:
                expected_verifier_start = initial_verifier_requirements(expected_start, expected_verifier)
            if current.attempts[0].verifier_execution_requirements != expected_verifier_start:
                raise ValueError("Infrastructure retry did not resume its exact approved verifier allocation")
            if start != expected_start:
                raise ValueError("Infrastructure retry did not resume its exact approved allocation")
            if (
                previous.declared_requirements != current.declared_requirements
                or expected_preserved != current.preserved_requirements
                or expected_verifier != current.verifier
                or previous.oracle_kind != current.oracle_kind
            ):
                raise ValueError("Infrastructure retry changed declared limits or the verifier environment")
        for phase_index, phase in enumerate(phases):
            transition = history[phase_index - 1] if phase_index else None
            expected_context = (
                InfrastructureRetryContext(
                    parent_result_sha256=transition.approval.parent_result_sha256,
                    decision_ref=transition.approval.decision_ref,
                    backend_fix_revision=transition.backend_fix.revision,
                )
                if transition is not None
                else None
            )
            if transition is not None and transition.local_host is not None:
                host = transition.local_host
                if backend["type"] != "local":
                    raise ValueError("A physical LOCAL host transition cannot change backend type")
                original_paths = {
                    key: value for key, value in backend.get("kwargs", {}).items() if key in LOCAL_EXECUTION_HOST_FIELDS
                }
                if original_paths != host.parent_kwargs:
                    raise ValueError("A physical host transition does not describe its actual parent worker")
                backend = {
                    **backend,
                    "kwargs": {
                        **{
                            key: value
                            for key, value in backend.get("kwargs", {}).items()
                            if key not in LOCAL_EXECUTION_HOST_FIELDS
                        },
                        **host.current_kwargs,
                    },
                }
            for trial in phase.attempts:
                if trial.infrastructure_retry != expected_context:
                    raise ValueError("Infrastructure attempt context differs from its approved fixed-backend phase")
                actual = trial.result["config"]["environment"]
                if actual["type"] != backend["type"] or actual.get("kwargs", {}) != backend.get("kwargs", {}):
                    raise ValueError("Infrastructure retry changed the original backend configuration")
            paths = {trial.path for trial in phase.attempts}
            results = {trial.result_sha256 for trial in phase.attempts}
            if seen_paths.intersection(paths) or seen_results.intersection(results):
                raise ValueError("Infrastructure retry repeats a prior attempt")
            seen_paths.update(paths)
            seen_results.update(results)
        if not self.calibration.passed:
            expected_reason = oracle_failure_reason(
                self.calibration.attempts[-1], self.source_environment_incompatibility
            )
            if self.exclusion_reason is not expected_reason:
                raise ValueError("Retry exclusion differs from its measured final source outcome")
        return self

    @property
    def passed(self) -> bool:
        if self.exclusion_reason is not None:
            return False
        if self.has_oracle:
            if self.negative_control_applicability is NegativeControlApplicability.WORKSPACE:
                control = self.negative_control
                if control is None or control.passed(self.native_success_policy):
                    return False
                if not control.conclusive and control.annotation is None:
                    return False
            if self.native_success_policy is not None:
                return self.calibration.oracle_kind == "original" and self.native_success_policy.passes(
                    self.calibration.attempts[-1], self.harbor_task_checksum
                )
            return (
                self.calibration.passed
                and self.calibration.oracle_kind == "original"
                and any(trial.passed(self.harbor_task_checksum) for trial in self.trials)
            )
        control = self.executability_run
        return (
            self.calibration.oracle_kind == "calibration_only"
            and control is not None
            and control.passed
            and control.trial == self.calibration.attempts[-1]
            and control.trial.result.get("task_checksum") == self.harbor_task_checksum
        )


class NegativeControlPolicy(BaseModel):
    """Versioned standing rules for every eligible workspace-graded source."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    scope: Literal["global_workspace"]
    max_identical_budget_repeats: Literal[1]
    answer_graded: Literal["not_applicable"]
    source_deadline: Literal["non_passing_annotation"]
    source_error: Literal["non_passing_annotation"]
    decision: ContentEvidenceReference


class OracleFilterReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[2, 3, 4] = 4
    negative_control_policy: NegativeControlPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    recipe_identity: str | None
    input_cohort_sha256: str
    input_rows: int = Field(ge=0)
    conversion_rejections: int = Field(ge=0)
    tasks: dict[str, OracleTaskReceipt | MechanicalOracleTaskReceipt]

    @model_validator(mode="after")
    def validate_population(self):
        if self.version == 4 and self.negative_control_policy is None:
            raise ValueError("Oracle filter wire version 4 requires an explicit global negative-control policy")
        if len(self.tasks) + self.conversion_rejections != self.input_rows:
            raise ValueError("Oracle filter receipt does not cover the full input population")
        for task_id, entry in self.tasks.items():
            if isinstance(entry, MechanicalOracleTaskReceipt):
                if self.version not in {3, 4}:
                    raise ValueError("Mechanical outcomes require oracle filter wire version 3 or 4")
                continue
            if entry.native_success_policy is not None and (
                entry.native_success_policy.input_cohort_sha256 != self.input_cohort_sha256
            ):
                raise ValueError(f"Native success policy belongs to another frozen cohort: {task_id}")
            if entry.legacy_role_qualification is not None and (
                entry.legacy_role_qualification.input_cohort_sha256 != self.input_cohort_sha256
            ):
                raise ValueError(f"Legacy role qualification belongs to another frozen cohort: {task_id}")
            if self.version == 4 and entry.has_oracle and entry.negative_control_applicability is None:
                raise ValueError(f"New oracle admission requires source-bound negative-control applicability: {task_id}")
            if self.version == 2:
                continue
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
        if self.version != 4:
            raise ValueError("New admission requires oracle filter wire version 4; versions 2/3 are archival evidence")
        if task.id not in self.tasks:
            raise ValueError(f"Oracle filter receipt omits task {task.id}")
        entry = self.tasks[task.id]
        if entry.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError(f"Task changed after oracle filtering: {task.id}")
        if entry.runtime_sha256 != (canonical_sha256(runtime) if runtime is not None else None):
            raise ValueError(f"Runtime changed after oracle filtering: {task.id}")
        if entry.has_oracle != has_oracle_solution(task):
            raise ValueError(f"Oracle presence changed after filtering: {task.id}")
        if isinstance(entry, MechanicalOracleTaskReceipt):
            return False
        if entry.has_oracle and entry.negative_control_applicability != negative_control_applicability(task):
            raise ValueError(f"Negative-control applicability differs from the source grading contract: {task.id}")
        if task.execution_requirements is None:
            raise ValueError(f"SAMPLE/FULL admission requires explicit execution requirements: {task.id}")
        if task.execution_requirements != entry.calibration.execution_requirements:
            raise ValueError(f"Task execution requirements differ from calibration: {task.id}")
        if (
            entry.calibration.attempts[-1].result["config"].get("task", {}).get("source_runtime") is not None
            and entry.verifier_execution_requirements is None
        ):
            raise ValueError(f"Converted trial requires measured or qualified verifier allocation: {task.id}")
        if task.verifier_execution_requirements != entry.verifier_execution_requirements:
            raise ValueError(f"Task verifier execution requirements differ from calibration: {task.id}")
        qualification = entry.legacy_role_qualification
        if qualification is not None and oracle_actor_required(task) != (qualification.actor is not None):
            raise ValueError(f"Legacy qualification changed the original actor plan: {task.id}")
        return entry.passed and trial_network_qualified(task, runtime, entry.calibration.attempts[-1])

    @property
    def uniform_requirements(self) -> ExecutionRequirements | None:
        """Return a source-wide shape only when every converted task passed at that shape."""
        if not self.tasks or self.conversion_rejections or any(not entry.passed for entry in self.tasks.values()):
            return None
        entries = [entry for entry in self.tasks.values() if isinstance(entry, OracleTaskReceipt)]
        shape = entries[0].calibration.execution_requirements
        return shape if all(entry.calibration.execution_requirements == shape for entry in entries) else None


def load_oracle_filter(path: str) -> OracleFilterReceipt:
    return OracleFilterReceipt.model_validate(decode_json_value(StoragePath(path).read_text()))


def require_oracle_filter(task: TaskSpec, receipt: OracleFilterReceipt | None, runtime: dict | None = None) -> bool:
    """Validate task identity before returning its recorded qualification outcome.

    The full TaskSpec hash, oracle presence and calibrated execution requirements
    must match the receipt. Runtime presence and its complete hash must also match.
    Identity errors raise even for an excluded task.

    Returns:
        True when the original oracle or bounded model trial qualifies, or False
        when the receipt excludes the task or network evidence is incomplete.

    Raises:
        ValueError: Execution requirements, the receipt or its task entry are missing,
            or a checked identity differs from the recorded measurement.
    """
    if receipt is None:
        if task.execution_requirements is None:
            raise ValueError(f"SAMPLE/FULL admission requires explicit execution requirements: {task.id}")
        raise ValueError(f"SAMPLE/FULL admission requires a full-population Harbor oracle filter receipt: {task.id}")
    return receipt.admits(task, runtime)
