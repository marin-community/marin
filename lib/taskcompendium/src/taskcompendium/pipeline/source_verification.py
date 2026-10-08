# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample completed source outputs and gate publication on repeated controls."""

import hashlib
import heapq
import inspect
import json
import re
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from importlib.metadata import distributions
from pathlib import Path
from typing import Any
from uuid import uuid4

import shellbox
import verifyit
from fray.types import ResourceConfig
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import UnsupportedMachineSpec
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

import taskcompendium
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.models import CheckResult, CheckStatus, CheckSuite, GraderReadiness, VerificationReport
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order
from taskcompendium.pipeline.stages import (
    ACCEPTED_SHARD_TEMPLATE,
    AUDIT_INPUT_PATTERN,
    AUDIT_SHARD_TEMPLATE,
    manifest_counts,
)
from taskcompendium.pipeline.transforms import is_accepted
from taskcompendium.pipeline.verification import grader_readiness
from taskcompendium.runtime.models import RolloutRecord

SOURCE_VERIFICATION_REVISION = "6"


@dataclass(frozen=True)
class SourceVerificationPolicy:
    sample_size: int
    seed: int
    attempts: int
    minimum_pass_fraction: float

    def __post_init__(self) -> None:
        if self.sample_size < 1 or self.attempts < 1:
            raise ValueError("Verification requires positive sample and attempt counts")
        if not 0 <= self.minimum_pass_fraction <= 1:
            raise ValueError("minimum_pass_fraction must be between zero and one")


class SourceVerificationStatus(StrEnum):
    PASSED = "passed"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerificationSample:
    eligible_count: int
    rows: list[dict[str, Any]]


class VerificationTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt: int
    status: CheckStatus
    checks: list[CheckResult]
    rollouts: tuple[RolloutRecord, ...]


class TrialEvidence(BaseModel):
    """Identity and original execution provenance for one independent trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    task_sha256: str
    verification_identity: str
    attempt: int
    execution_id: str
    original_report: str
    reused_from: str | None = None


class HistoricalTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trial: VerificationTrial
    evidence: TrialEvidence


class SampleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    source: Source
    trials: list[VerificationTrial]
    previous_trials: list[HistoricalTrial] = Field(default_factory=list)


@dataclass(frozen=True)
class SavedTrial:
    trial: VerificationTrial
    evidence: TrialEvidence
    history: tuple[HistoricalTrial, ...] = ()


@dataclass(frozen=True)
class VerifiedSample:
    result: SampleResult
    evidence: tuple[TrialEvidence, ...]


@dataclass(frozen=True)
class KnownFailure:
    task_sha256: str
    checks: list[CheckResult]


def verification_identity(suite: CheckSuite, policy: SourceVerificationPolicy) -> str:
    """Identify control code, dependencies, configured runtime and repetition policy."""
    function = suite.run
    while isinstance(function, partial):
        function = function.func
    source_file = inspect.getsourcefile(function)
    assert source_file is not None, "Verification controls require inspectable implementation code"
    digest = hashlib.sha256()
    for module in (taskcompendium, verifyit, shellbox):
        assert module.__file__ is not None
        root = Path(module.__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return canonical_sha256(
        {
            "revision": SOURCE_VERIFICATION_REVISION,
            "suite": {"id": suite.id, "revision": suite.revision, "parameters": suite.parameters},
            "function": {"name": function.__qualname__, "module": function.__module__},
            "source_sha256": hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
            "grading_code_sha256": digest.hexdigest(),
            "python": sys.version,
            "dependencies": sorted(
                (distribution.metadata["Name"], distribution.version) for distribution in distributions()
            ),
            "policy": asdict(policy),
        }
    )


def _reusable_runtime(suite: CheckSuite) -> bool:
    # Online graders do not declare an immutable remote model deployment.
    if suite.parameters.get("network") == "allow":
        return False
    # A local QEMU path can change without changing its name. The immutable
    # worker image identifies the packaged kernel, rootfs and QEMU runtime.
    if suite.parameters.get("backend") == "qemu":
        worker = suite.parameters.get("worker_image")
        return isinstance(worker, str) and re.fullmatch(r".+@sha256:[0-9a-f]{64}", worker) is not None
    return True


def saved_trials(report_path: str, *, identity: str, attempts: int) -> dict[tuple[str, int], SavedTrial]:
    """Read matching trials and their history without treating a source gate as evidence."""
    path = StoragePath(report_path)
    if not path.exists():
        return {}
    with path.open("rt") as stream:
        report = json.load(stream)
    if report["implementation_revision"] != SOURCE_VERIFICATION_REVISION:
        return {}
    results = [SampleResult.model_validate(value) for value in report["results"]]
    if len({result.task_id for result in results}) != len(results):
        raise ValueError("Verification evidence contains duplicate task results")
    by_task = {result.task_id: result for result in results}
    evidence = [TrialEvidence.model_validate(value) for value in report["evidence"]]
    keys = [(item.task_id, item.attempt) for item in evidence]
    if len(set(keys)) != len(keys) or len({item.execution_id for item in evidence}) != len(evidence):
        raise ValueError("Verification evidence must preserve distinct independent executions")
    expected = {(result.task_id, trial.attempt) for result in results for trial in result.trials}
    if set(keys) != expected or sum(len(result.trials) for result in results) != len(expected):
        raise ValueError("Verification evidence does not match its trial membership")
    execution_ids = {item.execution_id for item in evidence}
    for result in results:
        identities = {
            (item.task_sha256, item.verification_identity) for item in evidence if item.task_id == result.task_id
        }
        if len(identities) != 1:
            raise ValueError("Verification trials disagree on their exact task identity")
        for previous in result.previous_trials:
            item = previous.evidence
            current = next(item for item in evidence if item.task_id == result.task_id)
            if (
                item.execution_id in execution_ids
                or item.task_id != result.task_id
                or item.task_sha256 != current.task_sha256
                or item.verification_identity != current.verification_identity
                or item.attempt != previous.trial.attempt
                or item.attempt not in range(report["policy"]["attempts"])
                or previous.trial.status != _trial_status(previous.trial.checks)
            ):
                raise ValueError("Verification history contradicts its independent task evidence")
            execution_ids.add(item.execution_id)
    reused = {}
    for item in evidence:
        result = by_task[item.task_id]
        ordinals = {trial.attempt for trial in result.trials}
        if ordinals != set(range(report["policy"]["attempts"])):
            raise ValueError("Verification evidence has incomplete independent trial ordinals")
        if item.verification_identity != identity or item.attempt >= attempts:
            continue
        trial = next(trial for trial in result.trials if trial.attempt == item.attempt)
        if trial.status != _trial_status(trial.checks):
            raise ValueError("Saved trial status contradicts its actual control evidence")
        reused[(item.task_sha256, item.attempt)] = SavedTrial(
            trial,
            item.model_copy(update={"reused_from": report_path}),
            tuple(previous for previous in result.previous_trials if previous.trial.attempt == item.attempt),
        )
    for value in report.get("prior_failed_trials", []):
        previous = HistoricalTrial.model_validate(value)
        item, trial = previous.evidence, previous.trial
        if (
            item.task_id in by_task
            or item.execution_id in execution_ids
            or item.attempt != trial.attempt
            or item.attempt not in range(report["policy"]["attempts"])
            or trial.status != _trial_status(trial.checks)
            or not any(check.status == CheckStatus.FAIL for check in trial.checks)
        ):
            raise ValueError("Prior failed evidence contradicts current sample membership or actual controls")
        execution_ids.add(item.execution_id)
        if item.verification_identity != identity or item.attempt >= attempts:
            continue
        key = item.task_sha256, item.attempt
        old = reused.get(key)
        history = () if old is None else (*old.history, HistoricalTrial(trial=old.trial, evidence=old.evidence))
        reused[key] = SavedTrial(trial, item.model_copy(update={"reused_from": report_path}), history)
    return reused


def _failed_trials(saved: Mapping[tuple[str, int], SavedTrial]) -> list[HistoricalTrial]:
    return [
        record
        for item in saved.values()
        for record in (*item.history, HistoricalTrial(trial=item.trial, evidence=item.evidence))
        if any(check.status == CheckStatus.FAIL for check in record.trial.checks)
    ]


def _known_failures(trials: list[HistoricalTrial]) -> dict[str, KnownFailure]:
    failures = {}
    for record in trials:
        item = record.evidence
        previous = failures.get(item.task_id)
        if previous is not None and previous.task_sha256 != item.task_sha256:
            raise ValueError("Prior failed controls disagree on their exact task identity")
        checks = [] if previous is None else previous.checks
        failures[item.task_id] = KnownFailure(
            item.task_sha256,
            [*checks, *(check for check in record.trial.checks if check.status == CheckStatus.FAIL)],
        )
    return failures


def _complete_trial(trial: VerificationTrial) -> bool:
    return trial.status not in {CheckStatus.INFRA_ERROR, CheckStatus.UNSUPPORTED} and not any(
        check.status in {CheckStatus.INFRA_ERROR, CheckStatus.UNSUPPORTED} for check in trial.checks
    )


def sample_result_checks(result: SampleResult) -> list[CheckResult]:
    """Keep definite historical failures visible without retaining resolved runtime blockers."""
    return [
        *(check for trial in result.trials for check in trial.checks),
        *(
            check
            for previous in result.previous_trials
            for check in previous.trial.checks
            if check.status == CheckStatus.FAIL
        ),
    ]


def _trial_status(checks: list[CheckResult]) -> CheckStatus:
    statuses = {check.status for check in checks}
    if CheckStatus.FAIL in statuses:
        return CheckStatus.FAIL
    if CheckStatus.INFRA_ERROR in statuses:
        return CheckStatus.INFRA_ERROR
    if not statuses or CheckStatus.UNSUPPORTED in statuses:
        return CheckStatus.UNSUPPORTED
    if statuses == {CheckStatus.SKIPPED}:
        return CheckStatus.SKIPPED
    return CheckStatus.PASS


@dataclass
class VerificationCounts:
    checked: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    unsupported: int = 0
    infra_error: int = 0
    inconsistent: int = 0


class SourceReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: SourceVerificationStatus
    policy: SourceVerificationPolicy
    eligible_count: int
    sample_count: int
    counts: VerificationCounts
    pass_fraction: float | None
    results: list[SampleResult]


def _sample_key(row: dict[str, Any], seed: int) -> tuple[str, str]:
    return seeded_order(row["task_id"], seed)


def sample_rows(rows: Iterator[dict[str, Any]], *, size: int, seed: int) -> VerificationSample:
    """Select accepted rows by seeded task-ID hashes and count eligible rows."""
    count = 0

    def eligible() -> Iterator[dict[str, Any]]:
        nonlocal count
        for row in rows:
            if is_accepted(row):
                count += 1
                yield row

    selected = heapq.nsmallest(size, eligible(), key=partial(_sample_key, seed=seed))
    return VerificationSample(count, selected)


def merge_samples(samples: Iterator[VerificationSample], *, size: int, seed: int) -> VerificationSample:
    count, selected = merge_sample_rows(
        ((sample.eligible_count, sample.rows) for sample in samples), size=size, key=partial(_sample_key, seed=seed)
    )
    return VerificationSample(count, selected)


def _verify_sample_with_evidence(
    row: dict[str, Any], *, suite: CheckSuite, attempts: int, identity: str, report_path: str
) -> VerifiedSample:
    task = TaskSpec.model_validate_json(row["task_json"])
    task_sha256 = canonical_sha256(task.model_dump(mode="json"))
    trials = []
    evidence = []
    history = []
    previous: Mapping[int, SavedTrial] = row.get("saved_trials", {})
    metrics = counters.current_stage()
    for attempt in range(attempts):
        if attempt in previous:
            saved = previous[attempt]
            if (
                saved.evidence.task_sha256 != task_sha256
                or saved.evidence.task_id != task.id
                or saved.evidence.verification_identity != identity
                or saved.trial.attempt != attempt
            ):
                raise ValueError("Saved trial does not match the exact selected task and verification identity")
            history.extend(saved.history)
            if _complete_trial(saved.trial) and _reusable_runtime(suite):
                trials.append(saved.trial)
                evidence.append(saved.evidence)
                metrics.update_counter("verification/reused_attempts", 1)
                metrics.update_counter(f"verification/suite/{suite.id}/reused_attempts", 1)
                continue
            history.append(HistoricalTrial(trial=saved.trial, evidence=saved.evidence))
        started = time.monotonic()
        try:
            report = suite.run(task)
        except UnsupportedMachineSpec as error:
            report = VerificationReport(
                [
                    CheckResult(
                        check="runtime_compatibility",
                        status=CheckStatus.UNSUPPORTED,
                        detail=str(error),
                    )
                ]
            )
        finally:
            metrics.update_counter("verification/attempts", 1)
            metrics.update_counter("verification/executed_attempts", 1)
            elapsed = time.monotonic() - started
            metrics.update_counter("verification/attempt_seconds", elapsed)
            metrics.update_counter(f"verification/suite/{suite.id}/executed_attempts", 1)
            metrics.update_counter(f"verification/suite/{suite.id}/attempt_seconds", elapsed)
        for check in report.checks:
            metrics.update_counter(f"verification/control/{check.check}/{check.status.value}", 1)
        status = _trial_status(report.checks)
        metrics.update_counter(f"verification/trial/{status.value}", 1)
        trials.append(VerificationTrial(attempt=attempt, status=status, checks=report.checks, rollouts=report.rollouts))
        evidence.append(
            TrialEvidence(
                task_id=task.id,
                task_sha256=task_sha256,
                verification_identity=identity,
                attempt=attempt,
                execution_id=uuid4().hex,
                original_report=report_path,
            )
        )
    return VerifiedSample(
        SampleResult(task_id=task.id, source=task.source, trials=trials, previous_trials=history), tuple(evidence)
    )


def source_verification_report(
    sample: VerificationSample,
    results: list[SampleResult],
    policy: SourceVerificationPolicy,
) -> SourceReport:
    """Gate completed checks and report missing golden coverage separately."""
    expected = {row["task_id"] for row in sample.rows}
    if len(expected) != len(sample.rows) or len(results) != len(expected) or {r.task_id for r in results} != expected:
        raise ValueError("Verification results must cover each sampled task exactly once")
    counts = VerificationCounts()
    for result in results:
        if len(result.trials) != policy.attempts or {trial.attempt for trial in result.trials} != set(
            range(policy.attempts)
        ):
            raise ValueError("Verification result has incomplete trials")
        statuses = {trial.status for trial in result.trials}
        historical_failure = any(
            check.status == CheckStatus.FAIL for previous in result.previous_trials for check in previous.trial.checks
        )
        failed = CheckStatus.FAIL in statuses or historical_failure
        counts.passed += statuses == {CheckStatus.PASS} and not historical_failure
        counts.failed += failed
        counts.inconsistent += CheckStatus.PASS in statuses and failed
        all_checks = [check for trial in result.trials for check in trial.checks]
        counts.checked += historical_failure or any(c.status in (CheckStatus.PASS, CheckStatus.FAIL) for c in all_checks)
        counts.skipped += any(c.status == CheckStatus.SKIPPED for c in all_checks)
        counts.unsupported += CheckStatus.UNSUPPORTED in statuses or any(
            c.status == CheckStatus.UNSUPPORTED for c in all_checks
        )
        counts.infra_error += any(c.status == CheckStatus.INFRA_ERROR for c in all_checks)
    size = len(sample.rows)
    pass_fraction = counts.passed / counts.checked if counts.checked else None
    if not size or counts.unsupported or counts.infra_error:
        status = SourceVerificationStatus.INCONCLUSIVE
    elif not counts.checked:
        status = SourceVerificationStatus.SKIPPED
    elif pass_fraction is not None and pass_fraction >= policy.minimum_pass_fraction:
        status = SourceVerificationStatus.PASSED
    else:
        status = SourceVerificationStatus.REJECTED
    return SourceReport(
        status=status,
        policy=policy,
        eligible_count=sample.eligible_count,
        sample_count=size,
        counts=counts,
        pass_fraction=pass_fraction,
        results=sorted(results, key=lambda result: result.task_id),
    )


def gate_source_row(
    row: dict[str, Any],
    *,
    status: SourceVerificationStatus,
    results: dict[str, list[CheckResult]],
    sampled_readiness: GraderReadiness = GraderReadiness.SOURCE_SAMPLED,
    previous_failures: Mapping[str, KnownFailure] | None = None,
) -> dict[str, Any]:
    previous = None if previous_failures is None else previous_failures.get(row["task_id"])
    if previous is not None and row["task_json"] is not None:
        digest = canonical_sha256(TaskSpec.model_validate_json(row["task_json"]).model_dump(mode="json"))
        if digest == previous.task_sha256:
            results = {**results, row["task_id"]: [*results.get(row["task_id"], []), *previous.checks]}
    if row["task_id"] in results:
        checks = results[row["task_id"]]
        row = {
            **row,
            "checks": [check.model_dump(mode="json") for check in checks],
            "grader_readiness": grader_readiness(checks).value,
        }
        failed = [f"check:{check.check}" for check in checks if check.status == CheckStatus.FAIL]
        if failed:
            row = {**row, "filter_status": "reject", "filter_reasons": [*row["filter_reasons"], *sorted(set(failed))]}
    if not is_accepted(row):
        return row
    if status == SourceVerificationStatus.PASSED:
        if row["task_id"] not in results:
            return {**row, "grader_readiness": sampled_readiness.value}
        return row
    if status == SourceVerificationStatus.SKIPPED:
        return {**row, "grader_readiness": GraderReadiness.UNVERIFIED.value}
    return {
        **row,
        "filter_status": "defer" if status == SourceVerificationStatus.INCONCLUSIVE else "reject",
        "filter_reasons": [*row["filter_reasons"], f"source_verification:{status.value}"],
    }


def verify_source(
    source_path: str,
    output_path: str,
    policy: SourceVerificationPolicy,
    suite: CheckSuite,
    max_workers: int,
    worker_resources: ResourceConfig | None = None,
    *,
    telemetry: PhaseTelemetry | None = None,
    context: ZephyrContext | None = None,
    previous_report_path: str | None = None,
) -> dict[str, Any]:
    """Verify an output sample and retain the full audit when publication is denied."""
    source, output = StoragePath(source_path), StoragePath(output_path)
    identity = verification_identity(suite, policy)
    prior = {}
    local_report = output / "verification.json"
    previous = str(local_report) if local_report.exists() else previous_report_path
    if previous is not None:
        prior = saved_trials(previous, identity=identity, attempts=policy.attempts)
    failed_trials = _failed_trials(prior)
    inputs = str(source / AUDIT_INPUT_PATTERN)
    with (
        nullcontext(context)
        if context is not None
        else ZephyrContext(max_workers=max_workers, resources=worker_resources, name="verify-source")
    ) as context:
        sample = execute_phase(
            context,
            Dataset.from_files(inputs)
            .load_parquet(columns=["task_id", "task_json", "filter_status"])
            .reduce(
                partial(sample_rows, size=policy.sample_size, seed=policy.seed),
                partial(merge_samples, size=policy.sample_size, seed=policy.seed),
            ),
            telemetry=telemetry,
            operation="select",
        ).results[0]
        selected = []
        for row in sample.rows:
            digest = canonical_sha256(TaskSpec.model_validate_json(row["task_json"]).model_dump(mode="json"))
            selected.append(
                {
                    **row,
                    "saved_trials": {
                        attempt: prior[(digest, attempt)]
                        for attempt in range(policy.attempts)
                        if (digest, attempt) in prior
                    },
                }
            )
        verified = (
            execute_phase(
                context,
                Dataset.from_list(selected).map(
                    partial(
                        _verify_sample_with_evidence,
                        suite=suite,
                        attempts=policy.attempts,
                        identity=identity,
                        report_path=str(output / "verification.json"),
                    )
                ),
                telemetry=telemetry,
                operation="trials",
            ).results
            if sample.rows
            else []
        )
        results = [item.result for item in verified]
        sampled_ids = {result.task_id for result in results}
        outside_failures = [record for record in failed_trials if record.evidence.task_id not in sampled_ids]
        decision = source_verification_report(sample, results, policy)
        report = {
            **decision.model_dump(mode="json"),
            "source_path": str(source),
            "suite": {"id": suite.id, "revision": suite.revision, "parameters": suite.parameters},
            "implementation_revision": SOURCE_VERIFICATION_REVISION,
            "evidence": [evidence.model_dump(mode="json") for item in verified for evidence in item.evidence],
            "prior_failed_trials": [previous.model_dump(mode="json") for previous in outside_failures],
        }
        with (output / "verification.json").open("wt", auto_mkdir=True) as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        execute_phase(
            context,
            Dataset.from_files(inputs)
            .load_parquet()
            .map(
                partial(
                    gate_source_row,
                    status=decision.status,
                    sampled_readiness=(
                        GraderReadiness.UNVERIFIED if decision.counts.skipped else GraderReadiness.SOURCE_SAMPLED
                    ),
                    results={result.task_id: sample_result_checks(result) for result in results},
                    previous_failures=_known_failures(outside_failures),
                )
            )
            .write_parquet(str(output / AUDIT_SHARD_TEMPLATE), schema=TASK_SCHEMA),
            telemetry=telemetry,
            operation="row_gate",
        )
        execute_phase(
            context,
            Dataset.from_files(str(output / AUDIT_INPUT_PATTERN))
            .load_parquet()
            .filter(is_accepted)
            .write_parquet(str(output / ACCEPTED_SHARD_TEMPLATE), schema=TASK_SCHEMA),
            telemetry=telemetry,
            operation="accepted",
        )
        counts = manifest_counts(output, context, telemetry=telemetry)
    manifest = {**counts, "verification": report}
    with (source / "manifest.json").open("rt") as stream:
        expected = json.load(stream)["input_rows"]
    if manifest["input_rows"] != expected:
        raise ValueError("Source verification lost audit rows")
    with (output / "manifest.json").open("wt", auto_mkdir=True) as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest
