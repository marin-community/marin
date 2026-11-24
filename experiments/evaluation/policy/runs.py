# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check eval records against the runs a policy launch from the current checkout would make."""

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

from evalrecords.records import EvalRunRecord, RunStatus, TaskCoverage, read_record, record_path
from evalrecords.rollout_attempts import RUN_OBJECT
from evalrecords.rollout_results import score_curated_record, terminal_attempts
from finestore.reader import ReadView
from iris.rpc import job_pb2
from marin.evaluation.runner import Evaluation, EvaluationBatch
from taskcompendium.evaluation.spec import EvaluationProtocol, OperationKind

from experiments.evaluation.launch import (
    POLICY_RECORDS_PREFIXES,
    batch_model_ref,
    evaluation_identity,
    model_identity,
    prepare_evaluation_batch,
)
from experiments.evaluation.policy.launch import launch_specs
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate

_MAX_VALUE_CHARS = 60
_SINGLE_TURN = frozenset({OperationKind.CHAT_GENERATION, OperationKind.TEXT_GENERATION})
CAPPED_WARNING = 0.10
"""Share of single-turn replies stopped by their output cap above which a run is flagged."""
SCORED_WARNING = 0.95
"""Share of a task's attempts scored below which the run is flagged."""


class RunState(StrEnum):
    """A run's standing against the policy; ``STALE`` runs had other inputs and ran at another commit."""

    VALID = "valid"
    INVALID = "invalid"
    STALE = "differs from this checkout"


@dataclass(frozen=True)
class ExpectedRun:
    """One evaluation as a policy launch from this checkout would run it, and where its record goes."""

    batch: EvaluationBatch
    evaluation: Evaluation

    @property
    def name(self) -> str:
        return self.evaluation.identity.eval_ref.name

    @property
    def record_path(self) -> str:
        return record_path(self.batch.records_prefix, self.evaluation.identity.run_id)


@dataclass(frozen=True)
class RunCheck:
    """A record, scored from its archive, and how it compares with the policy."""

    record: EvalRunRecord
    state: RunState
    problems: tuple[str, ...]
    warnings: tuple[str, ...]
    """Results far outside expectations, such as replies cut off by their cap; they do not change the state."""


def expected_runs(
    policy: EvalPolicy, candidate: PolicyCandidate, *, limit: int | None, evaluations: Collection[str] | None = None
) -> tuple[ExpectedRun, ...]:
    """Every run ``policy launch`` would submit for ``candidate`` at ``limit``, resolved as a dry run does."""
    specs = launch_specs(
        policy, candidate, evaluations=evaluations, limit=limit, priority_band=job_pb2.PRIORITY_BAND_BATCH
    )
    return tuple(
        ExpectedRun(batch=batch, evaluation=evaluation)
        for batch in (prepare_evaluation_batch(spec) for spec in specs)
        for evaluation in batch.evaluations
    )


def differences(expected: object, actual: object, path: str) -> list[str]:
    """The dotted paths at which ``actual`` differs from ``expected``, with both values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [
            difference
            for key in sorted(expected.keys() | actual.keys())
            for difference in differences(expected.get(key), actual.get(key), f"{path}.{key}")
        ]
    if expected == actual:
        return []
    return [f"{path}: ran {_short(actual)}, policy gives {_short(expected)}"]


def _short(value: object) -> str:
    text = json.dumps(value)
    return text if len(text) <= _MAX_VALUE_CHARS else f"{text[:_MAX_VALUE_CHARS]}…"


@dataclass(frozen=True)
class RunEvidence:
    """What a succeeded run's archive shows: scored coverage, the window it read from the server, and how
    many of its single-turn replies stopped at their output cap."""

    coverage: Mapping[str, TaskCoverage]
    window: int | None
    replies: int
    capped: int


def run_evidence(record: EvalRunRecord, coverage: Mapping[str, TaskCoverage]) -> RunEvidence:
    """Read a succeeded run's archive; each reply's cap is the run budget or its protocol's own, if lower."""
    reader = ReadView(record.results_path)
    payload = reader.read_blob(RUN_OBJECT)
    assert payload is not None, f"{record.results_path} has no {RUN_OBJECT}"
    run = json.loads(payload)
    budget = run["config"]["max_tokens"]
    caps = {}
    for entry in run["datasets"]:
        protocol = EvaluationProtocol.model_validate(entry["protocol"])
        if protocol.operation in _SINGLE_TURN:
            assert protocol.sampling is not None
            caps[entry["dataset"]] = min(budget, protocol.sampling.max_tokens or budget)
    tokens = [
        (row["completion_tokens"], caps[row["dataset"]])
        for row in terminal_attempts(reader).values()
        if row["dataset"] in caps and row["completion_tokens"] is not None
    ]
    return RunEvidence(
        coverage=coverage,
        window=run["config"]["model"]["context_window"],
        replies=len(tokens),
        capped=sum(used >= cap for used, cap in tokens),
    )


def run_warnings(evidence: RunEvidence) -> tuple[str, ...]:
    """Flag a run whose replies mostly ran out of tokens or whose attempts mostly went unscored."""
    warnings = []
    if evidence.capped > CAPPED_WARNING * evidence.replies:
        warnings.append(f"{evidence.capped}/{evidence.replies} replies hit their output cap")
    warnings.extend(
        f"{task}: scored {coverage.n_scored} of {coverage.n_attempted} attempts"
        for task, coverage in sorted(evidence.coverage.items())
        if coverage.n_scored < SCORED_WARNING * coverage.n_attempted
    )
    return tuple(warnings)


def run_problems(
    record: EvalRunRecord, evidence: RunEvidence | None, expected: ExpectedRun
) -> tuple[RunState, tuple[str, ...]]:
    """Compare an unscored record and its archive's evidence with the run this checkout would launch.

    Input differences on a record from another commit make it stale rather than invalid: any later
    change to a model config, corpus or harness changes a run's inputs.
    """
    batch, evaluation = expected.batch, expected.evaluation
    drift = [
        *differences(model_identity(batch_model_ref(batch)), model_identity(record.model), "model"),
        *differences(evaluation_identity(evaluation.identity.eval_ref), evaluation_identity(record.evaluation), "eval"),
        *differences(evaluation.identity.eval_runtime, record.provenance.eval_runtime, "eval_runtime"),
    ]
    faults = []
    if record.version != batch.version:
        faults.append(f"version {record.version!r}, not {batch.version!r}")
    if record.status is not RunStatus.SUCCEEDED:
        faults.append(f"{record.status.value}: {record.error}")
    if evidence is not None:
        if evidence.window != batch.model.serve.max_model_len:
            faults.append(f"server window {evidence.window}, policy gives {batch.model.serve.max_model_len}")
        if not evidence.coverage:
            faults.append("no coverage in the run's archive")
        curated = evaluation.identity.eval_ref.curated
        assert curated is not None, "policy evaluations are curated selections"
        if curated.settings.source_limit is None:
            faults.extend(
                f"{task}: attempted {coverage.n_attempted} of {coverage.n_benchmark} items"
                for task, coverage in sorted(evidence.coverage.items())
                if coverage.n_attempted != coverage.n_benchmark
            )
    if faults or (drift and record.provenance.git_sha == batch.provenance.git_sha):
        return RunState.INVALID, (*faults, *drift)
    return (RunState.STALE if drift else RunState.VALID), tuple(drift)


def check_run(record: EvalRunRecord, expected: ExpectedRun) -> RunCheck:
    """Score a record from its archive and compare it with the run this checkout would launch."""
    if record.status is not RunStatus.SUCCEEDED:
        state, problems = run_problems(record, None, expected)
        return RunCheck(record=record, state=state, problems=problems, warnings=())
    scored = score_curated_record(record)
    evidence = run_evidence(record, scored.coverage)
    state, problems = run_problems(record, evidence, expected)
    return RunCheck(record=scored, state=state, problems=problems, warnings=run_warnings(evidence))


def find_record(run_id: str) -> EvalRunRecord:
    """Read a run's record from whichever policy record root holds it."""
    for prefix in POLICY_RECORDS_PREFIXES:
        try:
            return read_record(record_path(prefix, run_id))
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no record for {run_id} under {POLICY_RECORDS_PREFIXES}")


def expected_record(expected: ExpectedRun) -> EvalRunRecord | None:
    """The record at the run's expected path, or None if it has not finished."""
    try:
        return read_record(expected.record_path)
    except FileNotFoundError:
        return None
