# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check eval records against the runs a policy launch from the current checkout would make."""

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

from evalrecords.records import EvalRunRecord, RunStatus, TaskCoverage, read_record, record_path
from evalrecords.rollout_attempts import RUN_OBJECT
from evalrecords.rollout_results import score_curated_record
from finestore.reader import ReadView
from iris.rpc import job_pb2
from marin.evaluation.runner import Evaluation, EvaluationBatch

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


def served_window(record: EvalRunRecord) -> int | None:
    """The context window the evaluator read from the server, from the run's archive."""
    payload = ReadView(record.results_path).read_blob(RUN_OBJECT)
    return None if payload is None else json.loads(payload)["config"]["model"]["context_window"]


@dataclass(frozen=True)
class RunEvidence:
    """What a succeeded run's archive shows: its scored coverage and the window it read from the server."""

    coverage: Mapping[str, TaskCoverage]
    window: int | None


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
        return RunCheck(record=record, state=state, problems=problems)
    scored = score_curated_record(record)
    state, problems = run_problems(record, RunEvidence(coverage=scored.coverage, window=served_window(record)), expected)
    return RunCheck(record=scored, state=state, problems=problems)


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
