# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check eval records against the runs a policy launch from the current checkout would make."""

import json
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

from evalrecords.records import EvalRunRecord, RunStatus, TaskCoverage, read_record, record_path
from evalrecords.rollout_attempts import RUN_OBJECT
from evalrecords.rollout_results import score_curated_record, terminal_attempts
from finestore.reader import ReadView
from iris.rpc import job_pb2
from marin.evaluation.runner import Evaluation, EvaluationBatch
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, SelectionStatus
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


def expected_runs(
    policy: EvalPolicy,
    candidate: PolicyCandidate,
    *,
    limit: int | None,
    evaluations: Collection[str] | None = None,
    release_commit: str | None = None,
) -> tuple[ExpectedRun, ...]:
    """Every run ``policy launch`` would submit for ``candidate`` at ``limit``, resolved as a dry run does."""
    specs = launch_specs(
        policy,
        candidate,
        evaluations=evaluations,
        limit=limit,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        release_commit=release_commit,
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
    """What a succeeded run's archive shows: scored coverage, the window it read from the server, how many
    of its single-turn replies stopped at their output cap, and, by dataset, how many items its selection
    left undecided."""

    coverage: Mapping[str, TaskCoverage]
    window: int | None
    replies: int
    capped: int
    undecided: Mapping[str, int]


def run_evidence(record: EvalRunRecord, coverage: Mapping[str, TaskCoverage]) -> RunEvidence | None:
    """Read a succeeded run's archive, or None if it has no run object.

    A single-turn reply counts as capped when it used the whole output cap the evaluator gave its item. An
    item is undecided when the selection neither selected it nor excluded it under its protocol's rules,
    which happens only under an item limit.
    """
    reader = ReadView(record.results_path)
    payload = reader.read_blob(RUN_OBJECT)
    if payload is None:
        return None
    run = json.loads(payload)
    single_turn = {
        entry["dataset"]
        for entry in run["datasets"]
        if EvaluationProtocol.model_validate(entry["protocol"]).operation in _SINGLE_TURN
    }
    selections = list(reader.iter_rows(SELECTIONS_TABLE))
    caps = {
        (row["dataset"], row["source_row"]): row["max_tokens"]
        for row in selections
        if row["status"] == SelectionStatus.SELECTED and row["dataset"] in single_turn and row["max_tokens"] is not None
    }
    decided = Counter(row["dataset"] for row in selections if row["status"] != SelectionStatus.UNINSPECTED)
    tokens = [
        (row["completion_tokens"], caps[key])
        for row in terminal_attempts(reader).values()
        if (key := (row["dataset"], row["source_row"])) in caps and row["completion_tokens"] is not None
    ]
    return RunEvidence(
        coverage=coverage,
        window=run["config"]["model"]["context_window"],
        replies=len(tokens),
        capped=sum(used >= cap for used, cap in tokens),
        undecided={
            entry["dataset"]: missing
            for entry in run["datasets"]
            if (missing := entry["manifest"]["rows"] - decided[entry["dataset"]]) > 0
        },
    )


def run_warnings(evidence: RunEvidence) -> tuple[str, ...]:
    """Flag capped replies and unscored attempts that exceed their warning thresholds."""
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
    record: EvalRunRecord, evidence: RunEvidence | None, expected: ExpectedRun, *, release_commit: str | None = None
) -> tuple[RunState, tuple[str, ...]]:
    """Compare an unscored record and its archive's evidence with the run this checkout would launch.

    Input differences on a record from another commit make it stale rather than invalid: any later
    change to a model config, corpus or harness changes a run's inputs. A released policy's runs must
    come from ``release_commit``, its tag's commit.
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
    if release_commit is not None and record.provenance.git_sha != release_commit:
        faults.append(f"ran at commit {record.provenance.git_sha[:10]}, not the release's {release_commit[:10]}")
    if record.status is not RunStatus.SUCCEEDED:
        faults.append(f"{record.status.value}: {record.error}")
    elif evidence is None:
        faults.append(f"its archive has no {RUN_OBJECT}")
    if evidence is not None:
        if evidence.window != batch.model.serve.max_model_len:
            faults.append(f"server window {evidence.window}, policy gives {batch.model.serve.max_model_len}")
        if not evidence.coverage:
            faults.append("no coverage in the run's archive")
        curated = evaluation.identity.eval_ref.curated
        assert curated is not None, "policy evaluations are curated selections"
        # Items the protocol excludes, such as ones too long for its window, are not missing from a full run.
        if curated.settings.source_limit is None:
            faults.extend(
                f"{dataset}: the selection left {missing} items undecided"
                for dataset, missing in sorted(evidence.undecided.items())
            )
    if faults and not drift and record.status is RunStatus.SUCCEEDED and record.version == batch.version:
        faults.append("a relaunch skips this run, since its record succeeded")
    if faults or (drift and record.provenance.git_sha == batch.provenance.git_sha):
        return RunState.INVALID, (*faults, *drift)
    return (RunState.STALE if drift else RunState.VALID), tuple(drift)


def check_run(record: EvalRunRecord, expected: ExpectedRun, *, release_commit: str | None = None) -> RunCheck:
    """Score a record from its archive and compare it with the run this checkout would launch."""
    if record.status is not RunStatus.SUCCEEDED:
        state, problems = run_problems(record, None, expected, release_commit=release_commit)
        return RunCheck(record=record, state=state, problems=problems, warnings=())
    scored = score_curated_record(record)
    evidence = run_evidence(record, scored.coverage)
    state, problems = run_problems(record, evidence, expected, release_commit=release_commit)
    warnings = () if evidence is None else run_warnings(evidence)
    return RunCheck(record=scored, state=state, problems=problems, warnings=warnings)


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
