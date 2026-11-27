# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find a policy's runs and check them.

A policy run's ID is ``<base>-<commit>[-dirty<digest>]-a<attempt>``: the base digests its inputs, so listing the
base finds the runs of the same inputs at every commit and attempt.
"""

import json
import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

import draccus
from evalrecords.records import EvalRunRecord, RunStatus, TaskCoverage, read_record, record_path
from evalrecords.rollout_attempts import RUN_OBJECT
from evalrecords.rollout_results import score_curated_record, terminal_attempts
from finestore.reader import ReadView
from iris.rpc import job_pb2
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import Evaluation, EvaluationBatch
from marin.execution.step_runner import check_cache
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, SelectionStatus
from taskcompendium.evaluation.spec import EvaluationProtocol, OperationKind

from experiments.evaluation.launch import (
    POLICY_RECORDS_PREFIXES,
    batch_model_ref,
    evaluation_identity,
    model_identity,
    prepare_evaluation_batch,
)
from experiments.evaluation.policy.launch import ModelEvaluations, PolicyModel, launch_specs, policy_model
from experiments.evaluation.policy.schema import Policy

_MAX_VALUE_CHARS = 60
_SINGLE_TURN = frozenset({OperationKind.CHAT_GENERATION, OperationKind.TEXT_GENERATION})
_SUFFIX = re.compile(r"(?P<checkout>[0-9a-f]{12}(?:-dirty[0-9a-f]{7,40})?)-a(?P<attempt>[1-9][0-9]*)")
_RUN_ID = re.compile(rf"(?P<base>.+-[0-9a-f]{{32}})-{_SUFFIX.pattern}")
CAPPED_WARNING = 0.10
"""Share of single-turn replies stopped by their output cap above which a run is flagged."""
SCORED_WARNING = 0.95
"""Share of a task's attempts scored below which the run is flagged."""


def run_suffix(checkout: str, attempt: int) -> str:
    """The run-ID part naming the checkout a run was launched from and its attempt there."""
    return f"{checkout}-a{attempt}"


class RunState(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class ExpectedRun:
    """One evaluation as a policy launch from this checkout would run it, and where its record goes."""

    batch: EvaluationBatch
    evaluation: Evaluation
    base: str
    """The run ID without its commit and attempt, shared by every run of the same inputs."""

    @property
    def name(self) -> str:
        return self.evaluation.identity.eval_ref.name

    @property
    def run_id(self) -> str:
        return self.evaluation.identity.run_id

    @property
    def record_path(self) -> str:
        return record_path(self.batch.records_prefix, self.run_id)


@dataclass(frozen=True)
class RunName:
    """The parts of a policy run ID: its inputs digest, the checkout it was launched from, and its attempt there."""

    base: str
    checkout: str
    """The commit, with the tree of uncommitted changes if any."""
    attempt: int


@dataclass(frozen=True)
class FoundRun:
    """A run of an expected run's inputs, at some commit and attempt."""

    run_id: str
    checkout: str
    """The commit it was launched from, with the tree of uncommitted changes if any."""
    attempt: int
    record: EvalRunRecord | None
    """None until the run finishes."""
    step_succeeded: bool

    @property
    def dirty(self) -> bool:
        return "-dirty" in self.checkout


@dataclass(frozen=True)
class RunCheck:
    """A record, scored from its archive, and how it compares with the policy."""

    record: EvalRunRecord
    state: RunState
    problems: tuple[str, ...]
    warnings: tuple[str, ...]


def expected_runs(
    policy: Policy,
    plan: ModelEvaluations,
    *,
    limit: int | None,
    run_suffixes: Mapping[str, str],
    priority_band: int = job_pb2.PRIORITY_BAND_BATCH,
    federated_cluster: str | None = None,
) -> tuple[ExpectedRun, ...]:
    """Every run ``policy launch`` would submit for one model at ``limit``, resolved as a dry run does.

    Priority and cluster place the servers but do not change any run's path.
    """
    specs = launch_specs(
        policy,
        plan,
        limit=limit,
        priority_band=priority_band,
        run_suffixes=run_suffixes,
        federated_cluster=federated_cluster,
    )
    return tuple(
        ExpectedRun(
            batch=batch,
            evaluation=evaluation,
            base=evaluation.identity.run_id.removesuffix(f"-{run_suffixes[evaluation.identity.eval_ref.name]}"),
        )
        for batch in (prepare_evaluation_batch(spec) for spec in specs)
        for evaluation in batch.evaluations
    )


def found_runs(expected: ExpectedRun) -> tuple[FoundRun, ...]:
    """Every run of the expected run's inputs, at every commit and attempt, oldest attempt first."""
    fs, root = url_to_fs(expected.batch.records_prefix)
    runs = []
    for path in fs.glob(f"{root}/{expected.base}-*"):
        run_id = path.rsplit("/", 1)[-1]
        if (match := _SUFFIX.fullmatch(run_id.removeprefix(f"{expected.base}-"))) is None:
            continue
        try:
            record = read_record(record_path(expected.batch.records_prefix, run_id))
        except FileNotFoundError:
            record = None
        runs.append(
            FoundRun(
                run_id=run_id,
                checkout=match["checkout"],
                attempt=int(match["attempt"]),
                record=record,
                step_succeeded=check_cache(prefix_join(expected.batch.records_prefix, run_id)),
            )
        )
    return tuple(sorted(runs, key=lambda run: (run.checkout, run.attempt)))


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
    record: EvalRunRecord, evidence: RunEvidence | None, expected: ExpectedRun, *, step_succeeded: bool
) -> tuple[str, ...]:
    """Every way a record and its archive's evidence fall short of the run the policy expects.

    The record must be of the same inputs (its run ID extends the expected base, and its model, evaluation and
    runtime match), from a clean checkout, finished with its step's success marker, and its archive must show
    the served window and full coverage. Items the protocol excludes, such as ones too long for its window,
    are not missing from a full run.
    """
    batch, evaluation = expected.batch, expected.evaluation
    problems = [
        *differences(model_identity(batch_model_ref(batch)), model_identity(record.model), "model"),
        *differences(evaluation_identity(evaluation.identity.eval_ref), evaluation_identity(record.evaluation), "eval"),
        *differences(evaluation.identity.eval_runtime, record.provenance.eval_runtime, "eval_runtime"),
    ]
    # The base also covers what no recorded field shows: the inference build, accelerator and agent concurrency.
    if (match := _SUFFIX.fullmatch(record.run_id.removeprefix(f"{expected.base}-"))) is None:
        problems.append(f"run_id: ran {record.run_id}, not a run of {expected.base}")
    elif "-dirty" in match["checkout"]:
        problems.append("launched from a checkout with uncommitted changes")
    if record.version != batch.version:
        problems.append(f"version {record.version!r}, not {batch.version!r}")
    if record.status is not RunStatus.SUCCEEDED:
        problems.append(f"{record.status.value}: {record.error}")
    elif not step_succeeded:
        problems.append("its record succeeded but its step did not finish successfully")
    elif evidence is None:
        problems.append(f"its archive has no {RUN_OBJECT}")
    if evidence is not None:
        if evidence.window != batch.model.serve.max_model_len:
            problems.append(f"server window {evidence.window}, policy gives {batch.model.serve.max_model_len}")
        if not evidence.coverage:
            problems.append("no coverage in the run's archive")
        curated = evaluation.identity.eval_ref.curated
        assert curated is not None, "policy evaluations are curated selections"
        if curated.settings.source_limit is None:
            problems.extend(
                f"{dataset}: the selection left {missing} items undecided"
                for dataset, missing in sorted(evidence.undecided.items())
            )
    return tuple(problems)


def check_run(record: EvalRunRecord, expected: ExpectedRun, *, step_succeeded: bool) -> RunCheck:
    """Score a record from its archive and compare it with the run the policy expects."""
    if record.status is not RunStatus.SUCCEEDED:
        problems = run_problems(record, None, expected, step_succeeded=step_succeeded)
        return RunCheck(record=record, state=RunState.INVALID, problems=problems, warnings=())
    scored = score_curated_record(record)
    evidence = run_evidence(record, scored.coverage)
    problems = run_problems(record, evidence, expected, step_succeeded=step_succeeded)
    warnings = () if evidence is None else run_warnings(evidence)
    state = RunState.INVALID if problems else RunState.VALID
    return RunCheck(record=scored, state=state, problems=problems, warnings=warnings)


def check_found(run: FoundRun, expected: ExpectedRun) -> RunCheck | None:
    """Check a found run, or None if it has not finished."""
    if run.record is None:
        return None
    return check_run(run.record, expected, step_succeeded=run.step_succeeded)


def compliant_run(runs: Collection[FoundRun], expected: ExpectedRun) -> tuple[FoundRun, RunCheck] | None:
    """The newest run, at any commit, that complies, with its check; the newest attempt at each commit counts."""
    newest: dict[str, FoundRun] = {}
    for run in runs:
        if run.record is not None and not run.dirty:
            newest[run.checkout] = run
    for run in sorted(newest.values(), key=lambda run: run.record.created_at, reverse=True):
        check = check_found(run, expected)
        if check is not None and check.state is RunState.VALID:
            return run, check
    return None


def record_model(policy: Policy, record: EvalRunRecord) -> PolicyModel:
    """The model a record ran: the policy's baseline of that name, else the supplied model it records.

    A supplied model may not take the name of one of the policy's baselines.
    """
    if record.model.name in policy.baselines:
        baseline = policy.baseline(record.model.name)
        if baseline.location != record.model.location:
            raise ValueError(
                f"{record.run_id} ran {record.model.location} as {record.model.name}, which {policy.version} "
                f"declares at {baseline.location}"
            )
        return policy_model(policy, baseline, baseline=True)
    config = record.model.source_config or record.model.config
    if config is None:
        raise ValueError(f"{record.run_id}: its record names no model config to check it against")
    return policy_model(policy, draccus.decode(ModelConfig, config.model_dump(mode="json")), baseline=False)


def find_record(run_id: str) -> tuple[EvalRunRecord, str]:
    """Read a run's record, and the policy record root that holds it."""
    for prefix in POLICY_RECORDS_PREFIXES:
        try:
            return read_record(record_path(prefix, run_id)), prefix
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no record for {run_id} under {POLICY_RECORDS_PREFIXES}")


def split_run_id(run_id: str) -> RunName:
    """The parts of a policy run ID."""
    if (match := _RUN_ID.fullmatch(run_id)) is None:
        raise ValueError(f"{run_id} is not a policy run ID")
    return RunName(base=match["base"], checkout=match["checkout"], attempt=int(match["attempt"]))
