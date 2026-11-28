# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy runs are found by their inputs at every commit and attempt, checked for compliance, and never replaced."""

import json
import subprocess
from dataclasses import replace

import pytest
from evalrecords.records import PromptFormat, RunStatus, TaskCoverage, read_record
from evalrecords.rollout_attempts import ATTEMPTS_TABLE, RUN_OBJECT, Attempt, AttemptPhase
from finestore.schema import arrow_schema
from finestore.store import DataStore
from marin.evaluation.runner import _record
from marin.execution.step_status import StatusFile
from marin.inference import vllm_backend
from rigging.filesystem.step_status import STATUS_SUCCESS
from rigging.filesystem.storage_path import prefix_join
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, Selection, SelectionStatus

from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.policy import cli
from experiments.evaluation.policy.launch import model_evaluations, source_checkout
from experiments.evaluation.policy.runs import (
    ExpectedRun,
    RunCheck,
    RunEvidence,
    RunName,
    RunState,
    expected_runs,
    found_runs,
    record_model,
    run_evidence,
    run_problems,
    run_suffix,
    run_warnings,
    split_run_id,
)
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
QWEN_NAME = "qwen3.6-35b-a3b"
HEAD = "0123456789ab"
OLD = "ba9876543210"


def _runs(model, names, *, limit=None, checkout=HEAD, attempt=1, policy=POLICY) -> dict[str, ExpectedRun]:
    plan = model_evaluations(policy, model, names)
    suffixes = {evaluation.name: run_suffix(checkout, attempt) for evaluation in plan.evaluations}
    return {run.name: run for run in expected_runs(policy, plan, limit=limit, run_suffixes=suffixes)}


def _write(run: ExpectedRun, status: RunStatus = RunStatus.SUCCEEDED, *, step: bool = True):
    _record(run.batch, run.evaluation.identity, status, None if status is RunStatus.SUCCEEDED else "boom", {}, {}, {})
    if step and status is RunStatus.SUCCEEDED:
        StatusFile(prefix_join(run.batch.records_prefix, run.run_id), "test").write_status(STATUS_SUCCESS)
    return read_record(run.record_path)


def _evidence(
    run: ExpectedRun,
    *,
    attempted: int = 30,
    scored: int | None = None,
    window: int | None = None,
    capped: int = 0,
    undecided: int = 0,
) -> RunEvidence:
    return RunEvidence(
        coverage={
            run.name: TaskCoverage(
                n_benchmark=30, n_attempted=attempted, n_scored=attempted if scored is None else scored
            )
        },
        window=run.batch.model.serve.max_model_len if window is None else window,
        replies=attempted,
        capped=capped,
        undecided={"d": undecided} if undecided else {},
    )


def test_a_run_id_names_its_inputs_commit_and_attempt(qwen, records):
    full, again, smoke = _runs(qwen, {"aime24"}), _runs(qwen, {"aime24"}), _runs(qwen, {"aime24"}, limit=20)
    old, rerun = _runs(qwen, {"aime24"}, checkout=OLD), _runs(qwen, {"aime24"}, attempt=2)
    assert full["aime24"].run_id == again["aime24"].run_id
    assert full["aime24"].base == old["aime24"].base == rerun["aime24"].base != smoke["aime24"].base
    assert len({run["aime24"].run_id for run in (full, old, rerun)}) == 3
    assert split_run_id(rerun["aime24"].run_id) == RunName(base=full["aime24"].base, checkout=HEAD, attempt=2)


def test_runs_of_the_same_inputs_are_found_at_every_commit_and_attempt(qwen, records):
    for checkout, attempt in ((OLD, 1), (HEAD, 1), (HEAD, 2)):
        _write(_runs(qwen, {"aime24"}, checkout=checkout, attempt=attempt)["aime24"])
    _write(_runs(qwen, {"aime24"}, limit=20)["aime24"])
    found = found_runs(_runs(qwen, {"aime24"})["aime24"])
    assert [(run.checkout, run.attempt) for run in found] == [(HEAD, 1), (HEAD, 2), (OLD, 1)]
    assert all(run.step_succeeded for run in found)


def test_a_compliant_run_has_no_problems(qwen, records):
    run = _runs(qwen, {"aime24"})["aime24"]
    assert run_problems(_write(run), _evidence(run), run, step_succeeded=True) == ()


@pytest.mark.parametrize(
    ("status", "evidence", "step", "problem"),
    [
        pytest.param(RunStatus.INFRA_FAILED, None, True, "infra_failed", id="failed"),
        pytest.param(RunStatus.SUCCEEDED, {"window": 32768}, True, "server window", id="another-server-window"),
        pytest.param(RunStatus.SUCCEEDED, {"undecided": 1}, True, "undecided", id="full-run-left-items-undecided"),
        pytest.param(RunStatus.SUCCEEDED, None, True, RUN_OBJECT, id="no-archive"),
        pytest.param(RunStatus.SUCCEEDED, {}, False, "step did not finish", id="step-failed-after-its-record"),
    ],
)
def test_a_run_that_falls_short_has_problems(qwen, records, status, evidence, step, problem):
    run = _runs(qwen, {"aime24"})["aime24"]
    record = _write(run, status, step=step)
    problems = run_problems(record, None if evidence is None else _evidence(run, **evidence), run, step_succeeded=step)
    assert any(problem in found for found in problems)


def test_a_run_from_uncommitted_changes_never_complies(qwen, records):
    run = _runs(qwen, {"aime24"}, checkout=f"{HEAD}-dirty0badf00d")["aime24"]
    problems = run_problems(_write(run), _evidence(run), run, step_succeeded=True)
    assert problems == ("launched from a checkout with uncommitted changes",)


def test_a_full_run_need_not_attempt_items_its_protocol_excludes(qwen, records):
    # MRCR's selection drops conversations that do not fit its window; the run still covers the benchmark.
    run = _runs(qwen, {"mrcr-32k"})["mrcr-32k"]
    assert run_problems(_write(run), _evidence(run, attempted=6), run, step_succeeded=True) == ()


def test_a_smoke_runs_inputs_differ_from_a_full_runs(qwen, records):
    full, smoke = _runs(qwen, {"mrcr-32k"})["mrcr-32k"], _runs(qwen, {"mrcr-32k"}, limit=20)["mrcr-32k"]
    problems = run_problems(_write(smoke), _evidence(full), full, step_succeeded=True)
    assert any(problem.startswith("eval.curated.settings.source_limit:") for problem in problems)


@pytest.mark.parametrize(
    ("evidence", "flagged"),
    [
        pytest.param({"capped": 3}, False, id="a-tenth-capped"),
        pytest.param({"capped": 4}, True, id="over-a-tenth-capped"),
        pytest.param({"scored": 29}, False, id="most-scored"),
        pytest.param({"scored": 28}, True, id="too-few-scored"),
    ],
)
def test_runs_far_outside_expectations_are_flagged_but_comply(qwen, records, evidence, flagged):
    run = _runs(qwen, {"aime24"})["aime24"]
    evidence = _evidence(run, **evidence)
    assert run_problems(_write(run), evidence, run, step_succeeded=True) == ()
    assert bool(run_warnings(evidence)) is flagged


def test_a_new_inference_build_or_agent_concurrency_moves_only_the_runs_it_changes(qwen, records, monkeypatch):
    names = {"aime24", "swebench-verified"}
    before = {name: run.base for name, run in _runs(qwen, names).items()}
    agent = next(evaluation for evaluation in POLICY.evaluations if evaluation.name == "swebench-verified")
    faster = agent.model_copy(update={"run": agent.run.model_copy(update={"concurrency": 32})})
    policy = replace(POLICY, evaluations=tuple(faster if e is agent else e for e in POLICY.evaluations))
    after = {name: run.base for name, run in _runs(qwen, names, policy=policy).items()}
    assert before["aime24"] == after["aime24"] and before["swebench-verified"] != after["swebench-verified"]

    monkeypatch.setattr(vllm_backend, "VLLM_GPU_RELEASE", replace(vllm_backend.VLLM_GPU_RELEASE, release_tag="next"))
    rebuilt = {name: run.base for name, run in _runs(qwen, names).items()}
    assert all(rebuilt[name] != before[name] for name in names)


def test_a_record_of_a_policy_models_name_at_another_location_is_rejected(qwen, records):
    run = _runs(qwen, {"aime24"})["aime24"]
    record = _write(run)
    moved = record.model_copy(update={"model": record.model.model_copy(update={"location": "s3://elsewhere/qwen"})})
    with pytest.raises(ValueError, match="declares at"):
        record_model(POLICY, moved)


def test_uncommitted_changes_give_a_launch_its_own_state(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("one\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], check=True
    )
    clean = source_checkout(tmp_path)
    (tmp_path / "a.py").write_text("two\n")
    edited = source_checkout(tmp_path)
    (tmp_path / "a.py").write_text("three\n")
    assert len(clean) == 12 and edited.startswith(f"{clean}-dirty") and source_checkout(tmp_path) != edited


def _found(checkout: str, attempt: int, status: RunStatus | None) -> "cli.FoundRun":
    record = None if status is None else _FakeRecord(status)
    return cli.FoundRun(
        run_id=f"base-{checkout}-a{attempt}", checkout=checkout, attempt=attempt, record=record, step_succeeded=True
    )


class _FakeRecord:
    def __init__(self, status: RunStatus):
        self.status = status


@pytest.fixture
def verdicts(monkeypatch):
    """Stand in for archive checks: each found run's verdict by run ID."""
    table: dict[str, tuple[RunState, tuple[str, ...]]] = {}

    def check(run, _expected):
        if run.record is None:
            return None
        state, warnings = table[run.run_id]
        return RunCheck(record=run.record, state=state, problems=(), warnings=warnings)

    def compliant(runs, expected):
        valid = [run for run in runs if run.record is not None and table[run.run_id][0] is RunState.VALID]
        return (valid[-1], check(valid[-1], expected)) if valid else None

    monkeypatch.setattr(cli, "check_found", check)
    monkeypatch.setattr(cli, "compliant_run", compliant)
    return table


def test_a_compliant_run_at_an_older_commit_leaves_the_evaluation_out(verdicts):
    old = _found(OLD, 1, RunStatus.SUCCEEDED)
    verdicts[old.run_id] = (RunState.VALID, ())
    assert cli._decide((old,), None, HEAD, ()) == cli._Decision(attempt=None, run=old, compliant=True)


def test_a_finished_run_here_that_doesnt_comply_waits_for_rerun_and_a_failed_one_resumes(verdicts):
    invalid, failed = _found(HEAD, 1, RunStatus.SUCCEEDED), _found(HEAD, 1, RunStatus.INFRA_FAILED)
    verdicts[invalid.run_id] = (RunState.INVALID, ())
    assert cli._decide((invalid,), None, HEAD, ()) == cli._Decision(attempt=None, run=invalid)
    verdicts[failed.run_id] = (RunState.INVALID, ())
    assert cli._decide((failed,), None, HEAD, ()) == cli._Decision(attempt=1)
    assert cli._decide((), None, HEAD, ()) == cli._Decision(attempt=1)


def test_rerun_writes_the_next_attempt_for_an_invalid_or_flagged_run_only(verdicts):
    invalid = _found(HEAD, 1, RunStatus.SUCCEEDED)
    verdicts[invalid.run_id] = (RunState.INVALID, ())
    assert cli._decide((invalid,), None, HEAD, {invalid.run_id}) == cli._Decision(attempt=2, replaces=invalid.run_id)

    flagged = _found(OLD, 1, RunStatus.SUCCEEDED)
    verdicts[flagged.run_id] = (RunState.VALID, ("4/30 replies hit their output cap",))
    assert cli._decide((flagged,), None, HEAD, {flagged.run_id}).attempt == 1

    valid = _found(OLD, 1, RunStatus.SUCCEEDED)
    verdicts[valid.run_id] = (RunState.VALID, ())
    with pytest.raises(ValueError, match="valid"):
        cli._decide((valid,), None, HEAD, {valid.run_id})
    running = _found(HEAD, 1, None)
    with pytest.raises(ValueError, match="not finished"):
        cli._decide((running,), None, HEAD, {running.run_id})


def _selection(index: int, status: SelectionStatus, reason: str | None, max_tokens: int | None) -> dict:
    return Selection(
        dataset="d",
        source_row=f"row:{index}",
        source_ordinal=index,
        status=status,
        reason=reason,
        rank=index,
        context_window=1000,
        max_tokens=max_tokens,
        prompt_tokens=None,
        reference_tokens=None,
    ).model_dump(mode="json")


def _archive(path, caps_and_tokens: list[tuple[int, int]], *, excluded: int = 0, rows: int | None = None) -> None:
    """Write a single-turn run archive whose items have the given output caps and reply lengths.

    ``excluded`` more items are excluded by the protocol, and the dataset has ``rows`` items in all.
    """
    aime24 = next(evaluation for evaluation in POLICY.evaluations if evaluation.name == "aime24")
    protocol = selection_dataset(aime24, PromptFormat.CHAT).protocol
    with DataStore.open(str(path)) as store:
        store.table(
            ATTEMPTS_TABLE,
            schema=arrow_schema(Attempt),
            primary_key=("dataset", "source_row", "trial", "retry", "phase"),
            schema_version=1,
        )
        store.table(
            SELECTIONS_TABLE, schema=arrow_schema(Selection), primary_key=("dataset", "source_row"), schema_version=1
        )
        with store.transaction() as transaction:
            for index, (cap, tokens) in enumerate(caps_and_tokens):
                row = f"row:{index}"
                transaction.table(SELECTIONS_TABLE).add(_selection(index, SelectionStatus.SELECTED, None, cap))
                transaction.table(ATTEMPTS_TABLE).add(
                    Attempt(
                        dataset="d",
                        task_id=f"t{index}",
                        source_row=row,
                        trial=0,
                        retry=0,
                        phase=AttemptPhase.FINISHED,
                        seed=None,
                        outcome="graded",
                        error=None,
                        result_blob=None,
                        prompt_tokens=10,
                        completion_tokens=tokens,
                    ).model_dump(mode="json")
                )
            for index in range(len(caps_and_tokens), len(caps_and_tokens) + excluded):
                transaction.table(SELECTIONS_TABLE).add(
                    _selection(index, SelectionStatus.EXCLUDED, "context_window", None)
                )
            total = len(caps_and_tokens) + excluded if rows is None else rows
            run = {
                "config": {"model": {"context_window": 1000}},
                "datasets": [
                    {"dataset": "d", "manifest": {"rows": total}, "protocol": protocol.model_dump(mode="json")}
                ],
            }
            transaction.write_object(RUN_OBJECT, json.dumps(run).encode())


def test_a_reply_counts_as_capped_against_its_own_items_cap(qwen, records, tmp_path):
    # The evaluator shrinks a long prompt's cap, so 300 tokens fills the second item's cap but not the first's.
    _archive(tmp_path / "results", [(1000, 300), (300, 300), (1000, 999)])
    record = _write(_runs(qwen, {"aime24"})["aime24"])
    evidence = run_evidence(record.model_copy(update={"results_path": str(tmp_path / "results")}), {})
    assert (evidence.replies, evidence.capped) == (3, 1)


def test_items_the_protocol_excludes_are_decided_and_missing_ones_are_not(qwen, records, tmp_path):
    record = _write(_runs(qwen, {"aime24"})["aime24"])
    _archive(tmp_path / "excluded", [(1000, 10)], excluded=2)
    _archive(tmp_path / "missing", [(1000, 10)], excluded=2, rows=5)
    evidence = {
        name: run_evidence(record.model_copy(update={"results_path": str(tmp_path / name)}), {})
        for name in ("excluded", "missing")
    }
    assert (evidence["excluded"].undecided, evidence["missing"].undecided) == ({}, {"d": 2})


def test_an_archive_without_its_run_object_gives_no_evidence(qwen, records, tmp_path):
    record = _write(_runs(qwen, {"aime24"})["aime24"])
    assert run_evidence(record.model_copy(update={"results_path": str(tmp_path / "missing")}), {}) is None
