# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from evalrecords.records import PromptFormat, RunStatus, TaskCoverage, read_record, record_path
from evalrecords.rollout_attempts import ATTEMPTS_TABLE, RUN_OBJECT, Attempt, AttemptPhase
from finestore.schema import arrow_schema
from finestore.store import DataStore
from marin.evaluation.runner import _record
from taskcompendium.evaluation.selection import SELECTIONS_TABLE, Selection, SelectionStatus

from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.policy.launch import model_candidate
from experiments.evaluation.policy.runs import (
    ExpectedRun,
    RunEvidence,
    RunState,
    expected_runs,
    record_candidate,
    run_evidence,
    run_problems,
    run_warnings,
    succeeded_runs,
)
from experiments.evaluation.policy.versions.marin_2026_11_20_trial.policy import POLICY, QWEN

EVALUATIONS = {"aime24", "mrcr-32k"}


def _expected(limit: int | None) -> dict[str, ExpectedRun]:
    return {run.name: run for run in expected_runs(POLICY, QWEN, limit=limit, evaluations=EVALUATIONS)}


def _write(tmp_path, run: ExpectedRun, status: RunStatus = RunStatus.SUCCEEDED, git_sha: str | None = None):
    batch = replace(run.batch, records_prefix=str(tmp_path))
    if git_sha is not None:
        batch = replace(batch, provenance=replace(batch.provenance, git_sha=git_sha))
    _record(batch, run.evaluation.identity, status, None if status is RunStatus.SUCCEEDED else "boom", {}, {}, {})
    return read_record(record_path(str(tmp_path), run.evaluation.identity.run_id))


def _evidence(
    run: ExpectedRun, *, attempted: int = 30, scored: int | None = None, window: int | None = None, capped: int = 0
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
    )


def test_a_relaunch_finds_each_run_at_the_same_path_and_smoke_runs_elsewhere():
    first, second, smoke = _expected(None), _expected(None), _expected(20)
    assert {name: run.record_path for name, run in first.items()} == {
        name: run.record_path for name, run in second.items()
    }
    assert not {run.record_path for run in first.values()} & {run.record_path for run in smoke.values()}


def test_a_launch_finds_succeeded_runs_and_the_rest_keep_their_ids(tmp_path):
    runs = _expected(None)
    _write(tmp_path, runs["aime24"])
    _write(tmp_path, runs["mrcr-32k"], RunStatus.INFRA_FAILED)
    at_tmp = [replace(run, batch=replace(run.batch, records_prefix=str(tmp_path))) for run in runs.values()]
    assert set(succeeded_runs(at_tmp)) == {"aime24"}
    # A relaunch leaves the succeeded run out; the failed one keeps its path, so it resumes there.
    (rerun,) = expected_runs(POLICY, QWEN, limit=None, evaluations={"mrcr-32k"})
    assert rerun.record_path == runs["mrcr-32k"].record_path


def test_a_run_of_a_model_outside_the_policy_is_checked_against_its_recorded_model(tmp_path):
    checkpoint = replace(
        QWEN.model,
        name="qwen-step-100",
        location="s3://marin-us-east-02a/checkpoints/qwen-step-100",
        revision=None,
        tokenizer=QWEN.model.location,
        tokenizer_revision=QWEN.model.revision,
        serve=replace(QWEN.model.serve, auto_overrides=False),
    )
    (run,) = expected_runs(POLICY, model_candidate(checkpoint), limit=None, evaluations={"aime24"})
    record = _write(tmp_path, run)
    (expected,) = expected_runs(POLICY, record_candidate(POLICY, record), limit=None, evaluations={"aime24"})
    assert expected.evaluation.identity.run_id == run.evaluation.identity.run_id
    assert run_problems(record, _evidence(expected), expected) == (RunState.VALID, ())


def test_a_run_of_the_policy_is_valid(tmp_path):
    run = _expected(None)["aime24"]
    assert run_problems(_write(tmp_path, run), _evidence(run), run) == (RunState.VALID, ())


@pytest.mark.parametrize(
    ("status", "evidence"),
    [
        pytest.param(RunStatus.INFRA_FAILED, None, id="failed"),
        pytest.param(RunStatus.SUCCEEDED, {"window": 32768}, id="another-server-window"),
        pytest.param(RunStatus.SUCCEEDED, {"attempted": 29}, id="full-run-missed-items"),
    ],
)
def test_a_run_whose_outcome_falls_short_is_invalid(tmp_path, status, evidence):
    run = _expected(None)["aime24"]
    record = _write(tmp_path, run, status)
    state, problems = run_problems(record, None if evidence is None else _evidence(run, **evidence), run)
    assert state is RunState.INVALID
    # A succeeded record keeps its step marked done, so the problem says a relaunch will not rerun it.
    assert any("relaunch skips" in problem for problem in problems) is (status is RunStatus.SUCCEEDED)


def test_a_succeeded_run_without_an_archive_is_invalid(tmp_path):
    run = _expected(None)["aime24"]
    state, problems = run_problems(_write(tmp_path, run), None, run)
    assert state is RunState.INVALID
    assert any(RUN_OBJECT in problem for problem in problems)


def test_a_smoke_run_need_not_attempt_every_item(tmp_path):
    run = _expected(20)["aime24"]
    assert run_problems(_write(tmp_path, run), _evidence(run, attempted=20), run) == (RunState.VALID, ())


def test_different_inputs_name_the_fields_and_are_stale_only_from_another_commit(tmp_path):
    full, smoke = _expected(None)["mrcr-32k"], _expected(20)["mrcr-32k"]
    here = _write(tmp_path, smoke)
    state, problems = run_problems(here, _evidence(full), full)
    assert state is RunState.INVALID
    assert any(problem.startswith("eval.curated.settings.source_limit:") for problem in problems)

    elsewhere = _write(tmp_path / "old", smoke, git_sha="0" * 40)
    state, problems = run_problems(elsewhere, _evidence(full), full)
    assert state is RunState.STALE
    assert any(problem.startswith("eval.curated.settings.source_limit:") for problem in problems)


def test_a_released_policys_runs_must_come_from_its_tagged_commit(tmp_path):
    run = _expected(None)["aime24"]
    record = _write(tmp_path, run)
    release = record.provenance.git_sha
    assert run_problems(record, _evidence(run), run, release_commit=release) == (RunState.VALID, ())

    # The same inputs at another commit are valid for a trial policy but not for a release.
    elsewhere = _write(tmp_path / "old", run, git_sha="0" * 40)
    assert run_problems(elsewhere, _evidence(run), run)[0] is RunState.VALID
    state, problems = run_problems(elsewhere, _evidence(run), run, release_commit=release)
    assert state is RunState.INVALID
    assert any("relaunch skips" in problem for problem in problems)


@pytest.mark.parametrize(
    ("evidence", "flagged"),
    [
        pytest.param({"capped": 3}, False, id="a-tenth-capped"),
        pytest.param({"capped": 4}, True, id="over-a-tenth-capped"),
        pytest.param({"scored": 29}, False, id="most-scored"),
        pytest.param({"scored": 28}, True, id="too-few-scored"),
    ],
)
def test_runs_far_outside_expectations_are_flagged_but_stay_valid(tmp_path, evidence, flagged):
    run = _expected(None)["aime24"]
    evidence = _evidence(run, **evidence)
    assert run_problems(_write(tmp_path, run), evidence, run) == (RunState.VALID, ())
    assert bool(run_warnings(evidence)) is flagged


def _archive(path, caps_and_tokens: list[tuple[int, int]]) -> None:
    """Write a single-turn run archive whose items have the given output caps and reply lengths."""
    protocol = selection_dataset(
        _expected(None)["aime24"].evaluation.identity.eval_ref.curated.selection, PromptFormat.CHAT
    ).protocol
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
                transaction.table(SELECTIONS_TABLE).add(
                    Selection(
                        dataset="d",
                        source_row=row,
                        source_ordinal=index,
                        status=SelectionStatus.SELECTED,
                        reason=None,
                        rank=index,
                        context_window=1000,
                        max_tokens=cap,
                        prompt_tokens=None,
                        reference_tokens=None,
                    ).model_dump(mode="json")
                )
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
            run = {
                "config": {"model": {"context_window": 1000}},
                "datasets": [{"dataset": "d", "protocol": protocol.model_dump(mode="json")}],
            }
            transaction.write_object(RUN_OBJECT, json.dumps(run).encode())


def test_a_reply_counts_as_capped_against_its_own_items_cap(tmp_path):
    # The evaluator shrinks a long prompt's cap, so 300 tokens fills the second item's cap but not the first's.
    _archive(tmp_path / "results", [(1000, 300), (300, 300), (1000, 999)])
    record = _write(tmp_path, _expected(None)["aime24"])
    evidence = run_evidence(record.model_copy(update={"results_path": str(tmp_path / "results")}), {})
    assert (evidence.replies, evidence.capped) == (3, 1)


def test_an_archive_without_its_run_object_gives_no_evidence(tmp_path):
    record = _write(tmp_path, _expected(None)["aime24"])
    assert run_evidence(record.model_copy(update={"results_path": str(tmp_path / "missing")}), {}) is None
