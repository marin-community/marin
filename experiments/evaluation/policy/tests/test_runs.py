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
from experiments.evaluation.policy.resolve import exclusions_for
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
from experiments.evaluation.policy.schema import PolicyEvaluation
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
QWEN = next(candidate for candidate in POLICY.candidates if candidate.name == "qwen3.6-35b-a3b")
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


def _checkpoint(tmp_path, *, name: str = "qwen-step-100", window: int = 32_768, weights: bytes = b"weights"):
    """A supplied Qwen checkpoint in a local directory, served at ``window``."""
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    (directory / "model.safetensors").write_bytes(weights)
    return replace(
        QWEN.model,
        name=name,
        location=f"file://{directory}",
        revision=None,
        tokenizer=QWEN.model.location,
        tokenizer_revision=QWEN.model.revision,
        serve=replace(QWEN.model.serve, auto_overrides=False, max_model_len=window),
    )


def test_a_run_of_a_model_outside_the_policy_is_checked_against_its_recorded_model(tmp_path):
    # Served at 32,768, the checkpoint cannot run MRCR's 73,728-token window, so it leaves MRCR out.
    policy = replace(POLICY, in_distribution=(PolicyEvaluation("aime24"), PolicyEvaluation("mrcr")))
    candidate = model_candidate(_checkpoint(tmp_path))
    assert [exclusion.evaluation.name for exclusion in exclusions_for(policy, candidate)] == ["mrcr"]
    (run,) = expected_runs(policy, candidate, limit=None)
    record = _write(tmp_path / "records", run)
    (expected,) = expected_runs(policy, record_candidate(policy, record), limit=None, evaluations={"aime24"})
    assert expected.evaluation.identity.run_id == run.evaluation.identity.run_id
    assert run_problems(record, _evidence(expected), expected) == (RunState.VALID, ())


def test_rewriting_a_checkpoint_gives_its_runs_new_identities(tmp_path):
    before = model_candidate(_checkpoint(tmp_path))
    after = model_candidate(_checkpoint(tmp_path, weights=b"retrained weights"))
    paths = [
        run.record_path
        for candidate in (before, after)
        for run in expected_runs(POLICY, candidate, limit=None, evaluations={"aime24"})
    ]
    assert paths[0] != paths[1]


def test_a_supplied_model_at_a_candidates_location_is_checked_against_its_own_config(tmp_path):
    variant = model_candidate(
        replace(QWEN.model, name="qwen-variant", serve=replace(QWEN.model.serve, reasoning_parser=None))
    )
    (run,) = expected_runs(POLICY, variant, limit=None, evaluations={"aime24"})
    record = _write(tmp_path, run)
    assert record_candidate(POLICY, record).name == "qwen-variant"


def test_a_run_of_the_policy_is_valid(tmp_path):
    run = _expected(None)["aime24"]
    assert run_problems(_write(tmp_path, run), _evidence(run), run) == (RunState.VALID, ())


@pytest.mark.parametrize(
    ("status", "evidence"),
    [
        pytest.param(RunStatus.INFRA_FAILED, None, id="failed"),
        pytest.param(RunStatus.SUCCEEDED, {"window": 32768}, id="another-server-window"),
        pytest.param(RunStatus.SUCCEEDED, {"undecided": 1}, id="full-run-left-items-undecided"),
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


def test_a_full_run_need_not_attempt_items_its_protocol_excludes(tmp_path):
    # MRCR's selection drops conversations that do not fit its window; the run still covers the benchmark.
    run = _expected(None)["mrcr-32k"]
    assert run_problems(_write(tmp_path, run), _evidence(run, attempted=6), run) == (RunState.VALID, ())


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


def test_a_release_runs_at_its_own_paths_not_the_unreleased_versions():
    # Runs made before the tag share the version; they must not occupy, or be skipped in place of, the release's.
    (before,) = expected_runs(POLICY, QWEN, limit=None, evaluations={"aime24"})
    (released,) = expected_runs(POLICY, QWEN, limit=None, evaluations={"aime24"}, release_commit="a" * 40)
    (again,) = expected_runs(POLICY, QWEN, limit=None, evaluations={"aime24"}, release_commit="a" * 40)
    assert released.record_path == again.record_path != before.record_path


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


def test_a_reply_counts_as_capped_against_its_own_items_cap(tmp_path):
    # The evaluator shrinks a long prompt's cap, so 300 tokens fills the second item's cap but not the first's.
    _archive(tmp_path / "results", [(1000, 300), (300, 300), (1000, 999)])
    record = _write(tmp_path, _expected(None)["aime24"])
    evidence = run_evidence(record.model_copy(update={"results_path": str(tmp_path / "results")}), {})
    assert (evidence.replies, evidence.capped) == (3, 1)


def test_items_the_protocol_excludes_are_decided_and_missing_ones_are_not(tmp_path):
    record = _write(tmp_path, _expected(None)["aime24"])
    _archive(tmp_path / "excluded", [(1000, 10)], excluded=2)
    _archive(tmp_path / "missing", [(1000, 10)], excluded=2, rows=5)
    evidence = {
        name: run_evidence(record.model_copy(update={"results_path": str(tmp_path / name)}), {})
        for name in ("excluded", "missing")
    }
    assert (evidence["excluded"].undecided, evidence["missing"].undecided) == ({}, {"d": 2})


def test_an_archive_without_its_run_object_gives_no_evidence(tmp_path):
    record = _write(tmp_path, _expected(None)["aime24"])
    assert run_evidence(record.model_copy(update={"results_path": str(tmp_path / "missing")}), {}) is None
