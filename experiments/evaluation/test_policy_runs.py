# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy runs land at stable paths, and records are checked against what this checkout would launch."""

from dataclasses import replace

import pytest
from evalrecords.records import RunStatus, TaskCoverage, read_record, record_path
from marin.evaluation.runner import _record

from experiments.evaluation.policies.marin_2026_11_20_trial.policy import POLICY, QWEN
from experiments.evaluation.policy.runs import ExpectedRun, RunEvidence, RunState, expected_runs, run_problems

EVALUATIONS = {"aime24", "mrcr-32k"}


def _expected(limit: int | None) -> dict[str, ExpectedRun]:
    return {run.name: run for run in expected_runs(POLICY, QWEN, limit=limit, evaluations=EVALUATIONS)}


def _write(tmp_path, run: ExpectedRun, status: RunStatus = RunStatus.SUCCEEDED, git_sha: str | None = None):
    batch = replace(run.batch, records_prefix=str(tmp_path))
    if git_sha is not None:
        batch = replace(batch, provenance=replace(batch.provenance, git_sha=git_sha))
    _record(batch, run.evaluation.identity, status, None if status is RunStatus.SUCCEEDED else "boom", {}, {}, {})
    return read_record(record_path(str(tmp_path), run.evaluation.identity.run_id))


def _evidence(run: ExpectedRun, *, attempted: int = 30, window: int | None = None) -> RunEvidence:
    return RunEvidence(
        coverage={run.name: TaskCoverage(n_benchmark=30, n_attempted=attempted, n_scored=attempted)},
        window=run.batch.model.serve.max_model_len if window is None else window,
    )


def test_a_relaunch_finds_each_run_at_the_same_path_and_smoke_runs_elsewhere():
    first, second, smoke = _expected(None), _expected(None), _expected(20)
    assert {name: run.record_path for name, run in first.items()} == {
        name: run.record_path for name, run in second.items()
    }
    assert not {run.record_path for run in first.values()} & {run.record_path for run in smoke.values()}


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
    assert state is RunState.INVALID and len(problems) == 1


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
