# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent-engine scheduling preserves source order, accounting and solver fairness."""

from dataclasses import replace

from finestore.rl.curation import SourceUse
from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty.budget import campaign_budget
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, execute_plan
from experiments.post_training.task_curation.difficulty.ledger import read_rows
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, plan
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, Outcome, cell_state
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE


def test_unlimited_ruling_preserves_accounting_epoch_and_admits_previously_over_budget_batch(tmp_path):
    ledger = str(tmp_path / "ledger")
    original = campaign_budget(ledger, 0.01)
    actions = plan([cell_state([], SOURCE, Solver.SMALL, CoveragePolicy())], [])
    backend = JobStore(str(tmp_path / "jobs"))
    notifications = EscalationLog(ledger)
    assert execute_plan(actions, [], [SOURCE], ledger, backend, notifications, original)["submitted"] == []
    unlimited = campaign_budget(ledger, None)
    assert unlimited.started_at == original.started_at
    assert execute_plan(actions, [], [SOURCE], ledger, backend, notifications, unlimited)["submitted"]
    assert campaign_budget(ledger, None) == unlimited


def test_train_sources_dispatch_before_eval_sources():
    train = replace(SOURCE, source_id="train-source", intended_use=SourceUse.TRAIN)
    evaluation = replace(SOURCE, source_id="eval-source", intended_use=SourceUse.EVAL)
    actions = plan([cell_state([], source, Solver.SMALL, CoveragePolicy()) for source in [evaluation, train]], [])
    attempts = [action for action in actions if isinstance(action, AttemptBatch)]
    assert [action.sources[0].source_id for action in attempts] == ["train-source", "eval-source"]


def test_one_solver_cannot_fill_all_attempt_lanes(tmp_path):
    sources = [replace(SOURCE, source_id=f"source-{i}") for i in range(6)]
    cells = [
        cell_state([], source, Solver.SMALL if i < 5 else Solver.LARGE, CoveragePolicy())
        for i, source in enumerate(sources)
    ]
    ledger = str(tmp_path / "ledger")
    actions = plan(cells, [])
    result = execute_plan(actions, [], sources, ledger, JobStore(str(tmp_path / "jobs")), EscalationLog(ledger))
    submitted = [row for row in read_rows(ledger) if row.outcome == Outcome.SUBMITTED]
    assert len(result["submitted"]) == 5
    assert sum(row.solver == Solver.SMALL for row in submitted) == 4
    assert sum(row.solver == Solver.LARGE for row in submitted) == 1
