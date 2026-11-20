# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger-derived coverage and cost reports."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime

from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty.planner import Action, AttemptBatch, action_id, action_payload
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CellState,
    CoveragePolicy,
    LedgerRow,
    Outcome,
    cell_state,
)


def cells_for_catalog(
    rows: Sequence[LedgerRow], catalog: Sequence[CatalogSource], policy: CoveragePolicy
) -> list[CellState]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.source_id, row.solver)].append(row)
    return [
        cell_state(grouped[(source.source_id, solver)], source, solver, policy)
        for source in catalog
        for solver in Solver
    ]


def status_report(
    rows: Sequence[LedgerRow],
    catalog: Sequence[CatalogSource],
    policy: CoveragePolicy,
    since: datetime | None = None,
    actions: Sequence[Action] = (),
) -> dict:
    cells = cells_for_catalog(rows, catalog, policy)
    by_cell = defaultdict(list)
    for action in actions:
        payload = action_payload(action)
        if "sources" in payload:
            keys = [(source["source_id"], payload["solver"]) for source in payload["sources"]]
        elif "source" in payload:
            keys = [(payload["source"]["source_id"], payload["solver"])]
        else:
            keys = payload["cells"]
        for source_id, solver in keys:
            by_cell[(source_id, solver)].append({"action_id": action_id(action), "kind": payload["kind"]})
    solvers = {}
    for solver in Solver:
        selected = [cell for cell in cells if cell.solver == solver]
        recent = [
            row
            for row in rows
            if row.solver == solver
            and (since is None or datetime.fromisoformat(row.recorded_at).astimezone(UTC) >= since)
        ]
        solvers[solver.value] = {
            "cells": dict(Counter(cell.state for cell in selected)),
            "missing_attempts": sum(cell.n_missing for cell in selected),
            "scored_attempts_since": sum(row.outcome == Outcome.SCORED for row in recent),
            "gpu_hours_since": round(sum(row.gpu_hours for row in recent), 6),
            "cost_basis": (
                "candidate reservation windows; startup, teardown, judge serving and interrupted windows excluded"
            ),
        }
    return {
        "solvers": solvers,
        "since": since.isoformat() if since else None,
        "cells": [
            {
                "source_id": cell.source.source_id,
                "solver": cell.solver,
                "state": cell.state,
                "n_missing": cell.n_missing,
                "gaps": [asdict(gap) for gap in cell.gaps],
                "job_ids": cell.job_ids,
                "actions": by_cell[(cell.source.source_id, cell.solver)],
            }
            for cell in cells
        ],
    }


def plan_report(actions: Sequence[Action]) -> dict:
    return {
        "summary": dict(Counter(type(action).__name__ for action in actions)),
        "max_gpu_hours": sum(action.max_gpu_hours for action in actions if isinstance(action, AttemptBatch)),
        "actions": [{"action_id": action_id(action), **action_payload(action)} for action in actions],
    }
