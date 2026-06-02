# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Golden tests for the controller replay scenarios.

For each scenario: run against a fresh ``ControllerDB`` with a frozen
clock, dump the DB deterministically, and assert the dump matches the
committed ``golden/<scenario>.json`` byte-for-byte.
"""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.types import TERMINAL_TASK_STATES
from rigging.timing import Timestamp
from sqlalchemy import Integer, case, func, literal, select

from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.replay.db_dump import deterministic_dump
from tests.cluster.controller.replay.scenarios import SCENARIO_NAMES, SCENARIOS, frozen_clock

GOLDEN_DIR = Path(__file__).parent / "golden"


def _with_scenario(name: str, fn: Callable[[ControllerDB], None]) -> None:
    """Run a scenario in a fresh DB, then invoke ``fn`` with the live DB.

    DB construction and migrations run with real time; the frozen clock
    wraps scenario execution only so goldens encode deterministic
    timestamps without mismatches from startup-path ``Timestamp.now()``
    calls.
    """
    with tempfile.TemporaryDirectory(prefix=f"iris-replay-test-{name}-") as db_dir_str:
        db = ControllerDB(db_dir=Path(db_dir_str))
        try:
            transitions = ControllerTestState(db)
            with frozen_clock() as clock:
                SCENARIOS[name](transitions, clock)
            fn(db)
        finally:
            db.close()


def _run(name: str) -> dict:
    result: dict = {}

    def capture(db: ControllerDB) -> None:
        result.update(deterministic_dump(db))

    _with_scenario(name, capture)
    return result


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_scenario_matches_golden(
    scenario_name: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    dump = _run(scenario_name)
    actual = json.dumps(dump, indent=2, sort_keys=True) + "\n"

    golden = GOLDEN_DIR / f"{scenario_name}.json"

    if request.config.getoption("--update-goldens"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
        return

    assert golden.exists(), f"missing golden {golden} (run with --update-goldens)"
    expected = golden.read_text()
    if expected != actual:
        actual_path = tmp_path / f"{scenario_name}.actual.json"
        actual_path.write_text(actual)
        pytest.fail(
            f"golden drift for scenario {scenario_name!r}:\n" f"  expected: {golden}\n" f"  actual:   {actual_path}",
        )


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_no_orphan_task_attempts(scenario_name: str) -> None:
    """Invariant: no scenario leaves a task_attempts row active under a terminal task.

    The original ``cancel_job`` bug (30K+ orphan rows in production) would have
    been caught immediately if any scenario terminating a task had run this
    check post-hoc. The class of bug is general — any termination path that
    forgets to finalize ``task_attempts`` after marking ``tasks`` terminal will
    fail this assertion.
    """

    terminal = list(sorted(TERMINAL_TASK_STATES))
    active = list(sorted(ACTIVE_TASK_STATES))

    ta = task_attempts_table.alias("ta")
    t = tasks_table.alias("t")

    def check(db: ControllerDB) -> None:
        with db.read_snapshot() as snap:
            rows = snap.execute(
                select(
                    ta.c.task_id,
                    ta.c.attempt_id,
                    ta.c.state.label("attempt_state"),
                    t.c.state.label("task_state"),
                    ta.c.finished_at_ms,
                )
                .join(t, t.c.task_id == ta.c.task_id)
                .where(
                    t.c.state.in_(terminal),
                    ta.c.state.in_(active),
                    ta.c.finished_at_ms.is_(None),
                )
            ).all()
        orphans = [
            f"{row.task_id} attempt={row.attempt_id} " f"task_state={row.task_state} attempt_state={row.attempt_state}"
            for row in rows
        ]
        assert not orphans, (
            f"scenario {scenario_name!r} left {len(orphans)} orphan task_attempts (terminal task, "
            f"active+unfinished attempt): {orphans}"
        )

    _with_scenario(scenario_name, check)


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_no_split_coscheduled_active_tasks(scenario_name: str) -> None:
    """Invariant: no coscheduled job ends a scenario with siblings spanning more than
    one assignment group (currently keyed by current_worker_id; in production it
    would be md_tpu_name, but the scenarios use plain CPU workers).

    A coscheduled job in a partial-PENDING state — e.g. one task RUNNING on worker
    A while another is PENDING waiting to be re-assigned — is the precondition for
    the split-slice bug: the next scheduling pass can place the lone PENDING task
    on a fresh worker and break the SPMD mesh. This invariant flags that
    precondition; for any scenario where it shouldn't happen, the test fails.
    """

    active_states = list(sorted(ACTIVE_TASK_STATES))
    PENDING_STATE = 1

    j = jobs_table.alias("j")
    jc = job_config_table.alias("jc")
    t = tasks_table.alias("t")

    active_count_col = func.sum(
        case((t.c.state.in_(active_states), literal(1, Integer)), else_=literal(0, Integer))
    ).label("active_count")
    pending_count_col = func.sum(
        case((t.c.state == PENDING_STATE, literal(1, Integer)), else_=literal(0, Integer))
    ).label("pending_count")
    task_count_col = func.count().label("task_count")

    stmt = (
        select(j.c.job_id, active_count_col, pending_count_col, task_count_col)
        .join(jc, jc.c.job_id == j.c.job_id)
        .join(t, t.c.job_id == j.c.job_id)
        .where(jc.c.has_coscheduling == 1, j.c.is_reservation_holder == 0)
        .group_by(j.c.job_id)
        .having(active_count_col > 0, pending_count_col > 0)
    )

    def check(db: ControllerDB) -> None:
        with db.read_snapshot() as snap:
            rows = snap.execute(stmt).all()
        split = [
            f"{row.job_id} active={row.active_count} pending={row.pending_count} total={row.task_count}" for row in rows
        ]
        assert not split, (
            f"scenario {scenario_name!r} left {len(split)} coscheduled job(s) in a partial-PENDING "
            f"state — siblings split between active and pending. Next scheduling pass can land the "
            f"lone PENDING task on a different slice, splitting the SPMD mesh: {split}"
        )

    _with_scenario(scenario_name, check)


def test_frozen_clock_restores_classmethod_descriptor() -> None:
    """``frozen_clock`` must restore the exact ``classmethod`` descriptor, not a bound method.

    Regression: a previous implementation saved ``Timestamp.now`` via
    attribute access, which triggered the descriptor protocol and
    captured a bound method. On teardown it assigned that bound method
    back to the class, leaving ``Timestamp.now`` as a plain method for
    the rest of the test process. Subclass calls stopped binding to
    the subclass, and parallel tests ran with a corrupted
    ``Timestamp`` class.
    """
    original = Timestamp.__dict__["now"]
    assert isinstance(original, classmethod)

    with frozen_clock():
        pass

    assert Timestamp.__dict__["now"] is original, (
        "frozen_clock replaced Timestamp.now with a different descriptor; "
        "the save/restore path must use ``__dict__`` to avoid the "
        "descriptor-protocol capture that converts the classmethod into a "
        "bound method."
    )
