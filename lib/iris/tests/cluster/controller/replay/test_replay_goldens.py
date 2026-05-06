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
from iris.cluster.controller.db import ACTIVE_TASK_STATES, ControllerDB
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import TERMINAL_TASK_STATES
from rigging.timing import Timestamp

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
            store = ControllerStore(db)
            transitions = ControllerTransitions(store)
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

    terminal = ",".join(str(s) for s in sorted(TERMINAL_TASK_STATES))
    active = ",".join(str(s) for s in sorted(ACTIVE_TASK_STATES))

    def check(db: ControllerDB) -> None:
        with db.read_snapshot() as snap:
            rows = snap.fetchall(
                f"SELECT ta.task_id, ta.attempt_id, ta.state AS attempt_state, t.state AS task_state, "
                f"ta.finished_at_ms FROM task_attempts ta JOIN tasks t ON t.task_id = ta.task_id "
                f"WHERE t.state IN ({terminal}) AND ta.state IN ({active}) "
                f"AND ta.finished_at_ms IS NULL"
            )
        orphans = [
            f"{row['task_id']} attempt={row['attempt_id']} "
            f"task_state={row['task_state']} attempt_state={row['attempt_state']}"
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

    active = ",".join(str(s) for s in sorted(ACTIVE_TASK_STATES))

    def check(db: ControllerDB) -> None:
        with db.read_snapshot() as snap:
            rows = snap.fetchall(
                f"SELECT j.job_id, "
                f"  SUM(CASE WHEN t.state IN ({active}) THEN 1 ELSE 0 END) AS active_count, "
                f"  SUM(CASE WHEN t.state = 1 THEN 1 ELSE 0 END) AS pending_count, "
                f"  COUNT(*) AS task_count "
                f"FROM jobs j "
                f"JOIN job_config jc ON jc.job_id = j.job_id "
                f"JOIN tasks t ON t.job_id = j.job_id "
                f"WHERE jc.has_coscheduling = 1 AND j.is_reservation_holder = 0 "
                f"GROUP BY j.job_id "
                f"HAVING active_count > 0 AND pending_count > 0"
            )
        split = [
            f"{row['job_id']} active={row['active_count']} pending={row['pending_count']} " f"total={row['task_count']}"
            for row in rows
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
