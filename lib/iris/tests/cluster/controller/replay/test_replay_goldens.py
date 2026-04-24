# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Golden tests for the controller replay scenarios.

For each scenario the test runs the scenario against a fresh
``ControllerDB`` with the SQL trace hook enabled, then asserts the DB
state matches the committed ``db.json`` byte-for-byte. SQL trace
differences are reported as warnings — they're expected to drift
mechanically across SQL refactors and aren't a regression signal.
"""

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.replay.db_dump import deterministic_dump
from iris.cluster.controller.replay.scenarios import SCENARIO_NAMES, run_scenario
from iris.cluster.controller.replay.sql_trace import sql_tracing
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions

GOLDEN_DIR = Path(__file__).parent / "golden"


def _run(name: str) -> tuple[dict, list[str]]:
    sql_log: list[str] = []
    with sql_tracing(sql_log):
        with tempfile.TemporaryDirectory(prefix=f"iris-replay-test-{name}-") as db_dir_str:
            db = ControllerDB(db_dir=Path(db_dir_str))
            try:
                store = ControllerStore(db)
                transitions = ControllerTransitions(store)
                run_scenario(name, transitions)
                dump = deterministic_dump(db)
            finally:
                db.close()
    return dump, sql_log


@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_scenario_matches_golden(scenario_name: str, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    db_dump, sql_log = _run(scenario_name)

    db_actual = json.dumps(db_dump, indent=2, sort_keys=True) + "\n"
    sql_actual = "\n".join(sql_log) + "\n"

    golden_dir = GOLDEN_DIR / scenario_name
    db_golden = golden_dir / "db.json"
    sql_golden = golden_dir / "sql.txt"

    if request.config.getoption("--update-goldens"):
        golden_dir.mkdir(parents=True, exist_ok=True)
        db_golden.write_text(db_actual)
        sql_golden.write_text(sql_actual)
        return

    # DB state is the strict bar.
    assert db_golden.exists(), f"missing golden {db_golden} (run with --update-goldens)"
    expected = db_golden.read_text()
    if expected != db_actual:
        actual_path = tmp_path / "db_actual.json"
        actual_path.write_text(db_actual)
        pytest.fail(
            f"db.json drift for scenario {scenario_name!r}:\n" f"  expected: {db_golden}\n" f"  actual:   {actual_path}",
        )

    # SQL trace is informational — surface drift without failing the test.
    if sql_golden.exists():
        sql_expected = sql_golden.read_text()
        if sql_expected != sql_actual:
            actual_path = tmp_path / "sql_actual.txt"
            actual_path.write_text(sql_actual)
            warnings.warn(
                f"sql.txt drift for scenario {scenario_name!r}: see {actual_path} vs {sql_golden}",
                stacklevel=2,
            )
