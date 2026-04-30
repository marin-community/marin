# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Golden tests for the controller replay scenarios.

For each scenario: run against a fresh ``ControllerDB`` with a frozen
clock, dump the DB deterministically, and assert the dump matches the
committed ``golden/<scenario>.json`` byte-for-byte.
"""

import json
import tempfile
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions
from rigging.timing import Timestamp

from tests.cluster.controller.replay.db_dump import deterministic_dump
from tests.cluster.controller.replay.scenarios import SCENARIO_NAMES, SCENARIOS, frozen_clock

GOLDEN_DIR = Path(__file__).parent / "golden"


def _run(name: str) -> dict:
    # DB construction and migrations run with real time; the frozen clock
    # wraps scenario execution only so goldens encode deterministic
    # timestamps without mismatches from startup-path ``Timestamp.now()``
    # calls.
    with tempfile.TemporaryDirectory(prefix=f"iris-replay-test-{name}-") as db_dir_str:
        db = ControllerDB(db_dir=Path(db_dir_str))
        try:
            store = ControllerStore(db)
            transitions = ControllerTransitions(store)
            with frozen_clock() as clock:
                SCENARIOS[name](transitions, clock)
            return deterministic_dump(db)
        finally:
            db.close()


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
