# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.belay.submit import tasks_started

GATED = """Job: /loom/belay-X0229v2-seed0-coord/grug-train-belay-X0229v2-seed0
State: running  exit=0  failures=0  preemptions=0
Tasks: 0/1 completed  building=1

  TASK  STATE     EXIT    DURATION    PEAK MEM    CUR MEM    DIAGNOSTIC
     0  building  -       -           -           -          SchedulingGated: Scheduling is blocked due to non-empty
"""


def test_a_run_whose_task_waits_for_a_node_has_not_started() -> None:
    assert not tasks_started(GATED)
    assert tasks_started(GATED.replace("building=1", "running=1"))
    assert tasks_started(GATED.replace("0/1 completed  building=1", "1/1 completed"))


def test_describe_output_without_a_task_summary_is_an_error() -> None:
    with pytest.raises(ValueError):
        tasks_started("Job: /loom/x\nState: running\n")
