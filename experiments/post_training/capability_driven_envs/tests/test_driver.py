# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import json

from taskforge.loop.events import Terminal
from taskforge.review.decision import BandOutcome

from experiments.post_training.capability_driven_envs import driver
from experiments.post_training.capability_driven_envs.scaling_task import (
    scaling_problem,
    scripted_solver,
)

CAPABILITY = "d43.culinary.scaling"
ITEM = f"{CAPABILITY}--0"


async def test_a_capability_idea_becomes_an_accepted_task_in_the_band(tmp_path):
    ideas = driver.selected_ideas(driver.CATALOG, [CAPABILITY])

    summary = await driver.run(tmp_path, ideas, driver.capability_policy(4, 1), scripted_solver)

    assert summary.items == {ITEM: Terminal.ACCEPTED}
    accepted = summary.accepted[ITEM]
    assert accepted.band is BandOutcome.IN_BAND
    task = json.loads((tmp_path / accepted.draft / "task.json").read_text())
    assert task["grader"]["parameters"]["expected"] == str(scaling_problem(f"{CAPABILITY}/0").answer)
    assert json.loads((tmp_path / driver.SUMMARY_FILE).read_text())["accepted"][ITEM]["band"] == "in_band"
