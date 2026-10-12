# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import json
from pathlib import Path

from taskforge.loop.events import Terminal
from taskforge.review.decision import BandOutcome

from experiments.post_training.capability_driven_envs import driver
from experiments.post_training.capability_driven_envs.scaling_task import (
    scaling_problem,
    scripted_solver,
)

CAPABILITY = "d43.culinary.scaling"
ITEM = f"{CAPABILITY}--0"
SECOND = f"{CAPABILITY}.again"


async def test_a_capability_idea_becomes_an_accepted_task_in_the_band(tmp_path):
    ideas = driver.selected_ideas(driver.CATALOG, [CAPABILITY])

    summary = await driver.run(tmp_path, ideas, driver.capability_policy(4, 1), scripted_solver)

    assert summary.items == {ITEM: Terminal.ACCEPTED}
    accepted = summary.accepted[ITEM]
    assert accepted.band is BandOutcome.IN_BAND
    task = json.loads((tmp_path / accepted.draft / "task.json").read_text())
    assert task["grader"]["parameters"]["expected"] == str(scaling_problem(f"{CAPABILITY}/0").answer)
    assert json.loads((tmp_path / driver.SUMMARY_FILE).read_text())["accepted"][ITEM]["band"] == "in_band"


def two_capability_catalog(path: Path) -> Path:
    """The catalog with a copy of its capability under ``SECOND``, written to ``path``."""
    document = json.loads(driver.CATALOG.read_text())
    sections = document["curricula"][0]["curriculum"]["sections"]
    sections.append({**sections[0], "id": SECOND})
    path.write_text(json.dumps(document))
    return path


async def test_the_summary_lists_the_items_of_every_run_on_the_root(tmp_path):
    catalog = two_capability_catalog(tmp_path / "catalog.json")
    root = tmp_path / "run"
    policy = driver.capability_policy(4, 1)
    await driver.run(root, driver.selected_ideas(catalog, [CAPABILITY]), policy, scripted_solver)

    summary = await driver.run(root, driver.selected_ideas(catalog, [SECOND]), policy, scripted_solver)

    both = {ITEM: Terminal.ACCEPTED, f"{SECOND}--0": Terminal.ACCEPTED}
    assert summary.items == both
    assert summary.accepted.keys() == both.keys()
    assert json.loads((root / driver.SUMMARY_FILE).read_text())["items"] == {item: "accepted" for item in both}
