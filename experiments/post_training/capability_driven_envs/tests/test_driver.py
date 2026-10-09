# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in Taskforge's own environment (lib/taskforge); the root environment skips them.
pytest.importorskip("taskforge")

import json
from pathlib import Path

from taskforge.llm.recording import CallLedger
from taskforge.loop.events import Terminal
from taskforge.queue.job import SUMMARY_FILE
from taskforge.queue.run import FailedItems
from taskforge.review.decision import BandOutcome
from taskforge.validate.trials import RolloutModel

from experiments.post_training.capability_driven_envs import driver
from experiments.post_training.capability_driven_envs.catalog import (
    capability_prompt_record,
    load_capability_ideas,
)
from experiments.post_training.capability_driven_envs.scaling_task import (
    scaling_problem,
    scripted_solver,
)

CAPABILITY = "d43.culinary.scaling"
DATA = Path(driver.__file__).resolve().parent / "data"
ITEM = f"{CAPABILITY}--0"


def test_the_catalog_idea_shows_the_models_its_recorded_capability_record():
    idea = load_capability_ideas(driver.CATALOG)[CAPABILITY]
    assert capability_prompt_record(idea) == json.loads((DATA / f"{CAPABILITY}.json").read_text())


async def test_a_capability_idea_becomes_an_accepted_task_in_the_band(tmp_path):
    ideas = driver.selected_ideas(driver.CATALOG, [CAPABILITY])

    summary = await driver.run(tmp_path, ideas, driver.capability_policy(4, 1), scripted_solver, FailedItems.SKIP)

    assert summary.items == {ITEM: Terminal.ACCEPTED}
    accepted = summary.accepted[ITEM]
    assert (accepted.band, accepted.solved, accepted.k) == (BandOutcome.IN_BAND, 2, 4)
    task = json.loads((tmp_path / accepted.draft / "task.json").read_text())
    assert task["grader"]["parameters"]["expected"] == str(scaling_problem(f"{CAPABILITY}/0").answer)
    assert json.loads((tmp_path / SUMMARY_FILE).read_text())["accepted"][ITEM]["band"] == "in_band"


async def test_a_task_every_trial_solves_is_accepted_as_too_easy_under_the_capability_policy(tmp_path):
    def always_scales(record: CallLedger) -> RolloutModel:
        return scripted_solver(CallLedger(record.ledger, record.item_id, record.round, "solver/0"))

    ideas = driver.selected_ideas(driver.CATALOG, [CAPABILITY])

    summary = await driver.run(tmp_path, ideas, driver.capability_policy(4, 1), always_scales, FailedItems.SKIP)

    assert summary.accepted[ITEM].band is BandOutcome.TOO_EASY
