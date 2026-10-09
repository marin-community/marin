# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from taskforge.loop.events import Terminal
from taskforge.queue.config import load_run_config, run_config
from taskforge.queue.job import SUMMARY_FILE, RunInputs, RunModel, run_job
from taskforge.queue.run import FailedItems
from taskforge.review.decision import BandOutcome

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "policy.example.json"


def config(tmp_path: Path, too_easy: str) -> dict:
    document = json.loads(EXAMPLE.read_text())
    document["root"] = str(tmp_path / "run")
    document["image_cache"] = str(tmp_path / "images")
    document["width"] = 2
    document["policy"]["band_rules"]["too_easy"] = {"repairs": 0, "then": too_easy}
    document["policy"]["validation"]["k"] = 2
    document["policy"]["validation"]["deadlines"] = {"total_turn_timeout": 30.0, "attempt_timeout": 60.0}
    return document


def test_the_committed_example_parses():
    assert load_run_config(EXAMPLE).policy.band_rules.too_easy.then == "accept"


async def test_a_run_exports_its_accepted_task_and_a_relaunch_runs_nothing_again(
    tmp_path, template_client, proposal_source, solver_models
):
    run = run_config(config(tmp_path, "accept"))
    model = RunModel(client=template_client, rollout_models=solver_models)

    def inputs(client, root):
        return RunInputs(ideas={"IDEA": "idea"}, source=proposal_source, describe_idea=lambda idea: {"idea": idea})

    summary = await run_job(run, inputs, FailedItems.SKIP, model)

    assert summary.items == {"IDEA--0": Terminal.ACCEPTED}
    accepted = summary.accepted["IDEA--0"]
    assert (accepted.band, accepted.solved, accepted.k) == (BandOutcome.TOO_EASY, 2, 2)
    assert json.loads((run.root / accepted.draft / "task.json").read_text())["id"] == "IDEA--0"
    exported = json.loads((run.root / SUMMARY_FILE).read_text())
    assert exported["accepted"]["IDEA--0"]["band"] == "too_easy"
    ledgers = {path: path.read_bytes() for path in (run.root / "ledger").iterdir()}
    calls = list(template_client.calls)

    again = await run_job(run, inputs, FailedItems.SKIP, model)

    assert again.items == summary.items and again.accepted == summary.accepted
    assert (proposal_source.calls, template_client.calls) == (1, calls)
    assert {path: path.read_bytes() for path in (run.root / "ledger").iterdir()} == ledgers


async def test_a_too_easy_task_is_rejected_under_the_reject_choice_and_not_exported(
    tmp_path, template_client, proposal_source, solver_models
):
    model = RunModel(client=template_client, rollout_models=solver_models)

    def inputs(client, root):
        return RunInputs(ideas={"IDEA": "idea"}, source=proposal_source, describe_idea=lambda idea: {"idea": idea})

    summary = await run_job(run_config(config(tmp_path, "reject")), inputs, FailedItems.SKIP, model)

    assert summary.items == {"IDEA--0": Terminal.REJECTED}
    assert summary.accepted == {}
