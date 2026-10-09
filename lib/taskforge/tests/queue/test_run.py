# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from taskforge.ledger.jsonl import read_entries
from taskforge.loop.events import Terminal
from taskforge.proposal.model import TaskProposal
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.queue.config import load_run_config, run_config
from taskforge.queue.job import SUMMARY_FILE, RunInputs, run_job
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


@pytest.mark.parametrize("rule", ["too_easy", "too_hard"])
def test_a_policy_that_asks_for_a_band_repair_is_refused_at_load(tmp_path, rule):
    document = config(tmp_path, "accept")
    document["policy"]["band_rules"][rule]["repairs"] = 1

    with pytest.raises(ValidationError, match="cannot repair"):
        run_config(document)


def inputs_for(template_client, proposal_source, solver_models):
    def inputs(root):
        return RunInputs(
            ideas={"IDEA": "idea"},
            source=proposal_source,
            describe_idea=lambda idea: {"idea": idea},
            model=template_client,
            rollout_models=solver_models,
        )

    return inputs


async def test_a_run_exports_its_accepted_task_and_a_relaunch_runs_nothing_again(
    tmp_path, template_client, proposal_source, solver_models
):
    run = run_config(config(tmp_path, "accept"))
    inputs = inputs_for(template_client, proposal_source, solver_models)

    summary = await run_job(run, inputs, FailedItems.SKIP)

    assert summary.items == {"IDEA--0": Terminal.ACCEPTED}
    accepted = summary.accepted["IDEA--0"]
    assert (accepted.band, accepted.solved, accepted.k) == (BandOutcome.TOO_EASY, 2, 2)
    assert json.loads((run.root / accepted.draft / "task.json").read_text())["id"] == "IDEA--0"
    exported = json.loads((run.root / SUMMARY_FILE).read_text())
    assert exported["accepted"]["IDEA--0"]["band"] == "too_easy"
    ledgers = {path: path.read_bytes() for path in (run.root / "ledger").iterdir()}
    calls = list(template_client.calls)

    again = await run_job(run, inputs, FailedItems.SKIP)

    assert again.items == summary.items and again.accepted == summary.accepted
    assert (proposal_source.calls, template_client.calls) == (1, calls)
    assert {path: path.read_bytes() for path in (run.root / "ledger").iterdir()} == ledgers


async def test_a_too_easy_task_is_rejected_under_the_reject_choice_and_not_exported(
    tmp_path, template_client, proposal_source, solver_models
):
    inputs = inputs_for(template_client, proposal_source, solver_models)

    summary = await run_job(run_config(config(tmp_path, "reject")), inputs, FailedItems.SKIP)

    assert summary.items == {"IDEA--0": Terminal.REJECTED}
    assert summary.accepted == {}


async def test_a_relaunch_under_another_policy_is_refused_and_keeps_the_stored_policy(
    tmp_path, template_client, proposal_source, solver_models
):
    inputs = inputs_for(template_client, proposal_source, solver_models)
    first = run_config(config(tmp_path, "accept"))
    await run_job(first, inputs, FailedItems.SKIP)
    stored = (first.root / "policy.json").read_bytes()
    changed = config(tmp_path, "accept")
    changed["policy"]["validation"]["k"] = 3
    second = run_config(changed)

    with pytest.raises(ValueError, match="use a new run root") as refusal:
        await run_job(second, inputs, FailedItems.SKIP)

    assert first.policy.digest in str(refusal.value) and second.policy.digest in str(refusal.value)
    assert (first.root / "policy.json").read_bytes() == stored


async def test_proposals_of_two_ideas_that_share_an_item_id_fail_the_run(
    tmp_path, template_client, proposal_source, solver_models
):
    run = run_config(config(tmp_path, "accept"))

    def inputs(root):
        shared = inputs_for(template_client, proposal_source, solver_models)(root)
        return RunInputs(**{**vars(shared), "ideas": {"A": "a", "B": "b"}})

    with pytest.raises(ValueError, match=r"IDEA--0 from ideas (A and B|B and A)"):
        await run_job(run, inputs, FailedItems.SKIP)


@dataclass
class HeldClient:
    """Answers like ``client`` but holds every build on its first call, after setting ``building``."""

    client: Any
    building: asyncio.Event

    @property
    def endpoint(self) -> Any:
        return self.client.endpoint

    async def structured(self, messages, output_type, name):
        self.building.set()
        await asyncio.Event().wait()


@dataclass
class GatedSource:
    """Proposes ``proposal`` for every idea; idea ``"b"`` proposes only once ``building`` is set."""

    proposal: TaskProposal
    building: asyncio.Event

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        if idea == "b":
            await self.building.wait()
        return ProposalBatch(planning_request=(), planning=(), slots=(SlotProposal(0, self.proposal, (), (), None),))


async def test_a_later_idea_that_reuses_a_running_item_id_fails_the_run_and_cancels_the_item(
    tmp_path, template_client, proposal, solver_models
):
    run = run_config(config(tmp_path, "accept"))
    building = asyncio.Event()

    def inputs(root):
        return RunInputs(
            ideas={"A": "a", "B": "b"},
            source=GatedSource(proposal, building),
            describe_idea=lambda idea: {"idea": idea},
            model=HeldClient(template_client, building),
            rollout_models=solver_models,
        )

    with pytest.raises(ValueError, match=r"IDEA--0 from ideas A and B"):
        await run_job(run, inputs, FailedItems.SKIP)

    steps = [entry.step for entry in read_entries(run.root / "ledger" / "IDEA--0.jsonl")]
    assert "opened" in steps and "terminal" not in steps
    assert asyncio.all_tasks() == {asyncio.current_task()}
