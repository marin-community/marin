# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The proposal driver: each capability's plan, slot documents, triage verdicts and failures on disk.

Invariants: none (pins the driver's output files and per-capability and per-slot failure isolation).
"""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import re
from pathlib import Path

from taskforge.defaults import DEFAULT_POLICY
from taskforge.llm.client import ModelUnavailable
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import REQUIRED_HEADINGS, Environment, parse
from taskforge.stage20_triage.verdict import TriageDecision, Verdict

from experiments.post_training.capability_driven_envs.catalog import example_catalog_path
from experiments.post_training.capability_driven_envs.generate_tasks import SUMMARY_FILE, selected_ideas
from experiments.post_training.capability_driven_envs.propose_tasks import propose_tasks, write_outputs
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel

CAPABILITY_ID = "d43.culinary.scaling"
SECTION = "A cook rescales the stock recipe for the banquet and records every quantity in its original unit. " * 4
SLOT = {
    "slot": 1,
    "title": "banquet scaling",
    "workflow": "banquet production",
    "environment": "reasoning",
    "solver_machine": "shellsim",
    "verification": "code",
    "distinctive_challenge": "trim loss after scaling",
    "research_need": "not_needed",
    "research_reason": "the recipe and yields are synthetic",
    "status": "propose",
    "reason": None,
}
PLAN = {"coverage_rationale": "one slot", "excluded_combinations": [], "slots": [SLOT], "research_priorities": []}
REASONING_ONLY = frozenset({Environment.REASONING})


def proposal_document(capability_hash: str) -> str:
    body = "\n".join(f"## {heading}\n{SECTION}\n" for heading in REQUIRED_HEADINGS)
    return json.dumps(
        {
            "header": {
                "id": f"{CAPABILITY_ID}/1",
                "source": {"kind": "capability", "ref": CAPABILITY_ID, "hash": capability_hash},
                "grounding": "unverified",
                "null_reason": None,
                "environment": "reasoning",
                "solver_machine": "shellsim",
                "verification": "code",
                "research_need": "not_needed",
                "research_reason": "the recipe and yields are synthetic",
                "research": [],
                "build": [{"name": "grader", "description": "the private checker for the scaled quantities"}],
                "resources": ["grader/check.py"],
            },
            "body": body,
        }
    )


def propose(tmp_path: Path, n: int, *replies: str | Exception) -> tuple[ScriptedModel, Path, dict]:
    """Run the driver for ``n`` slots of ``CAPABILITY_ID`` on the scripted ``replies``; return the model, the
    capability's directory and the summary."""
    model = ScriptedModel()
    for reply in replies:
        if isinstance(reply, Exception):
            model.fail(reply)
        else:
            model.reply(reply)
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY_ID])
    proposed = asyncio.run(propose_tasks(model, ideas, n, REASONING_ONLY, LLMPolicy(), DEFAULT_POLICY))
    write_outputs(tmp_path, proposed)
    return model, tmp_path / CAPABILITY_ID, json.loads((tmp_path / SUMMARY_FILE).read_text())


DOCUMENT = proposal_document(selected_ideas(example_catalog_path(), [CAPABILITY_ID])[CAPABILITY_ID].capability_hash)


def test_a_capability_is_proposed_and_each_proposal_triaged_by_the_preflight_without_a_model_call(tmp_path: Path):
    model, directory, summary = propose(tmp_path, 1, json.dumps(PLAN), DOCUMENT)

    assert summary == {CAPABILITY_ID: {"1": TriageDecision.ACCEPT}}
    assert len(model.requests) == 2
    assert parse((directory / "slot-01.json").read_text()).header.id == f"{CAPABILITY_ID}/1"
    verdict = Verdict.from_json((directory / "slot-01.verdict.json").read_text())
    assert (verdict.decision, verdict.reasons) == (TriageDecision.ACCEPT, ())
    assert len(json.loads((directory / "plan.completions.json").read_text())) == 1


def test_a_plan_invalid_after_its_repair_fails_only_its_capability(tmp_path: Path):
    unrunnable = json.dumps({**PLAN, "slots": [{**SLOT, "environment": "container", "solver_machine": None}]})
    model, directory, summary = propose(tmp_path, 1, unrunnable, unrunnable)

    assert "slot 1 is proposed in container" in summary[CAPABILITY_ID]
    assert sorted(path.name for path in directory.iterdir()) == ["plan.completions.json", "plan.request.json"]
    assert json.loads((directory / "plan.request.json").read_text()) == model.requests[0].messages
    completions = json.loads((directory / "plan.completions.json").read_text())
    assert [completion["content"] for completion in completions] == [unrunnable, unrunnable]


def test_a_slot_whose_model_runs_out_of_retries_fails_alone_and_keeps_the_plan_and_its_sibling(tmp_path: Path):
    unavailable = ModelUnavailable("the router answered 503 on every retry", ())
    plan = json.dumps({**PLAN, "slots": [SLOT, {**SLOT, "slot": 2, "verification": "judge"}]})
    _, directory, summary = propose(tmp_path, 2, plan, DOCUMENT, unavailable)

    assert summary == {CAPABILITY_ID: {"1": TriageDecision.ACCEPT, "2": str(unavailable)}}
    assert len(json.loads((directory / "plan.completions.json").read_text())) == 1
    assert parse((directory / "slot-01.json").read_text()).header.id == f"{CAPABILITY_ID}/1"
    assert (directory / "slot-02.failure.txt").read_text() == str(unavailable)
    assert json.loads((directory / "slot-02.completions.json").read_text()) == []
    assert not (directory / "slot-02.json").exists()


def test_an_unknown_capability_is_refused_before_the_run_starts():
    with pytest.raises(ValueError, match=re.escape("['d09.missing']")):
        selected_ideas(example_catalog_path(), [CAPABILITY_ID, "d09.missing"])
