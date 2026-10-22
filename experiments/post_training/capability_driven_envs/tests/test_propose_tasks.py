# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import re
from pathlib import Path

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
    return f"""---
id: "{CAPABILITY_ID}/1"
source: {{kind: capability, ref: "{CAPABILITY_ID}", hash: "{capability_hash}"}}
grounding: unverified
null_reason: null
environment: reasoning
solver_machine: shellsim
verification: code
research_need: not_needed
research_reason: "the recipe and yields are synthetic"
research: []
build:
  - "grader": "the private checker for the scaled quantities"
resources: ["grader/check.py"]
---
{body}"""


def test_a_capability_is_proposed_and_each_proposal_triaged_by_the_preflight_without_a_model_call(tmp_path: Path):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY_ID])
    model = ScriptedModel()
    model.reply(json.dumps(PLAN))
    model.reply(proposal_document(ideas[CAPABILITY_ID].capability_hash))
    output = tmp_path / "run"
    output.mkdir()

    proposed = asyncio.run(propose_tasks(model, ideas, 1, REASONING_ONLY, LLMPolicy()))
    write_outputs(output, proposed)

    assert json.loads((output / SUMMARY_FILE).read_text()) == {CAPABILITY_ID: {"1": TriageDecision.ACCEPT}}
    assert len(model.requests) == 2
    directory = output / CAPABILITY_ID
    assert parse((directory / "slot-01.md").read_text()).header.id == f"{CAPABILITY_ID}/1"
    verdict = Verdict.from_json((directory / "slot-01.verdict.json").read_text())
    assert (verdict.decision, verdict.reasons) == (TriageDecision.ACCEPT, ())
    assert len(json.loads((directory / "plan.completions.json").read_text())) == 1


def test_a_plan_invalid_after_its_repair_fails_only_its_capability(tmp_path: Path):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY_ID])
    model = ScriptedModel()
    unrunnable = json.dumps({**PLAN, "slots": [{**SLOT, "environment": "container", "solver_machine": None}]})
    model.reply(unrunnable)
    model.reply(unrunnable)
    output = tmp_path / "run"
    output.mkdir()

    proposed = asyncio.run(propose_tasks(model, ideas, 1, REASONING_ONLY, LLMPolicy()))
    write_outputs(output, proposed)

    summary = json.loads((output / SUMMARY_FILE).read_text())
    assert "slot 1 is proposed in container" in summary[CAPABILITY_ID]
    directory = output / CAPABILITY_ID
    assert sorted(path.name for path in directory.iterdir()) == ["plan.completions.json", "plan.request.json"]
    assert json.loads((directory / "plan.request.json").read_text()) == model.requests[0].messages
    completions = json.loads((directory / "plan.completions.json").read_text())
    assert [completion["content"] for completion in completions] == [unrunnable, unrunnable]


def test_a_slot_whose_model_runs_out_of_retries_fails_alone_and_keeps_the_plan_and_its_sibling(tmp_path: Path):
    ideas = selected_ideas(example_catalog_path(), [CAPABILITY_ID])
    unavailable = ModelUnavailable("the router answered 503 on every retry", ())
    model = ScriptedModel()
    model.reply(json.dumps({**PLAN, "slots": [SLOT, {**SLOT, "slot": 2, "verification": "judge"}]}))
    model.reply(proposal_document(ideas[CAPABILITY_ID].capability_hash))
    model.fail(unavailable)
    output = tmp_path / "run"
    output.mkdir()

    proposed = asyncio.run(propose_tasks(model, ideas, 2, REASONING_ONLY, LLMPolicy()))
    write_outputs(output, proposed)

    summary = json.loads((output / SUMMARY_FILE).read_text())
    assert summary == {CAPABILITY_ID: {"1": TriageDecision.ACCEPT, "2": str(unavailable)}}
    directory = output / CAPABILITY_ID
    assert len(json.loads((directory / "plan.completions.json").read_text())) == 1
    assert parse((directory / "slot-01.md").read_text()).header.id == f"{CAPABILITY_ID}/1"
    assert (directory / "slot-02.failure.txt").read_text() == str(unavailable)
    assert json.loads((directory / "slot-02.completions.json").read_text()) == []
    assert not (directory / "slot-02.md").exists()


def test_an_unknown_capability_is_refused_before_the_run_starts():
    with pytest.raises(ValueError, match=re.escape("['d09.missing']")):
        selected_ideas(example_catalog_path(), [CAPABILITY_ID, "d09.missing"])
