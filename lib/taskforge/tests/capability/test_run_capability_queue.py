# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``scripts/run_capability_queue.py``: a catalog's capabilities carried through ``queue.job.run_job``."""

import importlib.util
import json
import re
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from taskforge.llm.client import Pool
from taskforge.loop.events import Terminal
from taskforge.proposal.model import REQUIRED_HEADINGS
from taskforge.queue.config import LaptopGlm, load_run_config
from taskforge.queue.job import SUMMARY_FILE, run_job
from taskforge.queue.run import FailedItems
from taskforge.sandbox.factories import MachineHost

TASKFORGE = Path(__file__).parents[2]
SCRIPT = TASKFORGE / "scripts" / "run_capability_queue.py"
EXAMPLE = TASKFORGE / "docs" / "policy.example.json"
CAPABILITY_ID = "d01.algebra.linear-transformations"
SECTION = "A technician reconciles the stock sheet against the labels and records the total in millilitres. " * 4
REVIEW = {
    "realism": 2,
    "alignment": 2,
    "specificity": 2,
    "reward_validity": 2,
    "environment_fit": 2,
    "diversity": 2,
    "source_honesty": 2,
    "critical_failures": ["the task does not exercise linear transformations"],
    "issues": [],
    "required_changes": [],
    "recommendation": "reject",
}


def load_script():
    spec = importlib.util.spec_from_file_location("run_capability_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog(path: Path) -> Path:
    capability = {"id": CAPABILITY_ID, "kind": "capability", "name": "Linear transformations", "outcome": "Maps."}
    other = {**capability, "id": "d01.algebra.other", "name": "Other"}
    document = {
        "catalog_version": "v3",
        "curricula": [
            {"curriculum": {"subject_id": "D01", "subject_name": "Mathematics", "sections": [capability, other]}}
        ],
    }
    path.write_text(json.dumps(document))
    return path


def proposal_document(capability_hash: str) -> str:
    body = "\n".join(f"## {heading}\n{SECTION}\n" for heading in REQUIRED_HEADINGS)
    return f"""---
id: "{CAPABILITY_ID}/1"
source: {{kind: capability, ref: "{CAPABILITY_ID}", hash: "{capability_hash}"}}
environment: reasoning
verification: code
grounding: unverified
research: []
build:
  - "grader": "the private checker for the total"
resources: ["grader/check.py"]
null_reason: null
---
{body}"""


async def test_a_named_capability_is_proposed_triaged_against_its_catalog_record_and_ends_terminal(tmp_path, fake_glm):
    script = load_script()
    ideas = script.selected_ideas(catalog(tmp_path / "catalog.json"), [CAPABILITY_ID])
    (tmp_path / "glm.txt").write_text("GLM_API_TOKEN=test-token\n")
    example = load_run_config(EXAMPLE)
    config = replace(
        example,
        root=tmp_path / "run",
        host=MachineHost.LAPTOP,
        glm=LaptopGlm(fake_glm.base_url, tmp_path / "glm.txt", Pool.HIGH),
        web=None,
        policy=replace(example.policy, proposals_per_idea=1, max_idea_reproposals=0),
    )
    slot = {
        "slot": 1,
        "title": "stock totals",
        "workflow": "inventory",
        "environment": "reasoning",
        "verification": "code",
        "distinctive_challenge": "mixed units",
        "status": "propose",
        "reason": None,
    }
    plan = {"coverage_rationale": "one slot", "excluded_combinations": [], "slots": [slot], "research_priorities": []}
    fake_glm.stream(tool_calls=(("plan_slots", json.dumps(plan)),), finish="tool_calls")
    fake_glm.stream(content=proposal_document(ideas[CAPABILITY_ID].capability_hash))
    fake_glm.stream(tool_calls=(("record_review", json.dumps(REVIEW)),), finish="tool_calls")

    summary = await run_job(config, partial(script.capability_inputs, ideas, 1), FailedItems.SKIP)

    assert summary.items == {f"{CAPABILITY_ID}--1": Terminal.REJECTED}
    assert summary.failed == {}
    review_request = json.dumps(fake_glm.requests[2]["messages"])
    assert "Linear transformations" in review_request
    assert json.loads((config.root / SUMMARY_FILE).read_text())["items"] == {f"{CAPABILITY_ID}--1": "rejected"}


def test_an_unknown_capability_is_refused_before_the_run_starts(tmp_path):
    script = load_script()

    with pytest.raises(ValueError, match=re.escape("no capabilities ['d09.missing']")):
        script.selected_ideas(catalog(tmp_path / "catalog.json"), [CAPABILITY_ID, "d09.missing"])
