# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live: a review ``Repair`` drives a real revision through the author and the build.

GLM-5.3 authors and builds a program, review turns a shortcut finding into a ``Repair``, the brief goes
to ``build.author.author`` as ``Revision.failure``, and the revised program is rebuilt with the repair's
``invalidate``. The revised draft must ship the shortcut control verbatim, and its grader must reject
that control when RolloutEngine replays it. Takes ``parallel_key`` for the template's research step.
Artifacts go to ``.evidence/review/live-test/<timestamp>/``.
"""

import time
from pathlib import Path

import httpx
import pytest
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.grading_result import Outcome

from taskforge.build.author import BuildProgram, Revision, author
from taskforge.build.run import TaskDraft, item_id_for, run_build
from taskforge.build.sdk import BuildFailure, BuildServices
from taskforge.build.step import StepRole
from taskforge.build.template import standard
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.web import web_tools
from taskforge.proposal.model import TaskProposal, parse
from taskforge.review.decision import Repair
from taskforge.review.rules import ItemHistory, decide
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    reply,
)
from taskforge.validate.calibration import FINDING_ROLES, Finding, FindingKind
from taskforge.validate.controls import control_turns

EVIDENCE = Path(__file__).resolve().parents[2] / ".evidence" / "review" / "live-test"
BUILD_REVISIONS = 3

PROPOSAL = """---
id: "d00.arithmetic.products/1"
source: {kind: capability, ref: "d00.arithmetic.products", hash: "abc123"}
environment: reasoning
verification: simple
grounding: unverified
research:
  - {kind: web, purpose: "confirm the multiplication table"}
build:
  - "grader": "checks the final ANSWER line"
resources: ["grader/grade.py"]
null_reason: null
---
## Task
Multiply 6 by 7 and end the reply with `ANSWER = <n>`.

## Realism and workflow
A clerk checks a product.

## Research plan
Confirm the table.

## Build plan
Write the question and the grader.

## Grader design and controls
Full credit only for ANSWER = 42.

## Risks and null conditions
None.
"""

SHORTCUT = Control(
    id="adv-shortcut-0",
    kind=ControlKind.NEGATIVE,
    category=ControlCategory.REWARD_HACK,
    concern=ControlConcern.SHORTCUT,
    author="adversary/shortcut/0",
    payload=Transcript((reply("ANSWER = 41\nANSWER = 43\nANSWER = 42"),)),
    expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
)
SHORTCUT_DETAIL = (
    "Shortcut adversary 0 passed without computing anything: it listed three candidate answers, one ANSWER "
    "line each, ending with the right one, and the grader gave full reward. A reply must commit to exactly "
    "one answer; a reply with several ANSWER lines must score zero."
)


async def build_with_revisions(
    proposal: TaskProposal,
    revision: Revision | None,
    item_dir: Path,
    cache: Path,
    services: BuildServices,
    invalidate: tuple[str, ...],
    round: int,  # noqa: A002 - matches LedgerEntry.round
) -> tuple[BuildProgram, TaskDraft]:
    """Author and build, re-authoring on a build failure as the loop does, up to ``BUILD_REVISIONS``."""
    item_id = item_id_for(proposal)
    for _ in range(BUILD_REVISIONS):
        program = await author(proposal, standard, item_dir, services, item_id, revision, round=round)
        try:
            return program, await run_build(program, proposal, item_dir, cache, services, invalidate, round=round)
        except (BuildFailure, ValueError) as failure:
            revision = Revision(source=program.source, failure=str(failure))
    raise AssertionError(f"no buildable program in {BUILD_REVISIONS} authorings: {revision}")


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
async def test_repair_brief_revises_the_program_so_its_grader_rejects_the_shortcut(
    glm_settings, parallel_key, image_cache, summary, rules, scripted_model
):
    proposal = parse(PROPOSAL)
    run_dir = EVIDENCE / time.strftime("%Y%m%d-%H%M%S")
    factories = machine_factories(MachineHost.LAPTOP, controller_url=None, image_cache=image_cache)
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    async with GlmClient(endpoint) as client, httpx.AsyncClient() as http:
        services = BuildServices(
            client=client,
            policy=LLMPolicy(),
            factories=factories,
            ledger=JsonlLedger(run_dir / "ledger"),
            web_tools=web_tools(http, parallel_key.value),
        )
        first_dir, cache = run_dir / "round-0", run_dir / "cache"
        program, draft = await build_with_revisions(proposal, None, first_dir, cache, services, (), round=0)

        finding = Finding(
            kind=FindingKind.SHORTCUT_PASSED,
            detail=SHORTCUT_DETAIL,
            roles=FINDING_ROLES[FindingKind.SHORTCUT_PASSED],
            new_controls=(SHORTCUT,),
        )
        decision = decide(draft, summary((finding,)), ItemHistory(0, 2, {}), rules)
        assert isinstance(decision, Repair)
        condemned = {r.name for r in draft.provenance.steps if r.role in {StepRole.GRADER, StepRole.CONTROLS}}
        assert decision.program_digest == program.digest
        assert set(decision.invalidate) == condemned
        (run_dir / "brief.md").write_text(decision.brief.failure)

        revision = Revision(source=program.source, failure=decision.brief.failure)
        revised, repaired = await build_with_revisions(
            proposal, revision, run_dir / "round-1", cache, services, decision.invalidate, round=1
        )

    assert revised.digest != program.digest
    assert SHORTCUT in repaired.controls

    turns = control_turns(SHORTCUT)
    engine = ShellboxRolloutEngine(
        scripted_model(turns),
        factories,
        max_turns=len(turns),
        command_timeout=60,
        cleanup_timeout=60,
        convention=repaired.convention,
    )
    rollout = await engine.run(repaired.task, execution=repaired.execution)
    assert SHORTCUT.expect.met_by(rollout.grade), rollout.grade
