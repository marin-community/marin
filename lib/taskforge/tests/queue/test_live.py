# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One unattended run end to end on GLM-5.3's interactive pool: ``queue.job.run_job`` from a laptop
config through triage (the GLM rubric), authoring, a ShellSim build, controls, solver and adversary
trials and review, then a relaunch on the same root that reaches the same terminal without a model call.

The run root is kept under ``<evidence_root>/queue/<timestamp>/``, where ``evidence_root`` is the fixture in
``tests/conftest.py``.
"""

import json
import os
import time
from pathlib import Path

import pytest
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import ledger_files, read_entries
from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.llm.client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.store import CallStore
from taskforge.loop.events import EventKind, Terminal
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import LEDGER_DIR
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.queue.config import EngineConfig, LaptopGlm, RunConfig
from taskforge.queue.job import SUMMARY_FILE, RunInputs, run_job
from taskforge.queue.run import FailedItems
from taskforge.review.decision import BandOutcome
from taskforge.review.rules import BandChoice, BandRule, BandRules
from taskforge.sandbox.factories import MachineHost
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.triage.program import GlmRubric
from taskforge.validate.adversary import AdversaryRole, adversary_brief
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, RetryBackoff

TOKEN_FILE_ENV = "TASKFORGE_GLM_TOKEN_FILE"
SUBMISSIONS = 10

PROPOSAL = """---
id: "live.queue.units/0"
source: {kind: capability, ref: "live.queue.units", hash: "live"}
environment: reasoning
verification: simple
grounding: unverified
research: []
build:
  - "grader": "checks the converted total on the final ANSWER line with an exact numeric match"
resources: ["grader/grade.py"]
null_reason: null
---
## Task
A hospital pharmacy technician receives three reagent bottles labelled 2.5 L, 750 mL and 0.04 L and
must record their combined volume on an inventory sheet that only accepts whole millilitres. The agent
converts each label to millilitres, adds the three volumes, explains the conversion briefly, and ends
its reply with a final line of the form `ANSWER = <n>` where n is the total in millilitres.

## Realism and workflow
Pharmacy and laboratory inventory systems record liquid stock in millilitres, while suppliers print
labels in a mix of litres and millilitres. Technicians convert and total mixed labels every week when
they reconcile stock, and a factor-of-ten slip in one bottle silently corrupts the reorder quantities.

## Research plan
No research is needed: the conversion factor between litres and millilitres is fixed at one thousand,
and the three label values are given in the instruction itself. The builder should not search the web;
it writes the instruction, the grader and the controls directly from this proposal and its numbers.

## Build plan
The builder writes the instruction above, then the grader script grader/grade.py that reads the final
assistant reply, finds the last line starting with ANSWER =, parses the integer after it and compares it
with the expected total of 3290 millilitres. It then writes the controls listed below and checks them.

## Grader design and controls
The grader is an exact numeric match: full credit (reward 1.0) only when the final ANSWER line holds
exactly 3290, and zero otherwise, including a missing line. Controls: the correct reply ending ANSWER =
3290; 3290.5 and 329 and 32900 as factor or rounding errors; an empty reply; a reply listing two totals.

## Risks and null conditions
The arithmetic is simple, so a strong solver may solve it every time and review may find the task too
easy for the calibration band. That outcome is acceptable for this proposal, which exists to exercise the
whole pipeline end to end; there is no null condition because the inputs are fully specified above.
"""


class OneProposal:
    """The run's source: the fixed proposal above for its one idea."""

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        assert n == 1
        return ProposalBatch((), (), (SlotProposal(0, parse(PROPOSAL), (), (), None),))


def describe_unit(idea: str) -> dict[str, object]:
    return {"idea": idea}


def no_context(proposal: TaskProposal) -> str:
    return ""


def inputs(client: GlmClient, root: Path) -> RunInputs[str]:
    proposal: TaskProposal = parse(PROPOSAL)
    record = {"capability": "unit conversion in lab inventory", "difficulty": "easy"}
    rubric = GlmRubric(CallStore(root / "calls", client), LLMPolicy(), 1, {proposal.header.source: record})
    return RunInputs(
        ideas={"live.queue.units": "units"},
        source=OneProposal(),
        describe_idea=describe_unit,
        adversary_context=no_context,
        checks=CHECKS,
        rubric=rubric,
        check_context=CheckContext(ALL_COMBINATIONS),
    )


def policy() -> LoopPolicy:
    validation = ValidationPolicy(
        k=4,
        adversary_k=1,
        adversary_submissions=SUBMISSIONS,
        adversary_repair_submissions=3,
        band=CalibrationBand(0.125, 0.875),
        sampling=LLMPolicy(max_continuations=0),
        deadlines=Deadlines(agent_timeout=900, attempt_timeout=1200),
        max_retries=2,
        token_contract_retries=2,
        retry_backoff=RetryBackoff(initial=5, maximum=60, factor=2, jitter=0.1),
    )
    return LoopPolicy(
        proposals_per_idea=1,
        max_idea_reproposals=0,
        max_triage_repairs=1,
        max_build_revisions=2,
        max_repairs=0,
        max_validation_retries=1,
        max_build_retries=1,
        retry_backoff=RetryBackoff(initial=10, maximum=60, factor=2, jitter=0.1),
        output_token_budget=1_000_000,
        # Taskforge's own band policy: a task outside the band is rejected.
        band_rules=BandRules(BandRule(1, BandChoice.REJECT), BandRule(1, BandChoice.REJECT)),
        validation=validation,
    )


def entries(root: Path) -> list[LedgerEntry]:
    return [entry for path in ledger_files(root / LEDGER_DIR) for entry in read_entries(path)]


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "queue"


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
async def test_a_laptop_run_reaches_a_terminal_and_a_relaunch_repeats_no_model_call(
    glm_settings, image_cache, evidence_dir
):
    root = evidence_dir / time.strftime("%Y%m%d-%H%M%S")
    config = RunConfig(
        run_id=f"queue-live-{root.name}",
        root=root,
        host=MachineHost.LAPTOP,
        image_cache=image_cache,
        glm=LaptopGlm(glm_settings.base_url, Path(os.environ[TOKEN_FILE_ENV]).expanduser(), Pool.HIGH),
        web=None,
        policy=policy(),
        engine=EngineConfig(
            max_turns=20, command_timeout=120, cleanup_timeout=120, conventions=(PlainText(id="plain_text"),)
        ),
        width=64,
        restore_from=None,
    )

    first = await run_job(config, inputs, FailedItems.SKIP)

    (item,) = first.items
    assert first.items[item] in {Terminal.ACCEPTED, Terminal.REJECTED}, json.dumps(first.summary_json())
    assert first.failed == {}
    recorded = entries(root)
    steps = {entry.step for entry in recorded}
    assert {"author", EventKind.BUILT, EventKind.CONTROLS_REPLAYED, EventKind.DECIDED} <= steps, sorted(steps)
    assert sum(entry.tokens_out or 0 for entry in recorded if entry.kind is EntryKind.LLM_CALL) > 0
    assert any(entry.kind is EntryKind.TRIAL for entry in recorded)
    written = json.loads((root / SUMMARY_FILE).read_text())
    assert written["items"] == {item: str(first.items[item])}
    if first.items[item] is Terminal.ACCEPTED:
        accepted = first.accepted[item]
        assert accepted.k == config.policy.validation.k
        assert accepted.solve_rate == accepted.solved / accepted.k
        assert accepted.band is BandOutcome.IN_BAND  # the policy rejects outside the band
        assert written["accepted"][item]["solved"] == accepted.solved
    else:
        assert first.accepted == {}
    attempts = sorted(root.rglob(f"adversary/{AdversaryRole.SHORTCUT}/*/attempt-*.json"))
    for path in attempts:
        adversary = json.loads(path.read_text())["adversary"]
        assert adversary["system"].startswith(adversary_brief(SUBMISSIONS, "")), path
        assert len(adversary["submissions"]) <= SUBMISSIONS, path

    second = await run_job(config, inputs, FailedItems.SKIP)

    assert second.items == first.items
    assert (second.accepted, second.noted) == (first.accepted, first.noted)
    assert len(entries(root)) == len(recorded)
