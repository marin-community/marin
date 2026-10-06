# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live: one proposal through the whole loop against GLM-5.3 on the interactive (``high``) pool.

The proposal is a stored d43 culinary-scaling proposal that triage accepted in earlier live runs. The
GLM rubric triages it, GLM authors and builds the program (with web research through Parallel), the
controls replay through the server tokenizer, and GLM solves and attacks the task on ShellSim under
the first-run validation values (k=8, adversary_k=2, every role, band [0.125, 0.875]). Review decides
and the loop follows its decision to a terminal. The item must end ``ACCEPTED`` or ``REJECTED`` with a
``decision.json`` for every ``DECIDED`` event, and a relaunch must return the same terminal without a
model call. The run root goes to ``.evidence/loop/live-test/<utc>/`` with a ``summary.json``.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from rigging.timing import ExponentialBackoff
from taskcompendium.submission import JsonAnswer, JsonValueAnswer, PlainText

from taskforge.build.run import item_id_for
from taskforge.build.sdk import BuildServices
from taskforge.build.template import standard
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.llm.store import CallStore
from taskforge.llm.web import web_tools
from taskforge.loop.events import EventKind, Terminal, derive_state
from taskforge.loop.policy import POLICY, LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices, run_idea, run_item
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.review.decision import DECISION_FILE
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.triage.program import GlmRubric
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.controls import ServerTokenizer
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, EngineSettings

DATA = Path(__file__).resolve().parent / "data"
EVIDENCE = Path(__file__).resolve().parents[2] / ".evidence" / "loop" / "live-test"
IDEA = "d43.culinary.scaling"
RUBRIC_SAMPLES = 3
POOL = Pool.HIGH
"""Every validation run driven from a test uses the interactive pool; unattended runs name BULK in their config."""

SAMPLING = LLMPolicy(temperature=0.7, max_continuations=0)
POLICY_VALUES = LoopPolicy(
    proposals_per_idea=1,
    max_idea_reproposals=0,
    max_triage_repairs=1,
    max_build_revisions=3,
    max_repairs=1,
    max_validation_retries=2,
    retry_backoff=ExponentialBackoff(initial=60.0, maximum=900.0, factor=2.0, jitter=0.1),
    output_token_budget=1_000_000,
    validation=ValidationPolicy(
        k=8,
        adversary_k=2,
        roles=tuple(AdversaryRole),
        band=CalibrationBand(0.125, 0.875),
        sampling=SAMPLING,
        deadlines=Deadlines(agent_timeout=1800.0, attempt_timeout=2400.0),
        max_retries=2,
        token_contract_retries=2,
        retry_backoff=ExponentialBackoff(initial=1.0, maximum=30.0, factor=2.0, jitter=0.1),
    ),
)


class StoredProposal:
    """A proposal source that serves the stored proposal: proposal generation is layer 07's live test."""

    def __init__(self, proposal: TaskProposal):
        self.proposal = proposal

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        return ProposalBatch((), (), (SlotProposal(0, self.proposal, (), (), None),))


def event_summary(root: Path, item_id: str) -> list[dict[str, object]]:
    entries = read_entries(root / LEDGER_DIR / f"{item_id}.jsonl")
    return [
        {"step": e.step, "round": e.round, "input_hash": e.input_hash, "attrs": e.attrs}
        for e in entries
        if e.kind is EntryKind.EVENT
    ]


def llm_calls(root: Path, item_id: str) -> int:
    return sum(e.kind is EntryKind.LLM_CALL for e in read_entries(root / LEDGER_DIR / f"{item_id}.jsonl"))


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
async def test_a_proposal_runs_through_the_loop_to_a_terminal(glm_settings, parallel_key):
    proposal = parse((DATA / "d43.culinary.scaling-1.md").read_text())
    record = json.loads((DATA / "d43.culinary.scaling.json").read_text())
    root = EVIDENCE / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True)
    policy = POLICY.validate_json(POLICY.dump_json(POLICY_VALUES))
    (root / "policy.json").write_bytes(POLICY.dump_json(policy, indent=2))
    ledger = JsonlLedger(root / LEDGER_DIR)
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=POOL)
    started = time.monotonic()

    async with GlmClient(endpoint) as client, httpx.AsyncClient() as http:
        factories = machine_factories(MachineHost.LAPTOP, controller_url=None)

        def services() -> LoopServices:
            return LoopServices(
                client=client,
                source=StoredProposal(proposal),
                checks=CHECKS,
                rubric=GlmRubric(
                    CallStore(root / "calls", client), LLMPolicy(), RUBRIC_SAMPLES, {proposal.header.source: record}
                ),
                check_context=CheckContext(allowed_combinations=ALL_COMBINATIONS),
                template=standard,
                build=BuildServices(
                    client=client,
                    policy=LLMPolicy(),
                    factories=factories,
                    ledger=ledger,
                    web_tools=web_tools(http, parallel_key.value),
                ),
                engine=EngineSettings(
                    factories=factories,
                    capabilities=factory_capabilities(MachineHost.LAPTOP),
                    max_turns=40,
                    command_timeout=120.0,
                    cleanup_timeout=120.0,
                    conventions=(
                        PlainText(id="plain_text"),
                        JsonAnswer(id="json_answer"),
                        JsonValueAnswer(id="json_value"),
                    ),
                ),
                rollout_model=GlmRolloutModel(client, SAMPLING),
                tokenize=ServerTokenizer(client, SAMPLING),
                ledger=ledger,
                root=root,
                slots=asyncio.Semaphore(256),
            )

        (item,) = await run_idea(IDEA, IDEA, policy, services())
        terminal = await run_item(item, policy, services())
        wall_time = time.monotonic() - started
        item_id = item_id_for(item)
        calls = llm_calls(root, item_id)
        relaunched = await run_item(item, policy, services())

    events = event_summary(root, item_id)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "purpose": "one proposal through triage, build, validation and review to a terminal",
                "pool": POOL,
                "policy_digest": policy.digest,
                "terminal": terminal,
                "wall_time": wall_time,
                "llm_calls": calls,
                "events": events,
            },
            indent=1,
            default=str,
        )
    )

    assert terminal in {Terminal.ACCEPTED, Terminal.REJECTED}, events
    assert relaunched is terminal and llm_calls(root, item_id) == calls
    state = derive_state(read_entries(root / LEDGER_DIR / f"{item_id}.jsonl"))
    assert state.terminal is terminal
    for decided in (e for e in events if e["step"] == EventKind.DECIDED):
        built = [e for e in events if e["step"] == EventKind.BUILT and e["round"] == decided["round"]][-1]
        digest = str(built["input_hash"])
        evidence = root / "items" / item_id / "rounds" / str(decided["round"]) / f"evidence-{digest[:12]}"
        assert (evidence / DECISION_FILE).exists(), evidence
