# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live: one proposal through the whole loop against GLM-5.3 on the interactive (``high``) pool.

The proposal is a stored d43 culinary-scaling proposal that triage accepted in earlier live runs,
handed to ``run_item`` as a supplied proposal. The GLM rubric triages it, GLM authors and builds the
program (with web research through Parallel), the controls replay through the server tokenizer, and
GLM solves the task on ShellSim and attacks its verifier through ``submit`` under the first-run
validation values (k=8, adversary_k=2, 10 verifier submissions, repair threshold 3, band
[0.125, 0.875]) and Taskforge's own band rules (one repair per kind, then reject). Review decides and
the loop follows its decision to a terminal. The item must end ``ACCEPTED`` or ``REJECTED`` with a
``decision.json`` for every ``DECIDED`` event, every adversary attempt file must hold the system turn
the adversary ran under, and a relaunch must return the same terminal without a model call. The run
root goes to ``<evidence_root>/loop/live-test/<utc>/`` with a ``summary.json``; ``evidence_root`` is the
fixture in ``tests/conftest.py``.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx
import pytest

from taskforge.builder.run import item_id_for
from taskforge.builder.sdk import BuildServices
from taskforge.builder.template import standard
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.llm.store import CallStore
from taskforge.llm.web import web_tools
from taskforge.loop.events import EventKind, ProposalOrigin, Terminal, derive_state
from taskforge.loop.policy import POLICY, LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices, run_item
from taskforge.proposal.model import parse
from taskforge.proposal.source import ProposalBatch
from taskforge.review.decision import DECISION_FILE
from taskforge.review.rules import BandChoice, BandRule, BandRules
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.triage.program import GlmRubric
from taskforge.validate.adversary import adversary_brief
from taskforge.validate.attempts import load_adversary_attempt, trial_files
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.controls import ServerTokenizer
from taskforge.validate.outcome import TrialKind
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff

DATA = Path(__file__).resolve().parent / "data"
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
    max_build_retries=2,
    retry_backoff=RetryBackoff(initial=60.0, maximum=900.0, factor=2.0, jitter=0.1),
    output_token_budget=1_000_000,
    band_rules=BandRules(BandRule(1, BandChoice.REJECT), BandRule(1, BandChoice.REJECT)),
    validation=ValidationPolicy(
        k=8,
        adversary_k=2,
        adversary_submissions=10,
        adversary_repair_submissions=3,
        band=CalibrationBand(0.125, 0.875),
        sampling=SAMPLING,
        deadlines=Deadlines(total_turn_timeout=1800.0, attempt_timeout=2400.0),
        max_retries=2,
        token_contract_retries=2,
        retry_backoff=RetryBackoff(initial=1.0, maximum=30.0, factor=2.0, jitter=0.1),
    ),
)


class NoSource:
    """The test supplies its proposal to ``run_item``; proposal generation is layer 07's live test."""

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        raise AssertionError("the live loop test supplies its proposal and proposes nothing")


def event_summary(root: Path, item_id: str) -> list[dict[str, object]]:
    entries = read_entries(root / LEDGER_DIR / f"{item_id}.jsonl")
    return [
        {"step": e.step, "round": e.round, "input_hash": e.input_hash, "attrs": e.attrs}
        for e in entries
        if e.kind is EntryKind.EVENT
    ]


def llm_calls(root: Path, item_id: str) -> int:
    return sum(e.kind is EntryKind.LLM_CALL for e in read_entries(root / LEDGER_DIR / f"{item_id}.jsonl"))


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "loop" / "live-test"


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
async def test_a_proposal_runs_through_the_loop_to_a_terminal(glm_settings, parallel_key, image_cache, evidence_dir):
    proposal = parse((DATA / "d43.culinary.scaling-1.md").read_text())
    record = json.loads((DATA / "d43.culinary.scaling.json").read_text())
    root = evidence_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True)
    policy = POLICY.validate_json(POLICY.dump_json(POLICY_VALUES))
    (root / "policy.json").write_bytes(POLICY.dump_json(policy, indent=2))
    ledger = JsonlLedger(root / LEDGER_DIR)
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=POOL)
    started = time.monotonic()

    async with GlmClient(endpoint) as client, httpx.AsyncClient() as http:
        factories = machine_factories(MachineHost.LAPTOP, controller_url=None, image_cache=image_cache)

        def services() -> LoopServices:
            return LoopServices(
                client=client,
                source=NoSource(),
                describe_idea=lambda idea: {"idea": idea},
                checks=CHECKS,
                rubric=GlmRubric(
                    CallStore(root / "calls", client), LLMPolicy(), RUBRIC_SAMPLES, {proposal.header.source: record}
                ),
                check_context=CheckContext(allowed_combinations=ALL_COMBINATIONS),
                template=standard,
                build=BuildServices(
                    client=client,
                    policy=LLMPolicy(),
                    host=MachineHost.LAPTOP,
                    factories=factories,
                    images=None,
                    ledger=ledger,
                    web_tools=web_tools(http, parallel_key.value),
                ),
                engine=EngineSettings(
                    factories=factories,
                    capabilities=factory_capabilities(MachineHost.LAPTOP),
                    max_turns=40,
                    command_timeout=120.0,
                    tool_turn_timeout=240.0,
                    model_turn_timeout=600.0,
                    cleanup_timeout=120.0,
                ),
                rollout_models=partial(GlmRolloutModel, client, SAMPLING),
                adversary_context=lambda proposal: "",
                tokenize=ServerTokenizer(client, SAMPLING),
                ledger=ledger,
                root=root,
                slots=asyncio.Semaphore(256),
            )

        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services())
        wall_time = time.monotonic() - started
        item_id = item_id_for(proposal)
        calls = llm_calls(root, item_id)
        relaunched = await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services())

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
        for files in trial_files(evidence, TrialKind.ADVERSARY).values():
            assert files.last_path is not None
            trial = load_adversary_attempt(files.last_path)
            assert trial.system.startswith(adversary_brief(policy.validation.adversary_submissions, "")), files.last_path
            assert len(trial.submissions) <= policy.validation.adversary_submissions
