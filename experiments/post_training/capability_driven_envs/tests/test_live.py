# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live proposal and review runs against GLM-5.3's high pool on capabilities of the full catalog.

``test_every_proposed_slot_parses_on_its_first_request`` runs ``CapabilitySource.propose`` with
``SLOTS`` slots on a host that runs every environment, once per capability in ``LIVE_CAPABILITIES``.
It writes to ``<evidence_root>/proposal/<run name>/``:

* the planning request and completions;
* per slot, the request, the completions and the rendered ``.md``, or the raw replies of a failed slot;
* ``summary.json``, with parse and repair counts, pairings, each slot's planned and written research
  decision, wall time, and tokens per slot.

``test_the_rubric_flags_a_prose_lookup_and_repairs_a_proposal_under_the_regex`` reviews the three
GLM-written proposals under ``data/``:

* ``needed.md`` cites six research items;
* ``not_needed.md`` describes no lookup;
* ``prose_lookup.md`` is not_needed, and its Risks section describes a web contamination search with
  no research entry.

Each gets ``RUBRIC_SAMPLES`` rubric samples, and the first proposal with a REPAIR verdict gets a
repair under the document regex. The run writes ``calls/`` (the ``CallStore``), a verdict per
proposal under ``verdicts/``, ``repair.md`` and ``summary.json`` with decisions, per-sample scores,
wall times and tokens under ``<evidence_root>/triage/<run name>/``.
"""

import asyncio
import json
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from taskforge.llm.client import Completion
from taskforge.llm.glm_client import FixedEndpoint, GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, TaskProposal, parse, render
from taskforge.stage10_proposal.source import SlotFailure, SlotProposal
from taskforge.stage20_triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.stage20_triage.program import GlmRubric, evaluate, sample_decision
from taskforge.stage20_triage.verdict import RubricAxis, TriageDecision, Verdict

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER
from experiments.post_training.capability_driven_envs.proposal_source import CapabilitySource
from experiments.post_training.capability_driven_envs.propose_tasks import capability_rubric

SLOTS = 4
LIVE_CAPABILITIES = ("d01.algebra.linear-transformations", "d27.reporting.close_measurement")
DATA = Path(__file__).parent / "data"
FIXTURES = ("needed", "not_needed", "prose_lookup")
RUBRIC_SAMPLES = 3
PROSE_LOOKUP_CEILING = 3
"""A ``source_honesty`` score at or below this marks a sample that caught the prose lookup."""


def unique_run_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-{uuid.uuid4().hex[:8]}"


def completion_record(completion: Completion) -> dict[str, object]:
    """Record ``completion`` without its attempts' raw stream events, which are bulky and repeat the content."""
    return {
        **{key: value for key, value in asdict(completion).items() if key != "attempts"},
        "attempts": [
            {"segment": a.segment, "outcome": a.outcome, "max_tokens": a.max_tokens, "http_status": a.http_status}
            for a in completion.attempts
        ],
    }


def tokens(completions: Sequence[Completion]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(c.usage.prompt_tokens for c in completions),
        "completion_tokens": sum(c.usage.completion_tokens for c in completions),
        "reasoning_tokens": sum(c.usage.reasoning_tokens for c in completions),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


@pytest.mark.live_glm
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("capability_id", LIVE_CAPABILITIES)
def test_every_proposed_slot_parses_on_its_first_request(
    capability_id: str, live_endpoint: FixedEndpoint, capability_catalog: Path, evidence_root: Path
):
    idea = load_capability_ideas(capability_catalog)[capability_id]
    policy = LLMPolicy()
    out = evidence_root / "proposal" / unique_run_name(capability_id)
    out.mkdir(parents=True)

    async def go():
        async with GlmClient(live_endpoint, GLM_PROVIDER) as client:
            return await CapabilitySource(client, policy, frozenset(Environment)).propose(idea, SLOTS)

    started = time.monotonic()
    batch = asyncio.run(go())
    total_wall = time.monotonic() - started
    planned = {s["slot"]: s for s in json.loads(batch.planning[-1].content)["slots"]}
    write_json(out / "plan.request.json", list(batch.planning_request))
    write_json(out / "plan.completions.json", [completion_record(c) for c in batch.planning])
    slots = []
    for outcome in batch.slots:
        name = f"slot-{outcome.slot:02d}"
        if outcome.request:
            write_json(out / f"{name}.request.json", list(outcome.request))
        write_json(out / f"{name}.completions.json", [completion_record(c) for c in outcome.completions])
        entry: dict[str, object] = {"slot": outcome.slot, "calls": len(outcome.completions)}
        if isinstance(outcome, SlotProposal):
            (out / f"{name}.md").write_text(render(outcome.proposal))
            header = outcome.proposal.header
            entry |= {
                "parsed": True,
                "planned_null": not outcome.completions,
                "repair_error": outcome.repair_error,
                "null_reason": header.null_reason,
                "pairing": f"{header.environment}x{header.verification}",
                "planned_research_need": planned[outcome.slot]["research_need"],
                "research_need": header.research_need,
                "research_items": len(header.research),
                "digest": outcome.proposal.digest,
            }
        else:
            assert isinstance(outcome, SlotFailure)
            for index, completion in enumerate(outcome.completions):
                (out / f"{name}.failed-{index}.txt").write_text(completion.content)
            entry |= {"parsed": False, "error": outcome.error}
        entry |= {
            "wall_time": sum(c.wall_time for c in outcome.completions),
            "tokens": tokens(outcome.completions),
            "finish_reasons": [c.finish_reason for c in outcome.completions],
        }
        slots.append(entry)
    summary = {
        "model": GLM_PROVIDER.model,
        "policy": asdict(policy),
        "slots_requested": SLOTS,
        "capability_id": idea.capability_id,
        "plan_wall_time": sum(c.wall_time for c in batch.planning),
        "plan_calls": len(batch.planning),
        "plan_tokens": tokens(batch.planning),
        "pairings": [e.get("pairing") for e in slots],
        "parsed": sum(bool(e["parsed"]) for e in slots),
        "repaired": sum(e.get("repair_error") is not None for e in slots),
        "failed": len(batch.failures),
        "total_wall_time": total_wall,
        "slots": slots,
    }
    write_json(out / "summary.json", summary)

    proposed = [s for s in slots if s.get("planned_null") is not True]
    assert [s for s in proposed if not s["parsed"] or s["repair_error"] is not None] == []


def evaluation_record(verdict: Verdict, wall_time: float) -> dict[str, object]:
    return {
        "decision": verdict.decision,
        "reasons": list(verdict.reasons),
        "sample_decisions": [sample_decision(r)[0] for r in verdict.rubric],
        "sample_scores": [{str(a): s for a, s in r.scores} for r in verdict.rubric],
        "issues": [list(r.issues) for r in verdict.rubric],
        "wall_time": wall_time,
        "calls": len(verdict.calls),
        "prompt_tokens": sum(c.usage.prompt_tokens for c in verdict.calls),
        "completion_tokens": sum(c.usage.completion_tokens for c in verdict.calls),
    }


async def timed_evaluate(p: TaskProposal, rubric: GlmRubric, ctx: CheckContext) -> tuple[Verdict, float]:
    started = time.monotonic()
    verdict = await evaluate(p, CHECKS, rubric, ctx)
    return verdict, time.monotonic() - started


@pytest.mark.live_glm
@pytest.mark.timeout(600)
def test_the_rubric_flags_a_prose_lookup_and_repairs_a_proposal_under_the_regex(
    live_endpoint: FixedEndpoint, capability_catalog: Path, evidence_root: Path
):
    ideas = load_capability_ideas(capability_catalog)
    proposals = {name: parse((DATA / f"{name}.md").read_text()) for name in FIXTURES}
    out = evidence_root / "triage" / unique_run_name("triage")
    (out / "verdicts").mkdir(parents=True)
    ctx = CheckContext(allowed_combinations=ALL_COMBINATIONS)

    async def run():
        async with GlmClient(live_endpoint, GLM_PROVIDER) as client:
            rubric = capability_rubric(client, out / "calls", LLMPolicy(), RUBRIC_SAMPLES, ideas)
            outcomes = await asyncio.gather(*(timed_evaluate(p, rubric, ctx) for p in proposals.values()))
            verdicts = dict(zip(FIXTURES, outcomes, strict=True))
            repairable = [name for name in FIXTURES if verdicts[name][0].decision is TriageDecision.REPAIR]
            if not repairable:
                return verdicts, None, None
            name = repairable[0]
            started = time.monotonic()
            repair = await rubric.repair(proposals[name], verdicts[name][0])
            return verdicts, (name, repair), time.monotonic() - started

    verdicts, repaired, repair_wall = asyncio.run(run())
    for name, (verdict, _) in verdicts.items():
        (out / "verdicts" / f"{name}.json").write_text(verdict.to_json())
    summary: dict[str, object] = {
        "model": GLM_PROVIDER.model,
        "rubric_samples": RUBRIC_SAMPLES,
        "evaluations": {name: evaluation_record(v, w) for name, (v, w) in verdicts.items()},
    }
    if repaired is not None:
        name, repair = repaired
        (out / "repair.md").write_text(render(repair.proposal))
        summary["repair"] = {"proposal": name, "wall_time": repair_wall, "calls": len(repair.calls)}
    write_json(out / "summary.json", summary)

    prose = verdicts["prose_lookup"][0]
    caught = [s.score(RubricAxis.SOURCE_HONESTY) <= PROSE_LOOKUP_CEILING for s in prose.rubric]
    assert sum(caught) * 2 > len(caught), summary["evaluations"]
    assert repaired is not None, "no fixture got a REPAIR verdict"
    name, repair = repaired
    assert repair.proposal.header.id == proposals[name].header.id
    assert parse(render(repair.proposal)) == repair.proposal
