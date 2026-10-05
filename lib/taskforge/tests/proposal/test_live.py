# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of the capability proposal source against GLM-5.3 on the interactive tier.

Runs ``CapabilitySource.propose`` with 10 slots on each of two real catalog capabilities. Evidence
goes to ``lib/taskforge/.evidence/proposal/run-<timestamp>/<capability>/``: the planning request
and completions, and per slot the request messages, the completions, and the rendered ``.md`` (or
the raw replies of a slot that failed). ``summary.json`` has parse and repair counts, pairings,
wall time, and tokens per slot.
"""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from taskforge.llm.client import Completion, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.proposal.model import render
from taskforge.proposal.source import CapabilityIdea, SlotFailure, SlotProposal
from taskforge.proposal.sources.capability import CapabilitySource, load_capability_ideas

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / ".evidence" / "proposal"
CAPABILITY_IDS = ("d01.algebra.linear-transformations", "d27.reporting.close_measurement")
SLOTS = 10


def completion_record(completion: Completion) -> dict[str, object]:
    """The completion without its raw stream events, which are bulky and repeat the content."""
    return {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "tool_calls": [asdict(c) for c in completion.tool_calls],
        "finish_reason": completion.finish_reason,
        "usage": asdict(completion.usage),
        "wall_time": completion.wall_time,
        "ttft": completion.ttft,
        "decode_time": completion.decode_time,
        "continuations": completion.continuations,
        "attempts": [
            {"segment": a.segment, "outcome": a.outcome, "max_tokens": a.max_tokens, "http_status": a.http_status}
            for a in completion.attempts
        ],
    }


def tokens(completions) -> dict[str, int]:
    return {
        "prompt_tokens": sum(c.usage.prompt_tokens for c in completions),
        "completion_tokens": sum(c.usage.completion_tokens for c in completions),
        "reasoning_tokens": sum(c.usage.reasoning_tokens for c in completions),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


async def run_capability(source: CapabilitySource, idea: CapabilityIdea, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    batch = await source.propose(idea, SLOTS)
    total_wall = time.monotonic() - started
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
            "continuations": [c.continuations for c in outcome.completions],
            "decode_tokens_per_second": [round(c.decode_tokens_per_second, 1) for c in outcome.completions],
        }
        slots.append(entry)
    return {
        "capability_id": idea.capability_id,
        "plan_wall_time": sum(c.wall_time for c in batch.planning),
        "plan_calls": len(batch.planning),
        "plan_tokens": tokens(batch.planning),
        "plan_finish_reasons": [c.finish_reason for c in batch.planning],
        "pairings": [e.get("pairing") for e in slots],
        "parsed": sum(bool(e["parsed"]) for e in slots),
        "repaired": sum(e.get("repair_error") is not None for e in slots),
        "planned_null": sum(bool(e.get("planned_null")) for e in slots),
        "failed": len(batch.failures),
        "total_wall_time": total_wall,
        "slots": slots,
    }


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
def test_capability_source_on_two_real_capabilities(glm_settings, capability_catalog):
    ideas = load_capability_ideas(capability_catalog)
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    policy = LLMPolicy()
    out = EVIDENCE_DIR / time.strftime("run-%Y%m%d-%H%M%S")

    async def go() -> list[dict]:
        async with GlmClient(endpoint) as client:
            source = CapabilitySource(client, policy)
            return await asyncio.gather(*(run_capability(source, ideas[cid], out / cid) for cid in CAPABILITY_IDS))

    summaries = asyncio.run(go())
    summary = {"model": glm_settings.model, "policy": asdict(policy), "slots": SLOTS, "capabilities": summaries}
    write_json(out / "summary.json", summary)
    failures = [e["error"] for s in summaries for e in s["slots"] if not e["parsed"]]
    assert not failures, failures
