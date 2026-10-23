# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live proposal runs against GLM-5.3's high pool on capabilities of the full catalog.

``test_every_proposed_slot_parses_on_its_first_request`` runs ``propose_tasks.propose_tasks`` with
``SLOTS`` slots on a host that runs every environment, once per capability in ``LIVE_CAPABILITIES``.
``propose_tasks.write_outputs`` writes its files to ``<evidence_root>/proposal/<run name>/``, and
the test adds ``live_summary.json`` there, with parse and repair counts, pairings, each slot's
planned and written research decision, wall time, and tokens per slot.
"""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from taskforge.llm.client import Completion
from taskforge.llm.glm_client import FixedEndpoint, GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment
from taskforge.stage10_proposal.source import SlotFailure, SlotProposal

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER
from experiments.post_training.capability_driven_envs.propose_tasks import propose_tasks, write_outputs

SLOTS = 4
LIVE_CAPABILITIES = ("d01.algebra.linear-transformations", "d27.reporting.close_measurement")
LIVE_SUMMARY_FILE = "live_summary.json"


def unique_run_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-{uuid.uuid4().hex[:8]}"


def tokens(completions: Sequence[Completion]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(c.usage.prompt_tokens for c in completions),
        "completion_tokens": sum(c.usage.completion_tokens for c in completions),
        "reasoning_tokens": sum(c.usage.reasoning_tokens for c in completions),
    }


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
            return await propose_tasks(client, {capability_id: idea}, SLOTS, frozenset(Environment), policy)

    started = time.monotonic()
    proposed = asyncio.run(go())
    total_wall = time.monotonic() - started
    write_outputs(out, proposed)
    (capability,) = proposed
    assert capability.failure is None, capability.failure
    batch = capability.batch
    planned = {s["slot"]: s for s in json.loads(batch.planning[-1].content)["slots"]}
    slots = []
    for outcome in batch.slots:
        entry: dict[str, object] = {"slot": outcome.slot, "calls": len(outcome.completions)}
        if isinstance(outcome, SlotProposal):
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
    (out / LIVE_SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")

    proposed = [s for s in slots if s.get("planned_null") is not True]
    assert [s for s in proposed if not s["parsed"] or s["repair_error"] is not None] == []
