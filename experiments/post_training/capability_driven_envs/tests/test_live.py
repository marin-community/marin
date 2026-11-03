# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live proposal runs against GLM-5.3's high pool on capabilities of the full catalog.

``test_every_proposed_slot_parses_on_its_first_request`` runs ``propose_curriculum.run_shard`` with ``SLOTS``
slots on a host that runs every environment, once per capability in ``LIVE_CAPABILITIES``, writing the driver's
files to ``<evidence_root>/proposal/<run name>/``, and adds ``live_summary.json`` there with each slot's request
count, pairing and decision.
"""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskforge.llm.glm_client import FixedEndpoint, GlmClient
from taskforge.stage10_proposal.model import Environment

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER
from experiments.post_training.capability_driven_envs.propose_curriculum import (
    RESULT_FILE,
    OutputRoot,
    ShardSpec,
    run_shard,
)

SLOTS = 4
LIVE_CAPABILITIES = ("d01.algebra.linear-transformations", "d27.reporting.close_measurement")
LIVE_SUMMARY_FILE = "live_summary.json"


def unique_run_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-{uuid.uuid4().hex[:8]}"


@pytest.mark.live_glm
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("capability_id", LIVE_CAPABILITIES)
def test_every_proposed_slot_parses_on_its_first_request(
    capability_id: str, live_endpoint: FixedEndpoint, capability_catalog: Path, evidence_root: Path
):
    idea = load_capability_ideas(capability_catalog)[capability_id]
    out = evidence_root / "proposal" / unique_run_name(capability_id)
    spec = ShardSpec(SLOTS, SLOTS, frozenset(Environment), 131072, str(capability_catalog), "live")

    async def go() -> list[str]:
        async with GlmClient(live_endpoint, GLM_PROVIDER) as client:
            return await run_shard(client, {capability_id: idea}, OutputRoot(str(out)), spec)

    assert asyncio.run(go()) == []
    directory = out / capability_id
    result = json.loads((directory / RESULT_FILE).read_text())
    slots: dict[int, dict[str, Any]] = {}
    for n in range(1, SLOTS + 1):
        completions = json.loads((directory / f"slot-{n:02d}.completions.json").read_text())
        proposal = directory / f"slot-{n:02d}.proposal.json"
        header = json.loads(proposal.read_text())["header"] if proposal.exists() else None
        slots[n] = {
            "calls": len(completions),
            "pairing": header and f"{header['environment']}x{header['verification']}",
            "null_reason": header and header["null_reason"],
            "decision": result["slots"][str(n)],
        }
    (out / LIVE_SUMMARY_FILE).write_text(json.dumps({"result": result, "slots": slots}, indent=2) + "\n")

    proposed = [s for s in slots.values() if s["calls"] > 0]
    assert [s for s in proposed if s["pairing"] is None or s["calls"] != 1] == []
