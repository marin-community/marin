# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live proposal runs against GLM-5.3's high pool on capabilities of the full catalog.

``test_every_proposed_slot_parses_on_its_first_request`` runs ``propose_tasks.propose_tasks`` with ``SLOTS``
slots on a host that runs every environment, once per capability in ``LIVE_CAPABILITIES``, and writes the
driver's files, completions with their usage included, to ``<evidence_root>/proposal/<run name>/``.

Invariants: none (pins that the live model's documents parse without a repair).
"""

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge_adapters")

import asyncio
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from taskforge.defaults import DEFAULT_POLICY
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment
from taskforge.stage10_proposal.source import SlotProposal
from taskforge_adapters.glm_client import FixedEndpoint, GlmClient

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER
from experiments.post_training.capability_driven_envs.propose_tasks import propose_tasks, write_outputs

SLOTS = 4
LIVE_CAPABILITIES = ("d01.algebra.linear-transformations", "d27.reporting.close_measurement")


@pytest.mark.live_glm
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("capability_id", LIVE_CAPABILITIES)
def test_every_proposed_slot_parses_on_its_first_request(
    capability_id: str, live_endpoint: FixedEndpoint, capability_catalog: Path, evidence_root: Path
):
    idea = load_capability_ideas(capability_catalog)[capability_id]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    out = evidence_root / "proposal" / f"{capability_id}-{stamp}-{worker}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True)

    async def go():
        async with GlmClient(live_endpoint, GLM_PROVIDER) as client:
            return await propose_tasks(
                client, {capability_id: idea}, SLOTS, frozenset(Environment), LLMPolicy(), DEFAULT_POLICY
            )

    proposed = asyncio.run(go())
    write_outputs(out, proposed)
    (capability,) = proposed
    assert capability.failure is None, capability.failure
    written = [s for s in capability.batch.slots if not isinstance(s, SlotProposal) or s.completions]
    assert [s.slot for s in written if not isinstance(s, SlotProposal) or s.repair_error is not None] == []
