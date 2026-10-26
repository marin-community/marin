# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one hydrated open-ended bundle through proposal and Taskforge workpieces."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from taskforge.content_hash import digest
from taskforge.stage10_proposal.source import SlotFailure, SlotProposal
from taskforge.stage_coordinator.coordinator import StageCoordinator, run_workpiece
from taskforge.stage_coordinator.event_log import InMemoryEventLog
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedBuildInput, OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_proposal_source import OpenEndedProposalSource
from experiments.post_training.capability_driven_envs.open_ended_proposal_store import (
    load_or_propose,
    proposal_batch_identity,
)


class SlotRunStatus(StrEnum):
    """Which boundary ended one requested proposal slot."""

    NULL = "null"
    PROPOSAL_FAILED = "proposal_failed"
    WORKPIECE = "workpiece"


class OpenEndedSlotResult(BaseModel):
    """Compact result row for one bundle slot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: int
    status: SlotRunStatus
    proposal_id: str | None = None
    workpiece_id: str | None = None
    terminal: Terminal | None = None
    detail: str | None = None


class OpenEndedBundleResult(BaseModel):
    """Compact result of processing every slot in one persisted proposal batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: str
    bundle_digest: str
    proposal_cache_key: str
    slots: tuple[OpenEndedSlotResult, ...]


@dataclass(frozen=True)
class OpenEndedRuntime:
    """Source and Taskforge coordinator construction for a local bundle run."""

    proposal_source: OpenEndedProposalSource
    coordinator: Callable[[OpenEndedTaskBundle], StageCoordinator]


def workpiece_id(bundle: OpenEndedTaskBundle, slot: int) -> str:
    """Return a filesystem-safe stable workpiece identity for one bundle slot."""
    return f"open-ended-{bundle.content_digest[:24]}-{slot:04d}"


async def run_open_ended_bundle(
    bundle: OpenEndedTaskBundle,
    runtime: OpenEndedRuntime,
    *,
    slot_count: int,
    root: Path,
) -> OpenEndedBundleResult:
    """Propose once and run every non-null proposal through ordinary Taskforge.

    Proposal failures become compact slot results. Taskforge infrastructure and
    model failures propagate so the caller can retry; the coordinator's event
    store makes completed and interrupted workpieces resumable.
    """
    identity = proposal_batch_identity(bundle, runtime.proposal_source, slot_count)
    batch_path = root / "proposal-batches" / bundle.content_digest / f"{identity.cache_key}.json"
    batch = await load_or_propose(batch_path, runtime.proposal_source, bundle, slot_count)
    coordinator = runtime.coordinator(bundle)
    actual_build_input = OpenEndedBuildInput.model_validate(coordinator.build_input)
    bundle_input = actual_build_input.model_copy(update={"judge_model": None})
    if digest(bundle_input) != digest(bundle.build_input()):
        raise ValueError("open-ended coordinator build_input does not match the bundle")

    results = []
    for outcome in batch.slots:
        if isinstance(outcome, SlotFailure):
            results.append(
                OpenEndedSlotResult(slot=outcome.slot, status=SlotRunStatus.PROPOSAL_FAILED, detail=outcome.error)
            )
            continue
        assert isinstance(outcome, SlotProposal)
        proposal = outcome.proposal
        if proposal.header.null_reason is not None:
            results.append(
                OpenEndedSlotResult(
                    slot=outcome.slot,
                    status=SlotRunStatus.NULL,
                    proposal_id=proposal.header.id,
                    detail=proposal.header.null_reason,
                )
            )
            continue
        identifier = workpiece_id(bundle, outcome.slot)
        terminal = await run_workpiece(coordinator, identifier, proposal, InMemoryEventLog())
        results.append(
            OpenEndedSlotResult(
                slot=outcome.slot,
                status=SlotRunStatus.WORKPIECE,
                proposal_id=proposal.header.id,
                workpiece_id=identifier,
                terminal=terminal,
            )
        )
    return OpenEndedBundleResult(
        bundle_id=bundle.bundle_id,
        bundle_digest=bundle.content_digest,
        proposal_cache_key=identity.cache_key,
        slots=tuple(results),
    )
