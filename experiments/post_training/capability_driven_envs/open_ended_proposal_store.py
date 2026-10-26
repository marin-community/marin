# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-identified local persistence for open-ended proposal portfolios."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from rigging.filesystem.atomic import atomic_rename
from taskforge.content_hash import canonical_json, digest, sha256_hex
from taskforge.stage10_proposal.source import ProposalBatch, SlotProposal

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalSource,
    proposal_id,
    source_ref,
)

PROPOSAL_SOURCE_VERSION = "open-ended-proposal-source/1"
PROPOSAL_BATCH_FORMAT = "open-ended-proposal-batch/1"
_BATCH = TypeAdapter(ProposalBatch)


class ProposalBatchIdentity(BaseModel):
    """Inputs under which a proposal portfolio may be reused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_version: Literal["open-ended-proposal-source/1"] = PROPOSAL_SOURCE_VERSION
    bundle_id: str = Field(min_length=1)
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    slot_count: int = Field(gt=0)

    @property
    def cache_key(self) -> str:
        """Return the content digest of every input that changes the portfolio."""
        return digest(self)


class StoredProposalBatch(BaseModel):
    """One self-verifying proposal batch and its complete generation identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["open-ended-proposal-batch/1"] = PROPOSAL_BATCH_FORMAT
    identity: ProposalBatchIdentity
    batch_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_payload(self) -> Self:
        if self.batch_json != canonical_batch_json(self.batch()):
            raise ValueError("batch_json is not Taskforge canonical JSON")
        observed = sha256_hex(self.batch_json.encode())
        if observed != self.batch_digest:
            raise ValueError(f"proposal batch digest differs: expected {self.batch_digest}, got {observed}")
        return self

    def batch(self) -> ProposalBatch:
        """Decode the stored proposal batch."""
        return _BATCH.validate_json(self.batch_json)


def proposal_batch_identity(
    bundle: OpenEndedTaskBundle,
    source: OpenEndedProposalSource,
    slot_count: int,
) -> ProposalBatchIdentity:
    """Return the complete cache identity for one source invocation."""
    return ProposalBatchIdentity(
        bundle_id=bundle.bundle_id,
        bundle_digest=bundle.content_digest,
        model=source.model.model,
        policy_digest=digest(source.policy),
        config_digest=digest(source.config),
        slot_count=slot_count,
    )


def canonical_batch_json(batch: ProposalBatch) -> str:
    """Return canonical JSON for a generic Taskforge proposal batch."""
    return canonical_json(_BATCH.dump_python(batch, mode="json"))


def stored_batch(identity: ProposalBatchIdentity, batch: ProposalBatch) -> StoredProposalBatch:
    """Package a proposal batch with its generation identity and payload digest."""
    payload = canonical_batch_json(batch)
    return StoredProposalBatch(identity=identity, batch_digest=sha256_hex(payload.encode()), batch_json=payload)


def write_proposal_batch(path: Path, record: StoredProposalBatch) -> None:
    """Atomically write a proposal-batch record to a local path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_rename(str(path)) as temporary:
        Path(temporary).write_text(canonical_json(record))


def read_proposal_batch(path: Path) -> StoredProposalBatch:
    """Read and validate a local proposal-batch record."""
    return StoredProposalBatch.model_validate_json(path.read_bytes())


async def load_or_propose(
    path: Path,
    source: OpenEndedProposalSource,
    bundle: OpenEndedTaskBundle,
    slot_count: int,
) -> ProposalBatch:
    """Load the exact persisted portfolio or generate and atomically persist it once.

    Reusing ``path`` with different generation inputs is an error rather than a
    request to resample. Distributed locking belongs to the future durable store;
    this local implementation assumes one owner for a bundle path.
    """
    identity = proposal_batch_identity(bundle, source, slot_count)
    if path.exists():
        record = read_proposal_batch(path)
        if record.identity != identity:
            raise ValueError(
                f"proposal batch at {path} has cache key {record.identity.cache_key}, not {identity.cache_key}"
            )
        batch = record.batch()
        _validate_batch(batch, bundle, slot_count)
        return batch
    batch = await source.propose(bundle, slot_count)
    _validate_batch(batch, bundle, slot_count)
    write_proposal_batch(path, stored_batch(identity, batch))
    return batch


def _validate_batch(batch: ProposalBatch, bundle: OpenEndedTaskBundle, slot_count: int) -> None:
    slots = [outcome.slot for outcome in batch.slots]
    if slots != list(range(1, slot_count + 1)):
        raise ValueError(f"proposal batch slots must be 1..{slot_count} in order, got {slots}")
    for outcome in batch.slots:
        if not isinstance(outcome, SlotProposal):
            continue
        expected_id = proposal_id(bundle, outcome.slot)
        if outcome.proposal.header.id != expected_id or outcome.proposal.header.source != source_ref(bundle):
            raise ValueError(f"proposal batch slot {outcome.slot} does not belong to bundle {bundle.bundle_id}")
