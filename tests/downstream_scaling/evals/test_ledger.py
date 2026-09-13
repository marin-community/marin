# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time
from dataclasses import dataclass

import pytest
from rigging.filesystem.distributed_lock import HEARTBEAT_TIMEOUT, create_lock

from experiments.downstream_scaling.evals.framework.xregion import ledger


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str


def chunk_spec(chunk_id: int) -> ChunkSpec:
    return ChunkSpec(
        chunk_id=chunk_id,
        chunk_start=chunk_id * 10,
        chunk_end=(chunk_id + 1) * 10,
        output_path=f"chunks/chunk-{chunk_id:06d}.jsonl.gz",
    )


def ledger_path(tmp_path) -> str:
    return str(tmp_path / "ledger")


def state_lock_path(path: str, chunk_id: int) -> str:
    return f"{path}/chunks/{chunk_id}.json.lock"


def test_ensure_manifest_rejects_mismatched_existing_manifest(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])

    with pytest.raises(ValueError, match="does not match"):
        ledger.ensure_manifest(path, [chunk_spec(1)])


def test_claim_next_chunk_claims_missing_state(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])

    with ledger.claim_next_chunk(path, "worker-a") as claim:
        assert claim is not None
        assert claim.chunk_id == 0
        assert ledger.read_chunk_state(path, 0) == ledger.ChunkState(
            status=ledger.ChunkStatus.CLAIMED,
            owner="worker-a",
        )


def test_claim_next_chunk_skips_done_state(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])
    ledger.write_chunk_state(path, 0, ledger.ChunkState(status=ledger.ChunkStatus.DONE, owner="worker-a"))

    with ledger.claim_next_chunk(path, "worker-b") as claim:
        assert claim is None


def test_claim_next_chunk_skips_live_claimed_state(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])

    with ledger.claim_next_chunk(path, "worker-a") as claim:
        assert claim is not None
        with ledger.claim_next_chunk(path, "worker-b") as competing_claim:
            assert competing_claim is None


def test_claim_next_chunk_reclaims_claimed_state_after_lock_takeover(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])
    ledger.write_chunk_state(path, 0, ledger.ChunkState(status=ledger.ChunkStatus.CLAIMED, owner="worker-a"))

    lock_path = state_lock_path(path, 0)
    old_lock = create_lock(lock_path, "worker-a")
    assert old_lock.try_acquire()
    with open(lock_path, "w") as f:
        json.dump({"worker_id": "worker-a", "timestamp": time.time() - HEARTBEAT_TIMEOUT - 1}, f)

    with ledger.claim_next_chunk(path, "worker-b") as claim:
        assert claim is not None
        assert claim.chunk_id == 0
        assert ledger.read_chunk_state(path, 0) == ledger.ChunkState(
            status=ledger.ChunkStatus.CLAIMED,
            owner="worker-b",
        )


def test_mark_done_records_terminal_progress(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])

    with ledger.claim_next_chunk(path, "worker-a") as claim:
        assert claim is not None
        ledger.mark_done(claim)

    assert ledger.done_chunk_ids(path) == [0]
    assert ledger.summarize(path) == ledger.LedgerSummary(total=1, claimed=0, done=1)


def test_exception_before_mark_done_leaves_chunk_not_done(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0)])

    with ledger.claim_next_chunk(path, "worker-a") as claim:
        assert claim is not None

    assert ledger.done_chunk_ids(path) == []
    assert ledger.read_chunk_state(path, 0) == ledger.ChunkState(
        status=ledger.ChunkStatus.CLAIMED,
        owner="worker-a",
    )


def test_resume_claims_only_chunks_not_done(tmp_path):
    path = ledger_path(tmp_path)
    ledger.ensure_manifest(path, [chunk_spec(0), chunk_spec(1)])
    ledger.write_chunk_state(path, 0, ledger.ChunkState(status=ledger.ChunkStatus.DONE, owner="worker-a"))

    with ledger.claim_next_chunk(path, "worker-b") as claim:
        assert claim is not None
        assert claim.chunk_id == 1
