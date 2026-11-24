# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import fsspec
import pytest

from experiments.glm_tool_serving.fleet import assigned_shards, reconcile_claims, recovery_archive, resume_cursor


@pytest.mark.parametrize("archive_owner", ["current", "predecessor"])
@pytest.mark.parametrize("both_claims", [False, True])
def test_reconcile_claims_stopped_archive_blocks_replay(tmp_path, archive_owner, both_claims):
    prefix = f"memory://{tmp_path.name}/current"
    predecessor = f"memory://{tmp_path.name}/predecessor"
    owners = {"current": prefix, "predecessor": predecessor}
    fs = fsspec.filesystem("memory")
    name = "shard-00000-rows-00000000-00000128"
    for owner in owners if both_claims else [archive_owner]:
        fs.pipe(f"{owners[owner]}/claims/{name}.json", b'{"first":0}')
    fs.pipe(f"{owners[archive_owner]}/batches/{name}.tar.gz", b"committed outputs")
    with pytest.raises(RuntimeError, match="requires reconciliation"):
        reconcile_claims(prefix, predecessor, name, 1)
    assert fs.cat(f"{owners[archive_owner]}/batches/{name}.tar.gz") == b"committed outputs"
    assert fs.exists(f"{owners[archive_owner]}/claims/{name}.json")
    assert not fs.exists(prefix + "/reconciliations")


@pytest.mark.parametrize("claim_owner", ["current", "predecessor"])
def test_reconcile_claims_archive_less_claim_preserves_evidence(tmp_path, claim_owner):
    prefix = f"memory://{tmp_path.name}/current"
    predecessor = f"memory://{tmp_path.name}/predecessor"
    owner = prefix if claim_owner == "current" else predecessor
    fs = fsspec.filesystem("memory")
    claim = f"{owner}/claims/batch.json"
    raw = b'{"first":0,"stop":128}'
    fs.pipe(claim, raw)
    reconcile_claims(prefix, predecessor, "batch", 1)
    evidence = json.loads(fs.cat(f"{prefix}/reconciliations/batch-{claim_owner}-selfheal.json"))
    assert evidence["claim_url"] == claim
    assert evidence["claim_evidence"] == json.loads(raw)
    assert fs.cat(f"{prefix}/reconciled-claims/batch-{claim_owner}") == raw
    assert fs.exists(claim) == (claim_owner == "predecessor")


@pytest.mark.parametrize("owner", ["current", "predecessor"])
def test_recovery_archive_locates_approved_bytes_beside_either_claim(tmp_path, owner):
    prefix = f"memory://{tmp_path.name}/current"
    predecessor = f"memory://{tmp_path.name}/predecessor"
    archive = f"{prefix if owner == 'current' else predecessor}/batches/batch.tar.gz"
    fs = fsspec.filesystem("memory")
    fs.pipe(archive, b"stopped outputs")
    expected = hashlib.sha256(b"stopped outputs").hexdigest()
    assert recovery_archive(prefix, predecessor, "batch", expected) == archive
    fs.pipe(archive, b"overwritten outputs")
    with pytest.raises(ValueError, match="Stopped archive changed"):
        recovery_archive(prefix, predecessor, "batch", expected)


def test_resume_cursor_first_batch_keeps_predecessor_progress(tmp_path):
    prefix = f"memory://{tmp_path.name}/current"
    predecessor = f"memory://{tmp_path.name}/predecessor"
    fs = fsspec.filesystem("memory")
    cursor = {"catalog_sha256": "pinned", "shard": 2, "row": 128}
    fs.pipe(predecessor + "/cursor.json", json.dumps(cursor).encode())
    fs.pipe(prefix + "/claims/batch.json", b"{}")
    fs.pipe(prefix + "/batches/batch.tar.gz", b"stopped")
    resumed, source = resume_cursor(prefix, predecessor, "pinned")
    assert resumed == cursor
    assert source == predecessor + "/cursor.json"
    with pytest.raises(RuntimeError, match="requires reconciliation"):
        reconcile_claims(prefix, predecessor, "batch", 1)
    assert not fs.exists(prefix + "/cursor.json")


@pytest.mark.parametrize("catalog_hash", ["parent", "legacy"])
def test_assigned_shards_fleets_stop_at_ownership_boundary(catalog_hash):
    shards = [{"source": "corpus", "shard": i} for i in range(10)]
    catalog = {"shards": shards}
    tail = {
        "shards": list(reversed(shards[5:])),
        "migration": {"parent_catalog_sha256": "parent", "boundary_catalog_index": 5},
    }
    current = [assigned_shards(catalog, catalog_hash, r, 3, tail) for r in range(1, 4)]
    assert current == [shards[0:5:3], shards[1:5:3], shards[2:5:3]]
    assert {s["shard"] for group in current for s in group}.isdisjoint(s["shard"] for s in tail["shards"])
    assert assigned_shards(tail, "tail", 1, 2, None) == tail["shards"][::2]
    assert catalog["shards"] == shards
