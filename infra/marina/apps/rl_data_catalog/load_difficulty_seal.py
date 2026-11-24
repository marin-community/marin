# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a verified difficulty seal and bind it immutably to an Atlas release."""

import json
from itertools import batched
from typing import Any

from marina.client import AppletQuery
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.seal import verify_snapshot
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.load import ensure_tables
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

CELL_BATCH_SIZE = 32


def load_difficulty_seal(reference: ArtifactRef, query: AppletQuery, load_id: str) -> dict[str, Any]:
    """Reject changed evidence or an existing load bound to another snapshot."""
    manifest = verify_snapshot(reference)
    snapshot_id = manifest["snapshot_id"]
    ensure_tables(query, ("atlas_difficulty_seals", "atlas_difficulty_seal_cells", "atlas_difficulty_seal_loads"))
    query(
        "INSERT INTO atlas_difficulty_seals(snapshot_id,manifest_uri,manifest_sha256,manifest) "
        "VALUES(:snapshot_id,:uri,:sha256,CAST(:manifest AS JSONB)) ON CONFLICT(snapshot_id) DO NOTHING",
        {"snapshot_id": snapshot_id, "uri": reference.uri, "sha256": reference.sha256, "manifest": json.dumps(manifest)},
    )
    stored = query(
        "SELECT manifest_uri,manifest_sha256,manifest FROM atlas_difficulty_seals WHERE snapshot_id=:snapshot_id",
        {"snapshot_id": snapshot_id},
    )["rows"]
    if stored != [{"manifest_uri": reference.uri, "manifest_sha256": reference.sha256, "manifest": manifest}]:
        raise ValueError("Atlas difficulty snapshot ID already names different evidence")
    root = StoragePath(reference.uri).parent
    cells = [json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()]
    for chunk in batched(cells, CELL_BATCH_SIZE):
        rows = [
            {
                "source_id": canonical_source_id(cell["source_id"]),
                "solver": cell["solver"],
                "registry_revision": cell["registry_revision"],
                "cell": cell,
            }
            for cell in chunk
        ]
        query(
            "INSERT INTO atlas_difficulty_seal_cells(snapshot_id,source_id,solver,registry_revision,cell) "
            "SELECT :snapshot_id,r.source_id,r.solver,r.registry_revision,r.cell "
            "FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
            "AS r(source_id TEXT,solver TEXT,registry_revision TEXT,cell JSONB) "
            "ON CONFLICT(snapshot_id,source_id,solver) DO NOTHING",
            {"snapshot_id": snapshot_id, "rows": json.dumps(rows)},
        )
        selected = query(
            "SELECT c.source_id,c.solver,c.cell FROM atlas_difficulty_seal_cells c "
            "JOIN jsonb_to_recordset(CAST(:keys AS JSONB)) AS r(source_id TEXT,solver TEXT) "
            "ON c.source_id=r.source_id AND c.solver=r.solver WHERE c.snapshot_id=:snapshot_id",
            {
                "snapshot_id": snapshot_id,
                "keys": json.dumps([{"source_id": row["source_id"], "solver": row["solver"]} for row in rows]),
            },
        )["rows"]
        found = {(row["source_id"], row["solver"]): row["cell"] for row in selected}
        if found != {(row["source_id"], row["solver"]): row["cell"] for row in rows}:
            raise ValueError("Atlas sealed difficulty cells differ on readback")
    query(
        "INSERT INTO atlas_difficulty_seal_loads(load_id,snapshot_id) VALUES(:load_id,:snapshot_id) "
        "ON CONFLICT(load_id) DO NOTHING",
        {"load_id": load_id, "snapshot_id": snapshot_id},
    )
    selected = query("SELECT snapshot_id FROM atlas_difficulty_seal_loads WHERE load_id=:load_id", {"load_id": load_id})[
        "rows"
    ]
    if selected != [{"snapshot_id": snapshot_id}]:
        raise ValueError("Atlas load is already bound to another difficulty seal")
    return {"snapshot_id": snapshot_id, "receipt": reference.model_dump(), **manifest["summary"]}
