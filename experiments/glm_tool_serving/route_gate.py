# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Route unprocessed rows using the SHA-pinned deterministic replacement manifest.

The manifest URLs below identify the durable selection evidence: a row is dispositioned
out_of_scope_deterministic only when the published exact-selection manifest
matches on (source_shard, source_row, input_ids_sha256) AND its route is
"deterministic" AND the entry carries a verified_replacement with an archive URI
and SHA-256. The manifest and its summary are downloaded once per run and their
pinned SHA-256s are verified before parsing; on any download/verification
failure the gate is disabled for the run (fail-open to the previous GLM
behavior - correctness is never at stake, only compute) and never loads
unverified data. The gate only ever affects rows still ahead of the cursor;
committed rows, claims, and cursors are untouched.
"""

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

GATE_SUMMARY_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization/"
    "deterministic-20261115-v3/handoff/exact-selection-summary.json"
)
GATE_SUMMARY_SHA256 = "ba3c02f4d63f91ddf6bc0eb4c950f54b5d057a381ff84f3a75a306f1ef8fead5"
GATE_MANIFEST_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization/"
    "deterministic-20261115-v3/handoff/exact-full-corpus-selection-20261115T191015Z.jsonl.gz"
)
GATE_MANIFEST_SHA256 = "fd770d3936d0d2679ebcdde4577004459981f316f4e65c149b9dc05846693402"


@dataclass(frozen=True)
class RouteGate:
    rows: dict[tuple[str, int], tuple[str, int, str]]
    archives: list[dict]


def _fetch_verified(uri: str, expected_sha256: str, destination: Path) -> bool:
    """Download uri to destination; True only if the bytes match the pinned SHA."""
    digest = hashlib.sha256()
    with StoragePath(uri).open("rb") as source, destination.open("wb") as target:
        for chunk in iter(lambda: source.read(1 << 22), b""):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest() == expected_sha256


def load_route_gate(root: Path) -> RouteGate | None:
    """Build {(source_shard, source_row): (input_ids_sha256, archive_index, record_prefix)}
    plus the archive table for route=deterministic rows; None disables the gate."""
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "route-gate-summary.json"
    manifest_path = root / "route-gate-manifest.jsonl.gz"
    try:
        if not _fetch_verified(GATE_SUMMARY_URI, GATE_SUMMARY_SHA256, summary_path):
            print(json.dumps({"phase": "route_gate_unavailable", "reason": "summary sha256 mismatch"}), flush=True)
            return None
        if not _fetch_verified(GATE_MANIFEST_URI, GATE_MANIFEST_SHA256, manifest_path):
            print(json.dumps({"phase": "route_gate_unavailable", "reason": "manifest sha256 mismatch"}), flush=True)
            return None
    except Exception as error:
        print(json.dumps({"phase": "route_gate_unavailable", "reason": str(error)[:300]}), flush=True)
        return None
    started = time.time()
    archives: list[dict] = []
    archive_index: dict[tuple[str, str], int] = {}
    gate: dict[tuple[str, int], tuple[str, int, str]] = {}
    with gzip.open(manifest_path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("route") != "deterministic":
                continue
            replacement = row.get("verified_replacement") or {}
            uri = replacement.get("archive_uri")
            sha = replacement.get("archive_sha256")
            if not uri or not sha:
                continue
            key = (uri, sha)
            index = archive_index.get(key)
            if index is None:
                index = len(archives)
                archive_index[key] = index
                archives.append({"archive_uri": uri, "archive_sha256": sha})
            gate[(row["source_shard"], int(row["source_row"]))] = (
                row["input_ids_sha256"],
                index,
                replacement.get("record_prefix", ""),
            )
    print(
        json.dumps(
            {
                "phase": "route_gate_loaded",
                "deterministic_rows": len(gate),
                "replacement_archives": len(archives),
                "seconds": round(time.time() - started, 1),
            }
        ),
        flush=True,
    )
    return RouteGate(rows=gate, archives=archives)


def gate_result(gate: RouteGate | None, item: dict) -> dict | None:
    """Return the out_of_scope_deterministic disposition when the row matches the
    gate on shard + row + input hash; None routes the row to the normal path."""
    if gate is None:
        return None
    entry = gate.rows.get((item.get("shard"), item.get("row")))
    if entry is None or entry[0] != item.get("input_ids_sha256"):
        return None
    archive = gate.archives[entry[1]]
    return {
        "status": "out_of_scope_deterministic",
        "reason": "deterministic path replacement per the 2026-11-15 refocus selection manifest",
        "replacement_archive_uri": archive["archive_uri"],
        "replacement_archive_sha256": archive["archive_sha256"],
        "replacement_record_prefix": entry[2],
    }
