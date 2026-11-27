# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze an append-only difficulty ledger into a verified, immutable snapshot."""

import argparse
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID
from pydantic import TypeAdapter
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.catalog import DEFAULT_APPLET, served_catalog
from experiments.post_training.task_curation.difficulty.controller import controller_environment
from experiments.post_training.task_curation.difficulty.ledger import (
    DEFAULT_LEDGER,
    LEDGER_READ_WORKERS,
    decode_partition_rows,
    digest,
)
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CellStatus,
    CoveragePolicy,
    GapReason,
    LedgerRow,
    Outcome,
)
from experiments.post_training.task_curation.difficulty_config import JUDGE_ROUTE_REPAIRS
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def json_lines(rows: list[dict]) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def ledger_partitions(ledger: str) -> tuple[str, ...]:
    return tuple(sorted(map(str, (StoragePath(ledger) / "v1/rows/*.jsonl").glob())))


def partition_payload(uri: str) -> tuple[dict, bytes]:
    """Read a content-addressed partition once and verify its original bytes."""
    payload = StoragePath(uri).read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    if StoragePath(uri).name != sha256 + ".jsonl":
        raise ValueError(f"Ledger partition content hash differs: {uri}")
    return {"uri": uri, "sha256": sha256, "bytes": len(payload)}, payload


def verified_partition(uri: str) -> tuple[dict, tuple[LedgerRow, ...]]:
    reference, payload = partition_payload(uri)
    return reference, decode_partition_rows(payload)


def copy_snapshot_partition(uri: str, selected: StoragePath, boundary: datetime) -> dict:
    """Validate a captured partition and preserve its pre-cutoff row bytes."""
    reference, payload = partition_payload(uri)
    decode_partition_rows(payload)
    kept = b"".join(
        line + b"\n"
        for line in payload.splitlines()
        if datetime.fromisoformat(json.loads(line)["recorded_at"]) < boundary
    )
    if kept:
        write_immutable_bytes(selected / "v1/rows" / (hashlib.sha256(kept).hexdigest() + ".jsonl"), kept)
    return reference


def snapshot_cells(rows: list[LedgerRow], catalog: list[CatalogSource], provenance: dict) -> list[dict]:
    """Keep incomplete cells unmeasured while retaining their observed attempt rates."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.source_id, row.registry_revision, row.solver)].append(row)
    output = []
    for cell in cells_for_catalog(rows, catalog, CoveragePolicy()):
        source = cell.source
        selected = set(source.tasks)
        attempts = [
            row
            for row in grouped[(source.source_id, source.registry_revision, cell.solver)]
            if (row.task_id, row.task_sha256) in selected and row.outcome == Outcome.SCORED
        ]
        tasks = {(row.task_id, row.task_sha256) for row in attempts}
        invalid = [row.attempt_id for row in attempts if row.reward is None or not 0 <= row.reward <= 1]
        rewards = [row.reward for row in attempts if row.reward is not None and 0 <= row.reward <= 1]
        hold = provenance.get("holds", {}).get(source.source_id)
        covered = cell.state == CellStatus.MEASURED and hold is None and not invalid
        gaps = [asdict(gap) for gap in cell.gaps]
        if invalid:
            gaps.append(
                {
                    "reason": GapReason.GRADER_INFRASTRUCTURE_FAILURE.value,
                    "detail": (
                        "Reward scale is unqualified: scored reward outside [0,1]; "
                        "original values preserved without clipping."
                    ),
                    "attempt_ids": invalid,
                }
            )
        if hold is not None:
            GapReason(hold["reason"])
            gaps.append(hold)
        proven = [fact for fact in provenance.get("proven_gap_classes", []) if source.source_id in fact["source_ids"]]
        if not covered:
            for fact in proven:
                GapReason(fact["reason"])
                gaps.append({"reason": fact["reason"], "detail": fact["detail"]})
        rate = sum(reward == 1 for reward in rewards) / len(rewards) if rewards and not invalid else None
        output.append(
            {
                "source_id": source.source_id,
                "registry_revision": source.registry_revision,
                "solver": cell.solver.value,
                "protocol_id": provenance.get("solver_protocol", {}).get("id", PROTOCOL_ID),
                "state": cell.state.value if hold is None and not invalid else CellStatus.BLOCKED.value,
                "measured": covered,
                "difficulty": 1 - rate if covered and rate is not None else None,
                "pass_rate": rate if covered else None,
                "observed_pass_rate": rate,
                "mean_reward": sum(rewards) / len(rewards) if rewards and not invalid else None,
                "unqualified_reward_attempts": invalid,
                "scored_attempts": len(attempts),
                "scored_tasks": len(tasks),
                "solved_attempts": sum(reward == 1 for reward in rewards),
                "selected_tasks": len(source.tasks),
                "missing_attempts": cell.n_missing,
                "gaps": gaps,
                "queue_uri": source.queue_uri,
                "queue_sha256": source.queue_sha256,
                "evidence_ids": list(cell.evidence_ids),
            }
        )
    return output


def write_immutable_bytes(path: StoragePath, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Immutable snapshot conflict: {path}")
    else:
        path.write_bytes(payload, auto_mkdir=True)
    if path.read_bytes() != payload:
        raise ValueError(f"Snapshot readback differs: {path}")


def seal_ledger(
    ledger: str,
    catalog: list[CatalogSource],
    destination: str,
    provenance: dict,
    sealed_at: str,
) -> ArtifactRef:
    """Hash all partitions, reject a moving inventory, and seal coverage plus provenance."""
    closure = StoragePath(ledger) / "v1/closed.json"
    if closure.exists():
        raise ValueError(f"Difficulty ledger is already sealed: {ledger}")
    before = ledger_partitions(ledger)
    if not before:
        raise ValueError("Cannot seal an empty ledger inventory")
    unique = {}
    partitions = []
    with ThreadPoolExecutor(max_workers=LEDGER_READ_WORKERS) as pool:
        for reference, records in pool.map(verified_partition, before):
            partitions.append(reference)
            for row in records:
                if row.attempt_id in unique and unique[row.attempt_id] != row:
                    raise ValueError(f"Conflicting attempt identity: {row.attempt_id}")
                unique[row.attempt_id] = row
    if ledger_partitions(ledger) != before:
        raise ValueError("Ledger changed during sealing; stop writers before retrying")
    rows = sorted(
        unique.values(), key=lambda row: (datetime.fromisoformat(row.recorded_at).astimezone(UTC), row.attempt_id)
    )
    cells = snapshot_cells(rows, catalog, provenance)
    payloads = {
        "ledger.jsonl": json_lines([asdict(row) for row in rows]),
        "attempts.jsonl": json_lines(
            [
                asdict(row)
                for row in rows
                if row.task_sha256 is not None and row.outcome in {Outcome.SCORED, Outcome.UNSCORED, Outcome.OBSERVATION}
            ]
        ),
        "catalog.json": json_bytes([asdict(source) for source in catalog]),
        "cells.jsonl": json_lines(cells),
        "scored-cells.jsonl": json_lines([cell for cell in cells if cell["measured"]]),
        "uncovered-cells.jsonl": json_lines([cell for cell in cells if not cell["measured"]]),
        "repairs.jsonl": json_lines([asdict(row) for row in rows if row.outcome == Outcome.REPAIRED]),
        "judge-route-repairs.json": json_bytes([asdict(repair) for repair in JUDGE_ROUTE_REPAIRS]),
    }
    side_objects = []
    for directory in ("repair-verifications-parked", "repair-verifications"):
        for path in sorted((StoragePath(ledger) / "v1" / directory / "**/*.json").glob(), key=str):
            raw = path.read_bytes()
            sha256 = hashlib.sha256(raw).hexdigest()
            name = f"provenance/{sha256}.json"
            payloads[name] = raw
            side_objects.append({"source_uri": str(path), "file": name, "sha256": sha256})
    if provenance.get("require_parked_hold") and not any(
        "repair-verifications-parked/" in row["source_uri"] for row in side_objects
    ):
        raise ValueError("Required parked repair-verification provenance was not found")
    payloads["provenance.json"] = json_bytes(
        {**provenance, "side_objects": side_objects, "ledger_partitions": partitions}
    )
    content = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": sealed_at,
        "ledger": ledger,
        "final_sequence_number": len(rows),
        "sequence_order": "recorded_at in UTC, then attempt_id; seal-defined sequence, not a backend counter",
        "partition_inventory_sha256": digest(partitions),
        "summary": {
            "sources": len(catalog),
            "cells": len(cells),
            "covered_cells": sum(cell["measured"] for cell in cells),
            "partial_cells": sum(not cell["measured"] and cell["scored_attempts"] > 0 for cell in cells),
            "outcomes": dict(Counter(row.outcome.value for row in rows)),
        },
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in sorted(payloads.items())
        },
    }
    snapshot_id = digest(content)
    root = StoragePath(destination) / snapshot_id
    for name, raw in payloads.items():
        write_immutable_bytes(root / name, raw)
    manifest = {"snapshot_id": snapshot_id, **content}
    path = root / "manifest.json"
    raw = json_bytes(manifest)
    write_immutable_bytes(path, raw)
    reference = ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())
    verify_snapshot(reference)
    if ledger_partitions(ledger) != before:
        raise ValueError("Ledger changed before closure; snapshot was not selected")
    write_immutable_bytes(closure, json_bytes({"snapshot": reference.model_dump(), "final_sequence_number": len(rows)}))
    return reference


def capture_snapshot_inventory(ledger: str, destination: str, provenance: dict, cutoff: str) -> dict:
    """Fix the release inventory before catalog reads or snapshot materialization."""
    boundary = datetime.fromisoformat(cutoff)
    if boundary.tzinfo is None:
        raise ValueError("Snapshot cutoff must include its timezone")
    if datetime.now(UTC) < boundary:
        raise ValueError("Cannot capture a future cutoff")
    root = StoragePath(destination)
    capture = root / "capture.json"
    if not capture.exists():
        inventory = ledger_partitions(ledger)
        if not inventory:
            raise ValueError("Cannot snapshot an empty ledger")
        side_objects = []
        for directory in ("repair-verifications-parked", "repair-verifications"):
            for path in sorted((StoragePath(ledger) / "v1" / directory / "**/*.json").glob(), key=str):
                payload = path.read_bytes()
                copied = root / "input-provenance" / hashlib.sha256(payload).hexdigest()
                write_immutable_bytes(copied, payload)
                side_objects.append(
                    {
                        "source_uri": str(path),
                        "captured_uri": str(copied),
                        "relative_path": str(path).removeprefix(str(StoragePath(ledger)) + "/"),
                    }
                )
        write_immutable_bytes(
            capture,
            json_bytes(
                {
                    "ledger": ledger,
                    "cutoff": cutoff,
                    "captured_at": datetime.now(UTC).isoformat(),
                    "partitions": inventory,
                    "side_objects": side_objects,
                    "provenance": provenance,
                }
            ),
        )
    frozen = json.loads(capture.read_text())
    if (frozen["ledger"], frozen["cutoff"]) != (ledger, cutoff):
        raise ValueError("Snapshot destination belongs to another cutoff or ledger")
    return frozen


def snapshot_active_ledger(
    ledger: str,
    catalog: list[CatalogSource],
    destination: str,
    provenance: dict,
    cutoff: str,
) -> ArtifactRef:
    """Seal a frozen copy of pre-cutoff rows while the source ledger stays writable.

    The first capture fixes the partition inventory; retries reuse it. Late writes
    absent from that inventory are excluded even if their recorded time is older.
    Only the private snapshot copy receives a closed.json marker.
    """
    frozen = capture_snapshot_inventory(ledger, destination, provenance, cutoff)
    boundary = datetime.fromisoformat(cutoff)
    root = StoragePath(destination)
    capture = root / "capture.json"
    catalog_path = root / "catalog.json"
    if not catalog_path.exists():
        write_immutable_bytes(catalog_path, json_bytes([asdict(source) for source in catalog]))
    catalog = TypeAdapter(list[CatalogSource]).validate_json(catalog_path.read_text())
    selected = root / "selected-ledger"
    result = root / "selected-snapshot.json"
    if result.exists():
        reference = ArtifactRef.model_validate_json(result.read_text())
        verify_snapshot(reference)
        return reference
    source_inventory = []
    copy_partition = partial(copy_snapshot_partition, selected=selected, boundary=boundary)
    with ThreadPoolExecutor(max_workers=LEDGER_READ_WORKERS) as pool:
        # map preserves the frozen inventory order, so concurrency cannot change the snapshot identity.
        for reference in pool.map(copy_partition, frozen["partitions"]):
            source_inventory.append(reference)
            if len(source_inventory) % 1024 == 0:
                LOGGER.info("SNAPSHOT_PARTITIONS_SELECTED %d/%d", len(source_inventory), len(frozen["partitions"]))
    for item in frozen["side_objects"]:
        payload = StoragePath(item["captured_uri"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != StoragePath(item["captured_uri"]).name:
            raise ValueError("Captured repair provenance changed")
        write_immutable_bytes(selected / item["relative_path"], payload)
    provenance = {
        **frozen["provenance"],
        "active_ledger": ledger,
        "cutoff": cutoff,
        "capture": str(capture),
        "captured_at": frozen["captured_at"],
        "source_partition_inventory": source_inventory,
        "source_ledger_remains_open": True,
        "selection": "recorded_at < cutoff, in the first captured partition inventory; late arrivals excluded",
    }
    closure = selected / "v1/closed.json"
    if closure.exists():
        reference = ArtifactRef.model_validate(json.loads(closure.read_text())["snapshot"])
        verify_snapshot(reference)
    else:
        reference = seal_ledger(str(selected), catalog, str(root / "snapshots"), provenance, cutoff)
    write_immutable_bytes(result, json_bytes(reference.model_dump()))
    return reference


def verify_snapshot(reference: ArtifactRef) -> dict:
    """Read back every sealed file, including parked holds and repair evidence."""
    manifest = read_ref(reference)
    content = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    if manifest["schema_version"] != SCHEMA_VERSION or digest(content) != manifest["snapshot_id"]:
        raise ValueError("Snapshot identity does not match its content")
    root = StoragePath(reference.uri).parent
    for name, expected in manifest["files"].items():
        raw = (root / name).read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Sealed snapshot file differs: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--applet-id", default=DEFAULT_APPLET)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--secrets-env", type=Path)
    args = parser.parse_args()
    if args.secrets_env is not None:
        os.environ.update(controller_environment(args.secrets_env))
    configure_coreweave_s3()
    reference = seal_ledger(
        args.ledger,
        served_catalog(args.applet_id),
        args.destination,
        json.loads(args.provenance.read_text()),
        datetime.now(UTC).isoformat(),
    )
    args.receipt.write_text(reference.model_dump_json(indent=2) + "\n")
    print(json.dumps({"reference": reference.model_dump(), "summary": read_ref(reference)["summary"]}))


if __name__ == "__main__":
    main()
