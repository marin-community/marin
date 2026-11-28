# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed ledger partitions shared by importers and trial workers."""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import cache

from finestore.rl.difficulty import PROTOCOL_ID, Solver
from pydantic import TypeAdapter
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.resource_deviations import ResourceDeviation
from experiments.post_training.task_curation.difficulty.state import (
    ActionFailure,
    ArtifactKind,
    Gap,
    GapReason,
    LedgerRow,
    Outcome,
)
from infra.marina.apps.rl_data_catalog.release_evidence import json_sha256

LEDGER_VERSION = "v1"
ACTIONS_DIRECTORY = "actions"
IMPORTS_DIRECTORY = "imports"
RESULTS_DIRECTORY = "results"
ESCALATIONS_DIRECTORY = "escalations"
PANELS_DIRECTORY = "panels"
REPAIR_VERIFICATIONS_DIRECTORY = "repair-verifications"
DEFAULT_LEDGER = "s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger"
LEDGER_READ_WORKERS = 32
LOGGER = logging.getLogger(__name__)
digest = json_sha256


def record_action_failure(ledger: str, manifest_uri: str, error: Exception) -> None:
    """Persist a child failure without replacing its subprocess's more specific receipt."""
    target = StoragePath(ledger) / LEDGER_VERSION / RESULTS_DIRECTORY / StoragePath(manifest_uri).name
    if target.exists():
        return
    reason = error.reason if isinstance(error, ActionFailure) else GapReason.CONTROLLER_ACTION_FAILED
    payload = {
        "status": "failed",
        "reason": reason,
        "error_identity": type(error).__name__,
        "detail": str(error) if isinstance(error, ActionFailure) else type(error).__name__,
    }
    target.parent.mkdirs()
    target.write_text(json.dumps(payload, sort_keys=True) + "\n")


def repair_verification_sources(ledger: str) -> frozenset[str]:
    """Sources a recorded repair verification covers, keyed by its typed escalation linkage."""
    verified: set[str] = set()
    for path in sorted(
        (StoragePath(ledger) / LEDGER_VERSION / REPAIR_VERIFICATIONS_DIRECTORY / "*.json").glob(), key=str
    ):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        sources = record.get("sources")
        if isinstance(sources, list) and record.get("escalation"):
            verified.update(str(source) for source in sources)
    return frozenset(verified)


def row_payload(row: LedgerRow) -> dict:
    """Keep historical wire bytes unchanged when optional conditions are absent."""
    payload = asdict(row)
    if payload["resource_deviation"] is None:
        del payload["resource_deviation"]
    if payload["solver_protocol_id"] == PROTOCOL_ID:
        del payload["solver_protocol_id"]
    if payload["supersedes_attempt_id"] is None:
        del payload["supersedes_attempt_id"]
    return payload


def append_rows(root: str, rows: list[LedgerRow]) -> str | None:
    """Return the immutable partition URI, or None for an empty batch.

    Replaying identical evidence returns its existing partition without writing.
    """
    if not rows:
        return None
    if (StoragePath(root) / LEDGER_VERSION / "closed.json").exists():
        raise ValueError(f"Difficulty ledger is sealed: {root}")
    payload = "".join(
        json.dumps(row_payload(row), sort_keys=True) + "\n" for row in sorted(rows, key=lambda row: row.attempt_id)
    )
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = StoragePath(root) / LEDGER_VERSION / "rows" / f"{key}.jsonl"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Ledger partition content changed: {path}")
        return str(path)
    path.parent.mkdirs(exist_ok=True)
    path.write_text(payload)
    if path.read_text() != payload:
        raise ValueError(f"Ledger readback differs: {path}")
    return str(path)


@cache
def partition_rows(uri: str) -> tuple[LedgerRow, ...]:
    """Decode one immutable content-addressed partition once per controller process."""
    return decode_partition_rows(StoragePath(uri).read_bytes())


def decode_partition_rows(payload: bytes) -> tuple[LedgerRow, ...]:
    """Decode ledger bytes without changing their recorded evidence."""
    decoded = []
    for line in payload.splitlines():
        value = json.loads(line)
        value["solver"] = Solver(value["solver"])
        value["outcome"] = Outcome(value["outcome"])
        value["reason"] = GapReason(value["reason"]) if value["reason"] else None
        if value.get("artifact_kind"):
            value["artifact_kind"] = ArtifactKind(value["artifact_kind"])
        value["secondary_gaps"] = tuple(
            Gap(
                GapReason(gap["reason"]),
                gap["artifact_uri"],
                gap["judge_model"],
                gap["capability"],
                ArtifactKind(gap["artifact_kind"]) if gap.get("artifact_kind") else None,
                gap.get("judge_temperature"),
                gap.get("error_identity"),
            )
            for gap in value.get("secondary_gaps", [])
        )
        if value.get("resource_deviation") is not None:
            value["resource_deviation"] = TypeAdapter(ResourceDeviation).validate_python(value["resource_deviation"])
        row = LedgerRow(**value)
        decoded.append(row)
    return tuple(decoded)


def read_rows(root: str) -> list[LedgerRow]:
    started = time.monotonic()
    before = partition_rows.cache_info()
    rows: dict[str, LedgerRow] = {}
    paths = sorted(str(path) for path in (StoragePath(root) / LEDGER_VERSION / "rows" / "*.jsonl").glob())
    with ThreadPoolExecutor(max_workers=LEDGER_READ_WORKERS) as pool:
        for partition in pool.map(partition_rows, paths):
            for row in partition:
                if row.attempt_id in rows and rows[row.attempt_id] != row:
                    raise ValueError(f"Conflicting ledger evidence: {row.attempt_id}")
                rows[row.attempt_id] = row
    after = partition_rows.cache_info()
    LOGGER.info(
        "DIFFICULTY_LEDGER_READ %s",
        json.dumps(
            {
                "partitions": len(paths),
                "rows": len(rows),
                "partition_cache_hits": after.hits - before.hits,
                "partition_cache_misses": after.misses - before.misses,
                "cached_partitions": after.currsize,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
    )
    return sorted(rows.values(), key=lambda row: (row.recorded_at, row.attempt_id))
