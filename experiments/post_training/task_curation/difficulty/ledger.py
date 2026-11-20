# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed ledger partitions shared by importers and trial workers."""

import hashlib
import json
from dataclasses import asdict

from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.state import ArtifactKind, Gap, GapReason, LedgerRow, Outcome

DEFAULT_LEDGER = "s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_rows(root: str, rows: list[LedgerRow]) -> str | None:
    """Publish an immutable partition; replaying the same evidence writes nothing."""
    if not rows:
        return None
    payload = "".join(
        json.dumps(asdict(row), sort_keys=True) + "\n" for row in sorted(rows, key=lambda row: row.attempt_id)
    )
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = StoragePath(root) / "v1" / "rows" / f"{key}.jsonl"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Ledger partition content changed: {path}")
        return str(path)
    path.parent.mkdirs(exist_ok=True)
    path.write_text(payload)
    if path.read_text() != payload:
        raise ValueError(f"Ledger readback differs: {path}")
    return str(path)


def read_rows(root: str) -> list[LedgerRow]:
    rows: dict[str, LedgerRow] = {}
    for path in sorted((StoragePath(root) / "v1" / "rows" / "*.jsonl").glob(), key=str):
        for line in path.read_text().splitlines():
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
                )
                for gap in value.get("secondary_gaps", [])
            )
            row = LedgerRow(**value)
            if row.attempt_id in rows and rows[row.attempt_id] != row:
                raise ValueError(f"Conflicting ledger evidence: {row.attempt_id}")
            rows[row.attempt_id] = row
    return sorted(rows.values(), key=lambda row: (row.recorded_at, row.attempt_id))
