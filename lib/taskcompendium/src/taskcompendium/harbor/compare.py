# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Harbor task outcomes and archive contents."""

import difflib
import hashlib
import io
import json
import sqlite3
import tarfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from taskcompendium.harbor.snapshots import FileSnapshot, TaskSnapshot, archive_snapshot, json_temporal

MAX_REVIEW_TEXT_BYTES = 65536


@dataclass(frozen=True)
class Difference:
    category: str
    kind: str
    path: str
    baseline: Any
    candidate: Any


def file_category(path: str) -> str:
    if path.startswith("oracle/"):
        return "oracle"
    if path == "task/instruction.md":
        return "instruction"
    if path == "task/task.toml":
        return "task_config"
    if path.startswith("task/environment/"):
        return "actor_environment"
    if path.startswith("task/tests/"):
        return "grader_config" if path.endswith(".toml") else "private_test"
    return "public_source"


def toml_identity(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, default=json_temporal).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "keys": sorted(value)}


def file_identity(value: FileSnapshot | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {**asdict(value), "parsed_toml": toml_identity(value.parsed_toml)}


def archive_contents(blob: bytes | None, namespace: str) -> dict[str, bytes]:
    if blob is None:
        return {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        files = {}
        for member in archive:
            if member.isfile():
                handle = archive.extractfile(member)
                assert handle is not None
                files[f"{namespace}/{member.name}"] = handle.read()
        return files


def write_archive_diff(baseline: bytes | None, candidate: bytes | None, namespace: str, output: TextIO) -> None:
    """Write selected-task member changes and bounded text diffs for manual review.

    Each side contributes at most 64 KiB of UTF-8 text per file. Truncation is
    explicit, and the member metadata retains hashes of the complete contents.
    """
    before, after = archive_snapshot(baseline, namespace), archive_snapshot(candidate, namespace)
    old_contents, new_contents = archive_contents(baseline, namespace), archive_contents(candidate, namespace)
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        output.write(f"\n=== {path} ===\n")
        output.write(
            json.dumps({"baseline": file_identity(before.get(path)), "candidate": file_identity(after.get(path))})
        )
        output.write("\n")
        old, new = old_contents.get(path, b""), new_contents.get(path, b"")
        if old == new:
            continue
        try:
            old_text, new_text = old.decode(), new.decode()
        except UnicodeDecodeError:
            output.write("Binary content differs; complete hashes are above.\n")
            continue
        if len(old) > MAX_REVIEW_TEXT_BYTES or len(new) > MAX_REVIEW_TEXT_BYTES:
            output.write(f"Text diff truncated to 65536 bytes per side (baseline={len(old)}, candidate={len(new)}).\n")
            old_text, new_text = old[:MAX_REVIEW_TEXT_BYTES].decode(errors="replace"), new[
                :MAX_REVIEW_TEXT_BYTES
            ].decode(errors="replace")
        for line in difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(), f"baseline/{path}", f"candidate/{path}", lineterm=""
        ):
            output.write(line + "\n")


def compare_tasks(
    baseline: TaskSnapshot | None,
    candidate: TaskSnapshot | None,
) -> list[Difference]:
    """Compare outcomes and every archived file without inferring release filtering.

    Rejected references can still carry converted bytes and a later static-filter
    decision. Neither static checks nor this comparison execute a grader.
    """
    if baseline is None or candidate is None:
        return [Difference("population", "missing_task", "", baseline is not None, candidate is not None)]
    if (baseline.source, baseline.path) != (candidate.source, candidate.path):
        raise ValueError("Compare tasks with the same original source/path")
    differences = []
    if baseline.outcome != candidate.outcome:
        differences.append(Difference("outcome", "conversion", "", baseline.outcome, candidate.outcome))
    if baseline.detail != candidate.detail:
        differences.append(Difference("outcome", "detail", "", baseline.detail, candidate.detail))
    for key in sorted(baseline.metadata.keys() | candidate.metadata.keys()):
        before, after = baseline.metadata.get(key), candidate.metadata.get(key)
        if before != after:
            differences.append(Difference("row_metadata", key, "", before, after))
    if baseline.legacy_static_rejection is not None and candidate.outcome == "converted":
        differences.append(
            Difference(
                "legacy_static_filter",
                baseline.legacy_static_rejection,
                "",
                baseline.legacy_static_detail,
                "accepted by candidate conversion and export",
            )
        )
    for path in sorted(baseline.files.keys() | candidate.files.keys()):
        before, after = baseline.files.get(path), candidate.files.get(path)
        category = file_category(path)
        if before is None or after is None:
            differences.append(Difference(category, "inventory", path, file_identity(before), file_identity(after)))
            continue
        for field in ("type", "mode", "link"):
            old, new = getattr(before, field), getattr(after, field)
            if old != new:
                differences.append(Difference(category, field, path, old, new))
        if before.sha256 != after.sha256 or before.size != after.size:
            differences.append(
                Difference(
                    category,
                    "bytes",
                    path,
                    {"sha256": before.sha256, "size": before.size},
                    {"sha256": after.sha256, "size": after.size},
                )
            )
        if before.parsed_toml != after.parsed_toml:
            differences.append(
                Difference(
                    category, "parsed_toml", path, toml_identity(before.parsed_toml), toml_identity(after.parsed_toml)
                )
            )
        if before.toml_error is not None or after.toml_error is not None:
            differences.append(Difference(category, "invalid_toml", path, before.toml_error, after.toml_error))
    return differences


class ParityReport:
    """Persist exact per-task results and deduplicate repeated file differences.

    Directory-level difference groups keep a bundled runtime from being repeated
    for every row. The database contains hashes and report details, never archives.
    """

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(path)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE difference_groups (id TEXT PRIMARY KEY, details BLOB NOT NULL);
            CREATE TABLE tasks (source TEXT NOT NULL, path TEXT NOT NULL,
                baseline_outcome TEXT, candidate_outcome TEXT, result TEXT NOT NULL,
                groups_json TEXT NOT NULL, legacy_static_rejection TEXT,
                PRIMARY KEY (source, path));
        """
        )
        self.counts: Counter[str] = Counter()
        self.categories: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.known_groups: dict[str, None] = {}

    def __enter__(self) -> "ParityReport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.connection.commit()
        self.connection.close()

    def add(
        self,
        baseline: TaskSnapshot | None,
        candidate: TaskSnapshot | None,
    ) -> None:
        task = baseline if baseline is not None else candidate
        assert task is not None
        differences = compare_tasks(baseline, candidate)
        result = "different" if differences else "equal"
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for difference in differences:
            key = (difference.category, difference.kind, difference.path.rpartition("/")[0])
            groups.setdefault(key, []).append(asdict(difference))
            self.categories[f"{difference.category}:{difference.kind}"] += 1
        identifiers = []
        for details in groups.values():
            encoded = json.dumps(details, sort_keys=True, default=json_temporal).encode()
            identity = hashlib.sha256(encoded).hexdigest()
            identifiers.append(identity)
            if identity not in self.known_groups:
                self.connection.execute(
                    "INSERT OR IGNORE INTO difference_groups VALUES (?, ?)", (identity, zlib.compress(encoded))
                )
                # This cache bounds memory; SQLite remains the authoritative deduplication index.
                if len(self.known_groups) >= 4096:
                    self.known_groups.clear()
                self.known_groups[identity] = None
        self.connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task.source,
                task.path,
                baseline.outcome if baseline else None,
                candidate.outcome if candidate else None,
                result,
                json.dumps(identifiers),
                baseline.legacy_static_rejection if baseline else None,
            ),
        )
        self.counts["tasks"] += 1
        self.counts[result] += 1
        if baseline:
            self.counts[f"baseline:{baseline.outcome}"] += 1
            if baseline.legacy_static_rejection:
                self.counts[f"legacy_static:{baseline.legacy_static_rejection}"] += 1
        if candidate:
            self.counts[f"candidate:{candidate.outcome}"] += 1
        if differences and len(self.examples) < 10:
            self.examples.append({"source": task.source, "path": task.path, "result": result, "groups": identifiers})
        if self.counts["tasks"] % 1000 == 0:
            self.connection.commit()

    def summary(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "differences": dict(self.categories), "examples": self.examples}
