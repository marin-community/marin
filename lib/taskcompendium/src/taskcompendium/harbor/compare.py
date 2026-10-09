# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Harbor archive content and audit counts against a pinned release manifest."""

import hashlib
import io
import json
import sqlite3
import tarfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import click
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig

from taskcompendium.harbor.export import TASKS_SCHEMA
from taskcompendium.harbor.snapshots import TaskSnapshot, json_temporal


class Tolerance(StrEnum):
    TOML_FORMATTING = "toml_formatting"


@dataclass(frozen=True)
class Difference:
    category: str
    kind: str
    path: str
    baseline: Any
    candidate: Any
    tolerance: Tolerance | None = None


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


def compare_tasks(
    baseline: TaskSnapshot | None,
    candidate: TaskSnapshot | None,
    *,
    tolerances: frozenset[Tolerance] = frozenset(),
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
            differences.append(
                Difference(
                    category, "inventory", path, asdict(before) if before else None, asdict(after) if after else None
                )
            )
            continue
        for field in ("type", "mode", "link"):
            old, new = getattr(before, field), getattr(after, field)
            if old != new:
                differences.append(Difference(category, field, path, old, new))
        if before.sha256 != after.sha256 or before.size != after.size:
            tolerance = None
            if (
                Tolerance.TOML_FORMATTING in tolerances
                and before.parsed_toml is not None
                and before.parsed_toml == after.parsed_toml
            ):
                tolerance = Tolerance.TOML_FORMATTING
            differences.append(
                Difference(
                    category,
                    "bytes",
                    path,
                    {"sha256": before.sha256, "size": before.size},
                    {"sha256": after.sha256, "size": after.size},
                    tolerance,
                )
            )
        if before.parsed_toml != after.parsed_toml:
            differences.append(Difference(category, "parsed_toml", path, before.parsed_toml, after.parsed_toml))
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
        *,
        tolerances: frozenset[Tolerance] = frozenset(),
    ) -> None:
        task = baseline if baseline is not None else candidate
        assert task is not None
        differences = compare_tasks(baseline, candidate, tolerances=tolerances)
        mismatch = any(difference.tolerance is None for difference in differences)
        result = "different" if mismatch else "tolerated" if differences else "equal"
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for difference in differences:
            key = (difference.category, difference.kind, difference.path.rpartition("/")[0])
            groups.setdefault(key, []).append(asdict(difference))
            prefix = "tolerated" if difference.tolerance else "different"
            self.categories[f"{prefix}:{difference.category}:{difference.kind}"] += 1
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


def compare_harbor(
    tasks: Path, golden_manifest: Path, *, sources: tuple[str, ...], golden_revision: str
) -> dict[str, Any]:
    """Check the wire schema, every native Harbor config, identities, and per-source counts."""
    golden_bytes = golden_manifest.read_bytes()
    golden = json.loads(golden_bytes)
    unknown = set(sources) - golden["by_source"].keys()
    if unknown:
        raise ValueError(f"Sources absent from the golden release: {sorted(unknown)}")
    parquet = pq.ParquetFile(tasks)
    if not parquet.schema_arrow.equals(TASKS_SCHEMA, check_metadata=False):
        raise ValueError(f"Incompatible task parquet schema: {parquet.schema_arrow}")
    counts: Counter[str] = Counter(dict.fromkeys(sources, 0))
    identities: set[tuple[str, str]] = set()
    task_ids: set[str] = set()
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            identity = (row["source"], row["path"])
            if row["source"] not in counts:
                raise ValueError(f"Unexpected generated source: {row['source']}")
            if identity in identities:
                raise ValueError(f"Duplicate source/path: {identity}")
            identities.add(identity)
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
                members = {member.name: member for member in archive}
                required = {"instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"}
                if not required <= members.keys():
                    raise ValueError(f"Missing Harbor files for {identity}: {required - members.keys()}")
                if any(name.startswith("solution/") for name in members):
                    raise ValueError(f"Oracle solution appears in task archive: {identity}")
                handle = archive.extractfile(members["task.toml"])
                assert handle is not None
                config = TaskConfig.model_validate_toml(handle.read().decode())
            if config.metadata["tasktrove_path"] != row["path"] or config.metadata["tasktrove_source"] != row["source"]:
                raise ValueError(f"Source identity changed in archive: {identity}")
            task_id = config.metadata["taskcompendium_id"]
            if task_id in task_ids:
                raise ValueError(f"Duplicate TaskSpec id: {task_id}")
            task_ids.add(task_id)
            counts[row["source"]] += 1
    comparisons = []
    for source, count in sorted(counts.items()):
        released = golden["by_source"][source]
        golden_count = released.get("converted", 0)
        comparisons.append(
            {
                "source": source,
                "generated": count,
                "golden": golden_count,
                "delta": count - golden_count,
                "golden_statuses": released,
            }
        )
    return {
        "schema_compatible": True,
        "harbor_configs_valid": True,
        "row_identities_unique": True,
        "oracle_solutions_separate": True,
        "rows": sum(counts.values()),
        "sources": comparisons,
        "golden_counts_match": bool(comparisons) and all(row["delta"] == 0 for row in comparisons),
        "golden_revision": golden_revision,
        "golden_manifest_sha256": hashlib.sha256(golden_bytes).hexdigest(),
        "runtime_verified": False,
        "comparison_scope": (
            "Release-manifest counts and generated archive structure; golden task binaries were not compared."
        ),
    }


@click.command(help=__doc__)
@click.option("--tasks", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--golden-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--source", "sources", multiple=True, required=True, help="Expected source config; include sources with no output."
)
@click.option("--golden-revision", required=True, help="Pinned release revision from which the manifest was downloaded.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(tasks: Path, golden_manifest: Path, sources: tuple[str, ...], golden_revision: str, output: Path) -> None:
    report = compare_harbor(tasks, golden_manifest, sources=sources, golden_revision=golden_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    click.echo(json.dumps(report))


if __name__ == "__main__":
    main()
