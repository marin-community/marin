# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lane's verified-task universe: join difficulty-ledger tasks to census TaskSpecs.

The Atlas difficulty evidence ledger keys tasks by ``(source_id, task_sha256)`` in the
freeze-time digest space, while the durable task material lives in the retained-census
releases' normalized parquets (``id, path, task_spec, task_sha256, runtime``) in the
current census digest space. The two digest spaces have drifted (serializer vintages), so
this module joins on ``task_id`` — stable across both — and records both digests in the
manifest for provenance. TaskSpec rows carry fields the lane's pinned taskcompendium does
not know (delivery-integrity metadata); unknown fields are stripped before validation and
recorded per row, since grading semantics live in the grader, not those fields.

The builder streams column-pruned parquet reads, so it runs in-region (an Iris CPU job)
against the census releases without staging populations locally.
"""

import json
import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import fsspec
import pyarrow.parquet as pq
from rigging.filesystem.factory import open_url
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    TaskDifficulty,
    url_protocol,
)

logger = logging.getLogger(__name__)

CENSUS_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/retained-census"
LEDGER_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger"
MANIFEST_FILENAME = "tasks.jsonl"
TASKS_DIRECTORY = "tasks"

GRADER_KINDS = frozenset({"verifyit", "script"})
"""The grader kinds the lane verifies with; ``none`` is the no-grader skip."""


@dataclass(frozen=True, slots=True)
class CensusEntry:
    """One source's normalized task material inside a census release."""

    source_id: str
    release: str
    normalized_path: str
    dataset_id: str | None


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One verified task offered to the lane, with its provenance across digest spaces."""

    task_id: str
    source_id: str
    task_sha256: str
    """The difficulty-ledger digest — the join key the plan bands on."""
    domain: str
    task_path: str
    census_task_sha256: str
    """The census release's digest for the same task bytes (its own digest space)."""
    dataset_id: str | None
    census_release: str
    stripped_fields: tuple[str, ...]

    def document(self) -> dict:
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "task_sha256": self.task_sha256,
            "domain": self.domain,
            "task_path": self.task_path,
            "census_task_sha256": self.census_task_sha256,
            "dataset_id": self.dataset_id,
            "census_release": self.census_release,
            "stripped_fields": list(self.stripped_fields),
        }


def domain_for(source_id: str) -> str:
    """The coarse domain tag: the source's own name (the part after its namespace)."""
    return source_id.split(":", 1)[-1].lower() or source_id.lower()


def read_census_entries(census_root: str = CENSUS_ROOT) -> list[CensusEntry]:
    """Enumerate every census release's per-source normalized material."""
    filesystem = fsspec.filesystem(url_protocol(census_root))
    entries: dict[tuple[str, str], CensusEntry] = {}
    for release_path in sorted(filesystem.ls(census_root)):
        release = release_path.rstrip("/").rsplit("/", 1)[-1]
        index_path = f"{release_path.rstrip('/')}/censuses.json"
        try:
            with open_url(index_path, "rt") as source:
                document = json.load(source)
        except FileNotFoundError:
            continue
        rows = document if isinstance(document, list) else document.get("censuses", [])
        for row in rows:
            source_id = row.get("source_id")
            normalized = row.get("normalized_path")
            if not source_id or not normalized:
                continue
            population = row.get("population") or {}
            entry = CensusEntry(
                source_id=source_id,
                release=release,
                normalized_path=normalized,
                dataset_id=population.get("dataset_id"),
            )
            # Deterministic: the first release (sorted) that covers a source wins, so a
            # task covered by several releases materializes exactly once.
            entries.setdefault(source_id, entry)
    return sorted(entries.values(), key=lambda entry: (entry.source_id, entry.release))


def validate_task_spec(spec_json: str) -> tuple[TaskSpec, tuple[str, ...]]:
    """Parse a census TaskSpec row under the pinned model, stripping fields it does not know.

    Unknown fields are delivery-integrity metadata that joined the serializer after this
    lane's pinned taskcompendium; grading semantics live in the grader. Stripped names are
    returned so the manifest can record exactly what was dropped.
    """
    document = json.loads(spec_json)
    stripped = sorted(set(document) - set(TaskSpec.model_fields))
    for name in stripped:
        del document[name]
    return TaskSpec.model_validate(document), tuple(stripped)


def build_tasks_manifest(
    ledger_difficulty: dict[tuple[str, str], TaskDifficulty],
    entries: list[CensusEntry],
    output_prefix: str,
    *,
    min_attempts: int = 1,
) -> dict:
    """Materialize census TaskSpecs for ledger-covered tasks and write the manifest bundle.

    Writes ``<output_prefix>/tasks/<task_id>.json`` per task plus ``<output_prefix>/tasks.jsonl``
    (one :class:`ManifestRow` document per line) and returns the build receipt. Tasks without
    a usable grader, without a user turn, or whose census release lacks their task_id are
    skipped and counted in the receipt.
    """
    wanted_by_source: dict[str, dict[str, str]] = {}
    for (source_id, task_sha256), difficulty in ledger_difficulty.items():
        if difficulty.scored < min_attempts:
            continue
        wanted_by_source.setdefault(source_id, {})[difficulty.task_id] = task_sha256

    filesystem = fsspec.filesystem(url_protocol(output_prefix))
    filesystem.makedirs(output_prefix, exist_ok=True)
    tasks_prefix = f"{output_prefix.rstrip('/')}/{TASKS_DIRECTORY}"

    rows: list[ManifestRow] = []
    skipped = {"no_ledger_task": 0, "no_grader": 0, "no_user_turn": 0, "spec_invalid": 0}
    for entry in entries:
        wanted = wanted_by_source.get(entry.source_id)
        if not wanted:
            continue
        matched = _stream_matched_rows(entry, set(wanted))
        for row in matched:
            task_id = row["id"]
            ledger_sha = wanted.get(task_id)
            if ledger_sha is None:
                skipped["no_ledger_task"] += 1
                continue
            try:
                task, stripped = validate_task_spec(row["task_spec"])
            except Exception as error:
                logger.warning("SKIP %s: census spec does not validate: %s", task_id, error)
                skipped["spec_invalid"] += 1
                continue
            if task.grader is None or task.grader.kind not in GRADER_KINDS:
                skipped["no_grader"] += 1
                continue
            if not any(getattr(event, "role", None) == "user" for event in task.context.events):
                skipped["no_user_turn"] += 1
                continue
            spec_path = f"{tasks_prefix}/{task_id}.json"
            payload = task.model_dump_json(indent=2, by_alias=True) + "\n"
            with open_url(spec_path, "wt") as sink:
                sink.write(payload)
            rows.append(
                ManifestRow(
                    task_id=task_id,
                    source_id=entry.source_id,
                    task_sha256=ledger_sha,
                    domain=domain_for(entry.source_id),
                    task_path=spec_path,
                    census_task_sha256=row["task_sha256"],
                    dataset_id=entry.dataset_id,
                    census_release=entry.release,
                    stripped_fields=stripped,
                )
            )

    manifest_path = f"{output_prefix.rstrip('/')}/{MANIFEST_FILENAME}"
    payload = "\n".join(json.dumps(row.document(), sort_keys=True) for row in rows) + ("\n" if rows else "")
    with open_url(manifest_path, "wt") as sink:
        sink.write(payload)
    receipt = {
        "rows": len(rows),
        "skipped": skipped,
        "sources": sorted({row.source_id for row in rows}),
        "manifest": manifest_path,
        "digest": sha256(payload.encode()).hexdigest(),
    }
    logger.info("TASKS manifest %s rows=%d skipped=%s", manifest_path, len(rows), skipped)
    return receipt


def _stream_matched_rows(entry: CensusEntry, wanted_ids: set[str]) -> list[dict[str, Any]]:
    """Column-pruned two-pass read of one census parquet: ids first, then matched rows' bytes."""
    if not wanted_ids:
        return []
    with open_url(entry.normalized_path, "rb") as source:
        handle = pq.ParquetFile(source)
        matched_ids: set[str] = set()
        for batch in handle.iter_batches(batch_size=2048, columns=["id"]):
            for value in batch.column("id").to_pylist():
                if value in wanted_ids:
                    matched_ids.add(value)
        if not matched_ids:
            return []
    rows: list[dict[str, Any]] = []
    with open_url(entry.normalized_path, "rb") as source:
        handle = pq.ParquetFile(source)
        for batch in handle.iter_batches(batch_size=512, columns=["id", "task_sha256", "task_spec"]):
            columns = batch.to_pydict()
            for row_id, task_sha, spec_json in zip(
                columns["id"], columns["task_sha256"], columns["task_spec"], strict=True
            ):
                if row_id in matched_ids:
                    rows.append({"id": row_id, "task_sha256": task_sha, "task_spec": spec_json})
    return rows


def manifest_rows(path: str) -> list[ManifestRow]:
    """Read back a manifest written by :func:`build_tasks_manifest`."""
    rows: list[ManifestRow] = []
    with open_url(path, "rt") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                ManifestRow(
                    task_id=row["task_id"],
                    source_id=row["source_id"],
                    task_sha256=row["task_sha256"],
                    domain=row["domain"],
                    task_path=row["task_path"],
                    census_task_sha256=row["census_task_sha256"],
                    dataset_id=row.get("dataset_id"),
                    census_release=row["census_release"],
                    stripped_fields=tuple(row.get("stripped_fields", ())),
                )
            )
    return rows
