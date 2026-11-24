# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gates on the sealed eval corpus manifests that the decontamination item table is built from.

A required suite must be complete, of the expected version and source pins, and hold every named benchmark
with its sealed record count; a best-effort suite lists the artifacts that loaded, and its required tasks must
all have loaded with records.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join


@dataclass(frozen=True)
class RequiredEvalSuite:
    """A sealed eval manifest whose benchmarks must all enter the item table.

    Args:
        manifest_path: Manifest that lists the suite's benchmark artifacts.
        names: Exact benchmark names the manifest must contain.
    """

    manifest_path: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class BestEffortEvalSuite:
    """A sealed eval manifest that lists the artifacts that loaded.

    The item table loads the manifest's required tasks through lm-eval; the
    artifacts it lists are validated against their hashes and must be below the
    manifest directory.

    Args:
        manifest_path: Manifest that lists the suite's artifacts.
        corpus_version: Version that the manifest must report.
        required_tasks: Tasks that must have loaded, each with its exact
            number of leaf tasks. The manifest's ``required_tasks`` must list
            that many leaf tasks for each one, and every leaf must have
            records.
    """

    manifest_path: str
    corpus_version: str
    required_tasks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredEvalManifest:
    """An independently versioned mandatory reference suite."""

    path: str
    version: str
    names: tuple[str, ...]
    source_catalog_sha256: str | None = None


def validate_required_manifest(
    manifest_path: str,
    expected_corpus_version: str,
    expected_names: tuple[str, ...],
    *,
    text_field: str = "text",
    expected_source_catalog_sha256: str | None = None,
) -> None:
    """Validate a sealed required suite: the manifest is complete, required and of
    the expected version and source pins, it names exactly *expected_names*, and every artifact matches its
    record count and checksum."""
    manifest_storage = StoragePath(manifest_path)
    if not manifest_storage.exists():
        raise ValueError(f"required eval manifest does not exist: {manifest_path}")
    with manifest_storage.open("r") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"required eval manifest must be an object: {manifest_path}")
    if manifest.get("corpus_version") != expected_corpus_version:
        raise ValueError(
            f"required eval manifest version is {manifest.get('corpus_version')!r}, expected {expected_corpus_version!r}"
        )
    if manifest.get("required") is not True:
        raise ValueError("required eval manifest does not mark the suite as required")
    if manifest.get("status") != "complete":
        raise ValueError(f"required eval manifest status is {manifest.get('status')!r}, expected 'complete'")

    if expected_source_catalog_sha256 is not None:
        if manifest.get("source_catalog_sha256") != expected_source_catalog_sha256:
            raise ValueError("required eval manifest source catalog differs from the configured source pins")

    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or not all(isinstance(entry, Mapping) for entry in benchmarks):
        raise ValueError("required eval manifest benchmarks must be a list of objects")
    if not all(isinstance(entry.get("name"), str) for entry in benchmarks):
        raise ValueError("required eval manifest benchmark entries need string names")
    actual_names = tuple(entry["name"] for entry in benchmarks)
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        raise ValueError(
            f"required eval manifest benchmarks are {sorted(str(name) for name in actual_names)}, "
            f"expected {sorted(expected_names)}"
        )

    manifest_parent = os.path.dirname(manifest_path)
    for entry in benchmarks:
        name = entry.get("name")
        artifact = entry.get("artifact")
        expected_records = entry.get("expected_records")
        if not isinstance(name, str) or not isinstance(artifact, str) or not isinstance(expected_records, int):
            raise ValueError("required eval manifest benchmark entries need name, artifact, and expected_records")
        if expected_records <= 0:
            raise ValueError(f"{name}: configured reference family yielded zero records")
        artifact_path = prefix_join(manifest_parent, artifact)
        artifact_storage = StoragePath(artifact_path)
        if not artifact_storage.exists():
            raise ValueError(f"{name}: required eval artifact does not exist: {artifact_path}")
        with artifact_storage.open("rb") as source:
            expected_sha256 = entry.get("sha256")
            if expected_sha256 is not None:
                actual_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError(f"{name}: artifact SHA-256 differs from its sealed manifest")
                source.seek(0)
            parquet = pq.ParquetFile(source)
            actual_records = parquet.metadata.num_rows
            if text_field not in parquet.schema_arrow.names:
                raise ValueError(f"{name}: required eval artifact does not contain text field {text_field!r}")
        if actual_records != expected_records:
            raise ValueError(f"{name}: manifest expects {expected_records} records, artifact contains {actual_records}")


def validate_best_effort_manifest(
    manifest_path: str,
    expected_corpus_version: str,
    required_tasks: Mapping[str, int],
) -> None:
    """Validate a best-effort manifest and the exact artifacts it lists."""
    manifest_storage = StoragePath(manifest_path)
    if not manifest_storage.exists():
        raise ValueError(f"best-effort eval manifest does not exist: {manifest_path}")
    with manifest_storage.open("r") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"best-effort eval manifest must be an object: {manifest_path}")
    if manifest.get("corpus_version") != expected_corpus_version:
        raise ValueError(
            f"best-effort eval manifest version is {manifest.get('corpus_version')!r}, "
            f"expected {expected_corpus_version!r}"
        )
    if manifest.get("required") is not False:
        raise ValueError("best-effort eval manifest must mark the suite as not required")
    if manifest.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError(
            f"best-effort eval manifest status is {manifest.get('status')!r}, "
            "expected 'complete' or 'complete_with_failures'"
        )

    included_tasks = manifest.get("included_leaf_tasks")
    artifacts = manifest.get("artifacts")
    if not isinstance(included_tasks, list) or not all(isinstance(task, str) for task in included_tasks):
        raise ValueError("best-effort eval manifest included_leaf_tasks must be a list of strings")
    if not isinstance(artifacts, list) or not all(isinstance(entry, Mapping) for entry in artifacts):
        raise ValueError("best-effort eval manifest artifacts must be a list of objects")
    manifest_required_tasks = manifest.get("required_tasks", {})
    if not isinstance(manifest_required_tasks, Mapping):
        raise ValueError("best-effort eval manifest required_tasks must be an object")

    manifest_parent = os.path.dirname(manifest_path)
    artifact_records: dict[str, int] = {}
    for entry in artifacts:
        task = entry.get("task")
        artifact = entry.get("artifact")
        expected_records = entry.get("records")
        if not isinstance(task, str) or not isinstance(artifact, str) or not isinstance(expected_records, int):
            raise ValueError("best-effort eval artifacts need task, artifact, and records")
        if artifact.startswith("/") or ".." in artifact.split("/"):
            raise ValueError(f"{task}: best-effort artifact must be relative to its root: {artifact}")
        artifact_path = prefix_join(manifest_parent, artifact)
        artifact_storage = StoragePath(artifact_path)
        if not artifact_storage.exists():
            raise ValueError(f"{task}: best-effort eval artifact does not exist: {artifact_path}")
        with artifact_storage.open("rb") as source:
            actual_records = pq.ParquetFile(source).metadata.num_rows
        if actual_records != expected_records:
            raise ValueError(
                f"{task}: best-effort manifest expects {expected_records} records, artifact contains {actual_records}"
            )
        artifact_records[task] = actual_records

    if sorted(artifact_records) != sorted(included_tasks) or len(artifacts) != len(included_tasks):
        raise ValueError("best-effort eval artifact tasks do not match included_leaf_tasks")
    failures = required_task_failures(required_tasks, manifest_required_tasks, artifact_records)
    if failures:
        raise ValueError(f"best-effort eval manifest has incomplete required tasks: {failures}")


def required_task_failures(
    required: Mapping[str, int], task_leaves: Mapping[str, list[str]], records: Mapping[str, int]
) -> list[str]:
    """Return one message for each required task that is not complete.

    Args:
        required: Each required task with its exact number of leaf tasks.
        task_leaves: Each task that loaded with its leaf tasks.
        records: Each leaf task that has an artifact with its record count.
    """
    failures: list[str] = []
    for task, expected_leaves in required.items():
        leaves = task_leaves.get(task)
        if not leaves:
            failures.append(f"{task}: did not load")
            continue
        if len(leaves) != expected_leaves:
            failures.append(f"{task}: loaded {len(leaves)} leaf tasks, expected {expected_leaves}")
        empty = sorted(leaf for leaf in leaves if not records.get(leaf))
        if empty:
            failures.append(f"{task}: no records for leaf tasks {empty}")
    return failures


@dataclass(frozen=True)
class CorpusGate:
    """The sealed suites that must be complete before the item table is built from the eval corpus."""

    corpus_version: str
    required_suites: tuple[RequiredEvalSuite, ...]
    required_manifests: tuple[RequiredEvalManifest, ...]
    best_effort_suite: BestEffortEvalSuite

    def check(self) -> None:
        """Raise unless every required suite and the best-effort suite's required tasks are complete."""
        for suite in self.required_suites:
            validate_required_manifest(suite.manifest_path, self.corpus_version, suite.names)
        for manifest in self.required_manifests:
            validate_required_manifest(
                manifest.path,
                manifest.version,
                manifest.names,
                expected_source_catalog_sha256=manifest.source_catalog_sha256,
            )
        validate_best_effort_manifest(
            self.best_effort_suite.manifest_path,
            self.best_effort_suite.corpus_version,
            self.best_effort_suite.required_tasks,
        )
