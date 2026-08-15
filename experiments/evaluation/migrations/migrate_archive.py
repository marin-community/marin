# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill legacy per-(sub)task sample parquets into finestore archives.

Legacy runs stored one ``samples_<task>_<ts>.parquet`` per (sub)task (each row an ``EvalSample``) and
referenced Harbor trajectories by an object-storage URI. This tool reads those files, writes the
same rows into the run's finestore ``samples`` table, and for agentic samples pulls each referenced
trajectory into the ``blobs`` table (rewriting the sample's ``trajectory_uri`` to a ``finestore://``
reference) and flattens its steps into the ``steps`` table. If a legacy trajectory is already
missing, its sample and original URI are preserved.

It is idempotent — samples dedupe on ``(task, doc_id, trial_id, filter)`` — and validates every normalized
sample, every available raw trajectory blob, and each flattened trajectory step before optionally
moving the superseded ``samples_*.parquet`` files to region-local 30-day storage. A later migration
can read those archived Parquets while writing replacement shards at the original results path.
Evalchemy's native JSONL and Harbor's job tree remain in place because current writers still use
them as mechanism-native artifacts and resume state.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable
from enum import StrEnum
from urllib.parse import urlsplit

import click
import pyarrow.parquet as pq
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    SAMPLES_PREFIX,
    SAMPLES_SUFFIX,
    SCHEMA_VERSION,
    TRIAL_ID_COLUMN,
    EvalSample,
    EvaluationStore,
    SampleKind,
    StepRecord,
    sample_from_archive_row,
    trajectory_step_rows,
)
from finestore.reader import CompositeReader
from marin.evaluation.records import list_records
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.migrations.archive_backup import legacy_archive_prefix
from experiments.evaluation.migrations.samples_v4 import preserve_and_replace_samples

logger = logging.getLogger(__name__)

_ARCHIVE_URI_PREFIX = "finestore://"
_MIGRATION_WRITER_ID = "migrate"
_OBJECT_CHECKSUM_KEYS = ("md5Hash", "md5", "ETag", "etag", "checksum")

SampleKey = tuple[str, str, str, str]
StepKey = tuple[str, str, str, int | None]


class _MigrationMode(StrEnum):
    KEEP_LEGACY = "keep_legacy"
    ARCHIVE_LEGACY = "archive_legacy"
    FROM_LEGACY_ARCHIVE = "from_legacy_archive"


@dataclasses.dataclass(frozen=True)
class MigrationCounts:
    """Legacy inputs processed, including trajectories already absent from object storage."""

    samples: int
    steps: int
    trajectories: int
    missing_trajectories: int = 0


@dataclasses.dataclass(frozen=True)
class LegacySampleFile:
    """One superseded normalized sample Parquet under a run's results path."""

    source: str
    relative_path: str
    size: int


@dataclasses.dataclass(frozen=True)
class ValidationCounts:
    """Observed legacy inputs and the archive rows validated against them."""

    legacy_files: int
    legacy_rows: int
    legacy_samples: int
    archive_samples: int
    trajectories: int
    steps: int
    missing_trajectories: int


@dataclasses.dataclass(frozen=True)
class MigrationResult:
    """One completed migration, validation, and optional legacy-file archival."""

    results_path: str
    counts: MigrationCounts
    validation: ValidationCounts
    archive_prefix: str | None
    archived_files: int


@dataclasses.dataclass(frozen=True)
class MigrationPlan:
    """Read-only plan for one results path."""

    results_path: str
    legacy_source: str
    legacy_files: int
    legacy_bytes: int
    archive_prefix: str | None


@dataclasses.dataclass(frozen=True)
class _ObjectSignature:
    size: int
    checksum_kind: str | None
    checksum: str | None


class MigrationValidationError(ValueError):
    """The finestore archive does not faithfully contain its legacy inputs."""


def _trial_id_from_uri(uri: str) -> str:
    """Recover a Harbor trial id from either legacy trajectory layout."""
    segments = urlsplit(uri).path.split("/")
    if len(segments) >= 3 and segments[-2:] == ["agent", "trajectory.json"]:
        return segments[-3]
    if len(segments) >= 2 and segments[-2] == "trajectories" and segments[-1].endswith(".json"):
        return segments[-1][: -len(".json")]
    raise ValueError(f"cannot recover Harbor trial id from trajectory URI {uri!r}")


def _sample_key(sample: EvalSample, trial_id: str) -> SampleKey:
    """The archive's identity for one sample: task, document, trial, and extraction filter.

    A legacy Parquet holds one row per (document, filter) for a multi-filter task, so the filter must
    take part or two distinct rows would compare as one and validation would accept a lossy archive.
    """
    return sample.task, sample.doc_id, trial_id, (sample.grading.filter or "") if sample.grading else ""


def _trial_id(sample: EvalSample) -> str:
    uri = sample.trajectory_uri
    if sample.kind != SampleKind.AGENTIC or not uri or uri.startswith(_ARCHIVE_URI_PREFIX):
        return ""
    return _trial_id_from_uri(uri)


def legacy_sample_files(legacy_source: str) -> tuple[LegacySampleFile, ...]:
    """List the legacy normalized sample Parquets beneath one live or archived source."""
    root = StoragePath(legacy_source)
    if not root.exists():
        raise FileNotFoundError(legacy_source)
    files = []
    for directory, _, names in root.walk():
        for name in names:
            if not (name.startswith(SAMPLES_PREFIX) and name.endswith(SAMPLES_SUFFIX)):
                continue
            source = directory / name
            relative_root = (
                dataclasses.replace(root, scheme=source.scheme, netloc=source.netloc, rooted=source.rooted)
                if root.is_local and source.is_local
                else root
            )
            files.append(
                LegacySampleFile(
                    source=str(source),
                    relative_path=source.relative_to(relative_root),
                    size=source.size(),
                )
            )
    return tuple(sorted(files, key=lambda file: file.source))


def _legacy_sample_rows(files: Iterable[LegacySampleFile]) -> Iterable[EvalSample]:
    for legacy_file in files:
        with StoragePath(legacy_file.source).open("rb") as handle:
            table = pq.read_table(handle)
        for row in table.to_pylist():
            yield sample_from_archive_row(row)


def _legacy_sample_inventory(files: Iterable[LegacySampleFile]) -> tuple[int, dict[SampleKey, EvalSample]]:
    rows = 0
    samples: dict[SampleKey, EvalSample] = {}
    for sample in _legacy_sample_rows(files):
        samples[_sample_key(sample, _trial_id(sample))] = sample
        rows += 1
    return rows, samples


def _replace_stale_samples(results_path: str) -> None:
    """Clear a samples table written under an older contract, preserving it outside the run first.

    The legacy parquets this migration reads reproduce every row of the table, agentic ones included,
    so it may replace a table outright where an lm-eval export refuses. Rows written under a narrower
    primary key would otherwise survive beside the new ones as duplicates.
    """
    stored_version = CompositeReader(results_path).schema_version(ARCHIVE_SAMPLES_TABLE)
    if stored_version is not None and stored_version != SCHEMA_VERSION:
        preserve_and_replace_samples(results_path, stored_version)


def migrate_run(
    results_path: str,
    *,
    files: tuple[LegacySampleFile, ...] | None = None,
    writer_id: str = _MIGRATION_WRITER_ID,
) -> MigrationCounts:
    """Backfill one run's legacy sample parquets into its finestore archive. Safe to re-run."""
    sample_count = step_count = trajectory_count = missing_trajectory_count = 0
    _replace_stale_samples(results_path)
    # Archive shards are named ``{seq}-{uid}.parquet`` under per-table subdirs, so the legacy
    # ``samples_*.parquet`` filter below never picks them up: a re-run reads only legacy sources.
    with EvaluationStore.open(results_path, writer_id=writer_id) as store:
        for sample in _legacy_sample_rows(legacy_sample_files(results_path) if files is None else files):
            trial_id = _trial_id(sample)
            uri = sample.trajectory_uri
            if sample.kind == SampleKind.AGENTIC and uri and not uri.startswith(_ARCHIVE_URI_PREFIX):
                try:
                    raw = StoragePath(uri).read_bytes()
                except FileNotFoundError:
                    # Some old runs already lost part of their Harbor job tree. Preserve the
                    # normalized sample and its original URI so the archive remains a faithful
                    # representation of the surviving legacy data.
                    logger.warning("legacy trajectory is already missing: %s", uri)
                    missing_trajectory_count += 1
                else:
                    stored = store.add_trajectory(
                        raw,
                        task=sample.task,
                        doc_id=sample.doc_id,
                        trial_id=trial_id,
                    )
                    sample = sample.model_copy(update={"trajectory_uri": stored.uri})
                    step_count += len(stored.steps)
                    trajectory_count += 1
            store.add_sample(sample, trial_id=trial_id)
            sample_count += 1
        store.seal()
    counts = MigrationCounts(
        samples=sample_count,
        steps=step_count,
        trajectories=trajectory_count,
        missing_trajectories=missing_trajectory_count,
    )
    logger.info("migrated %s: %s", results_path, counts)
    return counts


def _archive_samples(reader: CompositeReader) -> dict[SampleKey, EvalSample]:
    table = reader.scan(ARCHIVE_SAMPLES_TABLE)
    if table is None:
        return {}
    samples = {}
    for row in table.to_pylist(maps_as_pydicts="strict"):
        sample = sample_from_archive_row(row)
        samples[_sample_key(sample, row.get(TRIAL_ID_COLUMN) or "")] = sample
    return samples


def _archive_steps(reader: CompositeReader) -> dict[StepKey, StepRecord]:
    table = reader.scan(ARCHIVE_STEPS_TABLE)
    if table is None:
        return {}
    field_names = tuple(field.name for field in dataclasses.fields(StepRecord))
    steps = {}
    for row in table.to_pylist():
        step = StepRecord(**{name: row.get(name) for name in field_names})
        steps[(step.task, step.doc_id, step.trial_id, step.step_id)] = step
    return steps


def _expected_steps(raw: bytes, sample: EvalSample, trial_id: str) -> list[StepRecord]:
    trajectory = json.loads(raw)
    return trajectory_step_rows(trajectory, task=sample.task, doc_id=sample.doc_id, trial_id=trial_id)


def validate_run(
    results_path: str,
    *,
    files: tuple[LegacySampleFile, ...] | None = None,
) -> ValidationCounts:
    """Validate every surviving legacy sample and trajectory against the sealed archive.

    Extra archive rows are allowed so this remains safe to resume after a process copied all legacy
    files and then stopped partway through deleting their sources.
    """
    files = legacy_sample_files(results_path) if files is None else files
    legacy_rows, legacy_samples = _legacy_sample_inventory(files)
    reader = CompositeReader(results_path)
    if not reader.is_sealed():
        raise MigrationValidationError(f"archive at {results_path!r} is not sealed")
    archive_samples = _archive_samples(reader)
    archive_steps = _archive_steps(reader)
    trajectory_count = step_count = missing_trajectory_count = 0

    for key, expected in legacy_samples.items():
        actual = archive_samples.get(key)
        if actual is None:
            raise MigrationValidationError(f"archive at {results_path!r} is missing sample key {key!r}")
        uri = expected.trajectory_uri
        raw = None
        if expected.kind == SampleKind.AGENTIC and uri and not uri.startswith(_ARCHIVE_URI_PREFIX):
            if actual.trajectory_uri and actual.trajectory_uri.startswith(_ARCHIVE_URI_PREFIX):
                raw = StoragePath(uri).read_bytes()
                expected_for_comparison = expected.model_copy(update={"trajectory_uri": actual.trajectory_uri})
            elif actual.trajectory_uri == uri:
                try:
                    StoragePath(uri).read_bytes()
                except FileNotFoundError:
                    missing_trajectory_count += 1
                    expected_for_comparison = expected
                else:
                    raise MigrationValidationError(
                        f"archive sample {key!r} still references an available legacy trajectory"
                    )
            else:
                raise MigrationValidationError(f"archive sample {key!r} has invalid trajectory URI")
        else:
            expected_for_comparison = expected
        if actual != expected_for_comparison:
            raise MigrationValidationError(f"archive sample {key!r} does not match its legacy row")

        if raw is None:
            continue
        archived_raw = reader.resolve(expected_for_comparison.trajectory_uri)
        if archived_raw != raw:
            raise MigrationValidationError(f"archive trajectory for sample {key!r} does not match {uri!r}")
        trajectory_count += 1
        expected_steps = _expected_steps(raw, expected, key[2])
        for step in expected_steps:
            step_key = (step.task, step.doc_id, step.trial_id, step.step_id)
            if archive_steps.get(step_key) != step:
                raise MigrationValidationError(f"archive trajectory step {step_key!r} does not match {uri!r}")
        step_count += len(expected_steps)

    counts = ValidationCounts(
        legacy_files=len(files),
        legacy_rows=legacy_rows,
        legacy_samples=len(legacy_samples),
        archive_samples=len(archive_samples),
        trajectories=trajectory_count,
        steps=step_count,
        missing_trajectories=missing_trajectory_count,
    )
    logger.info("validated %s: %s", results_path, counts)
    return counts


def _object_signature(info: dict) -> _ObjectSignature:
    size = int(info.get("size", 0))
    for key in _OBJECT_CHECKSUM_KEYS:
        value = info.get(key)
        if value is not None:
            return _ObjectSignature(size=size, checksum_kind=key.lower(), checksum=str(value).strip('"'))
    return _ObjectSignature(size=size, checksum_kind=None, checksum=None)


def archive_legacy_sample_files(
    results_path: str,
    archive_prefix: str,
    *,
    files: tuple[LegacySampleFile, ...] | None = None,
) -> int:
    """Move validated legacy sample Parquets under ``archive_prefix``.

    Every object is copied and its size/provider checksum verified before any source is deleted. A
    failure during the copy phase therefore leaves the complete legacy layout in place. A failure
    during the delete phase is safe to resume because validation permits archive rows whose legacy
    source was already retired.

    Returns:
        The number of source files copied, verified, and deleted.
    """
    files = legacy_sample_files(results_path) if files is None else files
    if not files:
        return 0
    results = StoragePath(results_path)
    destination_root = StoragePath(archive_prefix)
    try:
        destination_root.relative_to(results)
    except ValueError:
        pass
    else:
        raise ValueError(f"legacy archive prefix {archive_prefix!r} must not be inside {results_path!r}")

    destinations = []
    for legacy_file in files:
        source = StoragePath(legacy_file.source)
        destination = destination_root / legacy_file.relative_path
        same_backend = (source.is_local and destination.is_local) or source.scheme == destination.scheme
        if not same_backend:
            raise ValueError(f"cannot move legacy sample across storage backends: {source} -> {destination}")
        source_fs, source_path = url_to_fs(str(source))
        destination_fs, destination_path = url_to_fs(str(destination))
        destination.parent.mkdirs()
        source_fs.cp_file(source_path, destination_path)
        source_signature = _object_signature(source_fs.info(source_path))
        destination_signature = _object_signature(destination_fs.info(destination_path))
        if source_signature.size != destination_signature.size:
            raise OSError(f"legacy sample copy has the wrong size: {source} -> {destination}")
        if (
            source_signature.checksum_kind is not None
            and source_signature.checksum_kind == destination_signature.checksum_kind
            and source_signature.checksum != destination_signature.checksum
        ):
            raise OSError(f"legacy sample copy has the wrong checksum: {source} -> {destination}")
        destinations.append((source, destination))

    for source, destination in destinations:
        source.rm()
        logger.info("archived legacy sample %s", destination)
    return len(destinations)


def migration_plan(results_path: str, *, legacy_source: str | None = None) -> MigrationPlan:
    """Return the read-only migration and archival plan for one run."""
    legacy_source = legacy_source or results_path
    files = legacy_sample_files(legacy_source)
    return MigrationPlan(
        results_path=results_path,
        legacy_source=legacy_source,
        legacy_files=len(files),
        legacy_bytes=sum(file.size for file in files),
        archive_prefix=legacy_archive_prefix(results_path) if files and legacy_source == results_path else None,
    )


def migrate_run_and_archive(
    results_path: str,
    *,
    archive_prefix: str | None = None,
    writer_id: str = _MIGRATION_WRITER_ID,
) -> MigrationResult:
    """Migrate, validate, then retire one run's legacy normalized sample Parquets."""
    files, counts, validation = _migrate_and_validate(results_path, writer_id)
    resolved_archive_prefix = archive_prefix or legacy_archive_prefix(results_path)
    archived_files = archive_legacy_sample_files(
        results_path,
        resolved_archive_prefix,
        files=files,
    )
    return MigrationResult(
        results_path=results_path,
        counts=counts,
        validation=validation,
        archive_prefix=resolved_archive_prefix,
        archived_files=archived_files,
    )


def _migrate_and_validate(
    results_path: str,
    writer_id: str,
    legacy_source: str | None = None,
) -> tuple[tuple[LegacySampleFile, ...], MigrationCounts, ValidationCounts]:
    files = legacy_sample_files(legacy_source or results_path)
    counts = migrate_run(results_path, files=files, writer_id=writer_id)
    validation = validate_run(results_path, files=files)
    return files, counts, validation


def migrate_run_and_validate(
    results_path: str,
    *,
    legacy_source: str | None = None,
    writer_id: str = _MIGRATION_WRITER_ID,
) -> MigrationResult:
    """Migrate and validate one run without retiring its legacy inputs."""
    _, counts, validation = _migrate_and_validate(results_path, writer_id, legacy_source)
    return MigrationResult(
        results_path=results_path,
        counts=counts,
        validation=validation,
        archive_prefix=None,
        archived_files=0,
    )


def archive_sample_count(results_path: str) -> int:
    """The number of samples in a run's finestore archive (for post-migration verification)."""
    table = CompositeReader(results_path).scan(ARCHIVE_SAMPLES_TABLE, columns=["task"])
    return 0 if table is None else table.num_rows


def _selected_results_paths(results_paths: tuple[str, ...], records_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(results_paths)
    for records_prefix in records_prefixes:
        selected.update(record.results_path for record in list_records(records_prefix))
    return tuple(sorted(selected))


def _migration_mode(*, archive_legacy: bool, from_legacy_archive: bool) -> _MigrationMode:
    if archive_legacy and from_legacy_archive:
        raise click.UsageError("--from-legacy-archive requires --keep-legacy")
    if archive_legacy:
        return _MigrationMode.ARCHIVE_LEGACY
    if from_legacy_archive:
        return _MigrationMode.FROM_LEGACY_ARCHIVE
    return _MigrationMode.KEEP_LEGACY


@click.command()
@click.argument("results_paths", nargs=-1)
@click.option(
    "--records-prefix",
    "records_prefixes",
    multiple=True,
    help="Migrate every recorded run under this prefix. Repeatable.",
)
@click.option("--writer-id", default=_MIGRATION_WRITER_ID, help="Writer identity stamped on the migrated shards.")
@click.option(
    "--archive-legacy/--keep-legacy",
    default=False,
    help="Move validated samples_*.parquet files to region-local 30-day storage.",
)
@click.option(
    "--from-legacy-archive",
    is_flag=True,
    help="Read inputs from each run's deterministic region-local 30-day legacy archive.",
)
@click.option("--dry-run", is_flag=True, help="List eligible runs, sizes, and archive destinations without writing.")
def main(
    results_paths: tuple[str, ...],
    records_prefixes: tuple[str, ...],
    writer_id: str,
    archive_legacy: bool,
    from_legacy_archive: bool,
    dry_run: bool,
) -> None:
    """Backfill one or more RESULTS_PATHS, or every run under --records-prefix."""
    logging.basicConfig(level=logging.INFO)
    if not results_paths and not records_prefixes:
        raise click.UsageError("pass at least one RESULTS_PATH or --records-prefix")
    mode = _migration_mode(archive_legacy=archive_legacy, from_legacy_archive=from_legacy_archive)
    if any(path.startswith("s3://") for path in (*results_paths, *records_prefixes)):
        configure_coreweave_s3()
    selected = _selected_results_paths(results_paths, records_prefixes)
    migrated = archived = skipped = 0
    for results_path in selected:
        legacy_source = (
            legacy_archive_prefix(results_path) if mode is _MigrationMode.FROM_LEGACY_ARCHIVE else results_path
        )
        if mode is _MigrationMode.FROM_LEGACY_ARCHIVE and not StoragePath(legacy_source).exists():
            skipped += 1
            continue
        plan = migration_plan(results_path, legacy_source=legacy_source)
        if not plan.legacy_files:
            skipped += 1
            continue
        if dry_run:
            click.echo(json.dumps({"status": "planned", **dataclasses.asdict(plan)}, sort_keys=True))
            continue
        if mode is _MigrationMode.ARCHIVE_LEGACY:
            result = migrate_run_and_archive(results_path, writer_id=writer_id)
        else:
            result = migrate_run_and_validate(results_path, legacy_source=legacy_source, writer_id=writer_id)
        migrated += 1
        archived += result.archived_files
        click.echo(json.dumps({"status": "migrated", **dataclasses.asdict(result)}, sort_keys=True))
    click.echo(
        json.dumps(
            {
                "status": "complete" if not dry_run else "planned",
                "selected_runs": len(selected),
                "migrated_runs": migrated,
                "archived_files": archived,
                "skipped_runs": skipped,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
