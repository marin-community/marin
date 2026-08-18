# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an ordered, resumable train of in-place Datakit Parquet rewrites.

Each directory is an :class:`ArtifactStep` whose record contains the Zephyr
counters from that rewrite. The coordinator runs the steps one at a time, so a
rerun skips their cached records and resumes at the first incomplete directory.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import click
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath

from scripts.ops.storage.recompress_parquet import (
    DEFAULT_WORKER_CPU,
    DEFAULT_WORKER_RAM,
    DEFAULT_WORKERS,
    RewriteMode,
    RewriteOptions,
    run_migration,
)

logger = logging.getLogger(__name__)

ARTIFACT_TTL_DAYS = 30
ARTIFACT_PREFIX = "datakit-rewrite"
REWRITE_VERSION = "2026.08.18.2"
DEFAULT_INVENTORY_MANIFEST = (
    "s3://marin-us-east-02a/marin/ops/parquet-rewrite-manifests/storage-scan-2026-08-18-100g.parquet"
)


@dataclass(frozen=True)
class RewritePrefix:
    """One set of quiescent directories selected for an in-place rewrite."""

    name: str
    source_globs: tuple[str, ...]
    inventory_files: int | None = None
    inventory_bytes: int | None = None


@dataclass(frozen=True)
class InventoryManifestRow:
    """One leaf Parquet directory assigned to a fixed rewrite step."""

    step_index: int
    step_name: str
    step_files: int
    step_bytes: int
    artifact_root: str
    artifact_files: int
    artifact_bytes: int
    source_glob: str
    directory_files: int
    directory_bytes: int


@dataclass(frozen=True)
class RewriteWorkerPool:
    """Resources for each Zephyr migration in the train."""

    workers: int = DEFAULT_WORKERS
    worker_cpu: int = DEFAULT_WORKER_CPU
    worker_ram: str = DEFAULT_WORKER_RAM


@dataclass(frozen=True)
class RewriteConfig:
    """Materialized inputs for one directory in the rewrite train."""

    source_globs: tuple[str, ...]
    pool: RewriteWorkerPool


class ParquetRewriteArtifact(Artifact):
    """Cached completion record and counters for one rewritten directory."""

    source_globs: tuple[str, ...]
    counters: dict[str, int | float]


def read_inventory_manifest(path: str) -> tuple[InventoryManifestRow, ...]:
    """Read the reviewed inventory snapshot consumed by the coordinator."""
    with StoragePath(path).open("rb") as source:
        rows = tuple(InventoryManifestRow(**row) for row in pq.read_table(source).to_pylist())
    if not rows:
        raise ValueError(f"inventory manifest is empty: {path}")
    source_globs = [row.source_glob for row in rows]
    if len(source_globs) != len(set(source_globs)):
        raise ValueError(f"inventory manifest contains duplicate source globs: {path}")
    if tuple(rows) != tuple(sorted(rows, key=lambda row: (row.step_index, row.source_glob))):
        raise ValueError(f"inventory manifest rows are not in step order: {path}")
    return rows


def inventory_rewrite_prefixes(rows: Sequence[InventoryManifestRow]) -> tuple[RewritePrefix, ...]:
    """Build the exact ordered steps recorded in an inventory manifest."""
    steps: list[RewritePrefix] = []
    current_step_index = -1
    for row in rows:
        if row.step_index == current_step_index:
            previous = steps[-1]
            if previous.name != row.step_name:
                raise ValueError(f"inventory step {row.step_index} has multiple names")
            if (previous.inventory_files, previous.inventory_bytes) != (row.step_files, row.step_bytes):
                raise ValueError(f"inconsistent totals for inventory step {row.step_name}")
            steps[-1] = RewritePrefix(
                name=previous.name,
                source_globs=(*previous.source_globs, row.source_glob),
                inventory_files=previous.inventory_files,
                inventory_bytes=previous.inventory_bytes,
            )
            continue
        if row.step_index != current_step_index + 1:
            raise ValueError(f"inventory step {row.step_name} is not contiguous")
        current_step_index = row.step_index
        steps.append(
            RewritePrefix(
                name=row.step_name,
                source_globs=(row.source_glob,),
                inventory_files=row.step_files,
                inventory_bytes=row.step_bytes,
            )
        )
    return tuple(steps)


def _rewrite_prefix(config: RewriteConfig) -> ParquetRewriteArtifact:
    counters = run_migration(
        config.source_globs,
        workers=config.pool.workers,
        worker_cpu=config.pool.worker_cpu,
        worker_ram=config.pool.worker_ram,
        options=RewriteOptions(mode=RewriteMode.APPLY),
    )
    return ParquetRewriteArtifact(source_globs=config.source_globs, counters=counters)


def rewrite_train(
    prefixes: Sequence[RewritePrefix],
    *,
    pool: RewriteWorkerPool = RewriteWorkerPool(),
) -> tuple[ArtifactStep[ParquetRewriteArtifact], ...]:
    """Build the ordered ArtifactSteps for the selected directories."""
    if not prefixes:
        raise ValueError("the Parquet rewrite train must contain at least one directory")
    names = [prefix.name for prefix in prefixes]
    if len(names) != len(set(names)):
        raise ValueError("Parquet rewrite directory names must be unique")

    steps: list[ArtifactStep[ParquetRewriteArtifact]] = []
    for prefix in prefixes:
        steps.append(
            ArtifactStep(
                name=prefix.name,
                version=REWRITE_VERSION,
                artifact_type=ParquetRewriteArtifact,
                run=_rewrite_prefix,
                build_config=lambda _ctx, prefix=prefix: RewriteConfig(
                    source_globs=prefix.source_globs,
                    pool=pool,
                ),
            )
        )
    return tuple(steps)


def run_rewrite_train(
    prefixes: Sequence[RewritePrefix],
    *,
    pool: RewriteWorkerPool = RewriteWorkerPool(),
) -> ParquetRewriteArtifact:
    """Run the train sequentially and return the final completion artifact."""
    steps = rewrite_train(prefixes, pool=pool)
    result = None
    for index, step in enumerate(steps, start=1):
        logger.info("Running Parquet rewrite directory %d/%d: %s", index, len(steps), step.name)
        result = run(step, max_concurrent=1)[0]
    assert result is not None
    return result


def _print_artifact_list(rows: Sequence[InventoryManifestRow]) -> None:
    seen_artifacts: set[str] = set()
    for row in rows:
        if row.artifact_root in seen_artifacts:
            continue
        seen_artifacts.add(row.artifact_root)
        click.echo(
            "\t".join(
                (
                    str(row.artifact_files),
                    f"{row.artifact_bytes / 1024**3:.3f}",
                    row.step_name,
                    row.artifact_root,
                )
            )
        )


@click.command()
@click.option("--inventory-manifest-path", default=DEFAULT_INVENTORY_MANIFEST, show_default=True)
@click.option("--workers", default=DEFAULT_WORKERS, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-cpu", default=DEFAULT_WORKER_CPU, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-ram", default=DEFAULT_WORKER_RAM, show_default=True)
@click.option("--list-manifest", is_flag=True, help="Print the ordered directory list without running it.")
@click.option(
    "--apply-to-quiescent-prefixes",
    is_flag=True,
    help="Confirm that every directory in the selected manifest has no active producer.",
)
def main(
    inventory_manifest_path: str,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    list_manifest: bool,
    apply_to_quiescent_prefixes: bool,
) -> None:
    """Rewrite every directory in a reviewed manifest in order."""
    rows = read_inventory_manifest(inventory_manifest_path)
    prefixes = inventory_rewrite_prefixes(rows)
    if list_manifest:
        _print_artifact_list(rows)
        return
    if not apply_to_quiescent_prefixes:
        raise click.UsageError("pass --apply-to-quiescent-prefixes after confirming that every prefix is quiescent")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    artifact_prefix = marin_temp_bucket(
        ttl_days=ARTIFACT_TTL_DAYS,
        prefix=ARTIFACT_PREFIX,
        source_prefix=prefixes[0].source_globs[0],
    )
    if marin_prefix().rstrip("/") != artifact_prefix.rstrip("/"):
        raise click.UsageError(f"MARIN_PREFIX must be {artifact_prefix} so completion records are region-local")
    logger.info("Caching Parquet rewrite records under %s", artifact_prefix)
    result = run_rewrite_train(
        prefixes,
        pool=RewriteWorkerPool(workers=workers, worker_cpu=worker_cpu, worker_ram=worker_ram),
    )
    click.echo(f"completed {len(prefixes)} directories; final record: {result.path}")


if __name__ == "__main__":
    main()
