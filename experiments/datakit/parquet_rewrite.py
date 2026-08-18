# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite reviewed Parquet inventory rollups in place with a shared worker pool."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import click
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, lower, run
from marin.execution.step_runner import step_is_built
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath
from zephyr.execution import ZephyrContext

from scripts.ops.storage.recompress_parquet import (
    DEFAULT_COORDINATOR_CPU,
    DEFAULT_WORKER_CPU,
    DEFAULT_WORKER_DISK,
    DEFAULT_WORKER_RAM,
    DEFAULT_WORKERS,
    RewriteMode,
    RewriteOptions,
    create_rewrite_context,
    run_migration,
)

logger = logging.getLogger(__name__)

ARTIFACT_TTL_DAYS = 30
ARTIFACT_PREFIX = "datakit-rewrite"
REWRITE_VERSION = "2026.08.18.2"
DEFAULT_INVENTORY_MANIFEST = (
    "s3://marin-us-east-02a/marin/ops/parquet-rewrite-manifests/storage-scan-2026-08-18-100g-cap4096.parquet"
)


@dataclass(frozen=True)
class RewriteStep:
    """One manifest rollup selected for an in-place rewrite."""

    name: str
    source_globs: tuple[str, ...]
    inventory_files: int
    inventory_bytes: int


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
    worker_disk: str = DEFAULT_WORKER_DISK
    coordinator_cpu: float = DEFAULT_COORDINATOR_CPU


@dataclass(frozen=True)
class RewriteConfig:
    """Materialized inputs for one rollup in the rewrite train."""

    source_globs: tuple[str, ...]


class ParquetRewriteArtifact(Artifact):
    """Cached completion record and counters for one rewritten rollup."""

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


def inventory_rewrite_steps(rows: Sequence[InventoryManifestRow]) -> tuple[RewriteStep, ...]:
    """Build the exact ordered steps recorded in an inventory manifest."""
    steps: list[RewriteStep] = []
    current_step_index = -1
    for row in rows:
        if row.step_index == current_step_index:
            previous = steps[-1]
            if previous.name != row.step_name:
                raise ValueError(f"inventory step {row.step_index} has multiple names")
            if (previous.inventory_files, previous.inventory_bytes) != (row.step_files, row.step_bytes):
                raise ValueError(f"inconsistent totals for inventory step {row.step_name}")
            steps[-1] = RewriteStep(
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
            RewriteStep(
                name=row.step_name,
                source_globs=(row.source_glob,),
                inventory_files=row.step_files,
                inventory_bytes=row.step_bytes,
            )
        )
    return tuple(steps)


def _rewrite_rollup(config: RewriteConfig, *, context: ZephyrContext) -> ParquetRewriteArtifact:
    counters = run_migration(
        config.source_globs,
        context=context,
        options=RewriteOptions(mode=RewriteMode.APPLY),
    )
    return ParquetRewriteArtifact(source_globs=config.source_globs, counters=counters)


def rewrite_train(
    rewrite_steps: Sequence[RewriteStep],
    *,
    context: ZephyrContext,
) -> tuple[ArtifactStep[ParquetRewriteArtifact], ...]:
    """Build the ordered ArtifactSteps for the selected manifest rollups."""
    if not rewrite_steps:
        raise ValueError("the Parquet rewrite train must contain at least one rollup")
    names = [rewrite_step.name for rewrite_step in rewrite_steps]
    if len(names) != len(set(names)):
        raise ValueError("Parquet rewrite step names must be unique")

    steps: list[ArtifactStep[ParquetRewriteArtifact]] = []
    for rewrite_step in rewrite_steps:
        steps.append(
            ArtifactStep(
                name=rewrite_step.name,
                version=REWRITE_VERSION,
                artifact_type=ParquetRewriteArtifact,
                run=partial(_rewrite_rollup, context=context),
                build_config=lambda _ctx, rewrite_step=rewrite_step: RewriteConfig(
                    source_globs=rewrite_step.source_globs,
                ),
            )
        )
    return tuple(steps)


def run_rewrite_train(
    rewrite_steps: Sequence[RewriteStep],
    *,
    pool: RewriteWorkerPool = RewriteWorkerPool(),
) -> ParquetRewriteArtifact:
    """Run the train sequentially and return the final completion artifact."""
    context = create_rewrite_context(
        workers=pool.workers,
        worker_cpu=pool.worker_cpu,
        worker_ram=pool.worker_ram,
        worker_disk=pool.worker_disk,
        coordinator_cpu=pool.coordinator_cpu,
    )
    steps = rewrite_train(rewrite_steps, context=context)
    pending_steps = tuple(step for step in steps if not step_is_built(lower(step)))
    logger.info("Running %d pending Parquet rewrite steps out of %d", len(pending_steps), len(steps))
    if pending_steps:
        with context:
            for index, step in enumerate(pending_steps, start=1):
                logger.info("Running pending Parquet rewrite step %d/%d: %s", index, len(pending_steps), step.name)
                run(step, max_concurrent=1)
    return ParquetRewriteArtifact.raw_load(steps[-1].path())


def _print_manifest(rows: Sequence[InventoryManifestRow]) -> None:
    for row in rows:
        click.echo(
            "\t".join(
                (
                    str(row.directory_files),
                    f"{row.directory_bytes / 1024**3:.3f}",
                    row.step_name,
                    row.source_glob,
                )
            )
        )


@click.command()
@click.option("--inventory-manifest-path", default=DEFAULT_INVENTORY_MANIFEST, show_default=True)
@click.option("--workers", default=DEFAULT_WORKERS, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-cpu", default=DEFAULT_WORKER_CPU, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-ram", default=DEFAULT_WORKER_RAM, show_default=True)
@click.option("--worker-disk", default=DEFAULT_WORKER_DISK, show_default=True)
@click.option(
    "--coordinator-cpu",
    default=DEFAULT_COORDINATOR_CPU,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
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
    worker_disk: str,
    coordinator_cpu: float,
    list_manifest: bool,
    apply_to_quiescent_prefixes: bool,
) -> None:
    """Rewrite each rollup in a reviewed manifest in order."""
    rows = read_inventory_manifest(inventory_manifest_path)
    rewrite_steps = inventory_rewrite_steps(rows)
    if list_manifest:
        _print_manifest(rows)
        return
    if not apply_to_quiescent_prefixes:
        raise click.UsageError("pass --apply-to-quiescent-prefixes after confirming that every prefix is quiescent")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    artifact_prefix = marin_temp_bucket(
        ttl_days=ARTIFACT_TTL_DAYS,
        prefix=ARTIFACT_PREFIX,
        source_prefix=rewrite_steps[0].source_globs[0],
    )
    if StoragePath(marin_prefix()) != StoragePath(artifact_prefix):
        raise click.UsageError(f"MARIN_PREFIX must be {artifact_prefix} so completion records are region-local")
    logger.info("Caching Parquet rewrite records under %s", artifact_prefix)
    result = run_rewrite_train(
        rewrite_steps,
        pool=RewriteWorkerPool(
            workers=workers,
            worker_cpu=worker_cpu,
            worker_ram=worker_ram,
            worker_disk=worker_disk,
            coordinator_cpu=coordinator_cpu,
        ),
    )
    click.echo(f"completed {len(rewrite_steps)} rollups; final record: {result.path}")


if __name__ == "__main__":
    main()
