# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an ordered, resumable train of in-place Datakit Parquet rewrites.

Each directory is an :class:`ArtifactStep` whose record contains the Zephyr
counters from that rewrite. The coordinator runs the steps one at a time, so a
rerun skips their cached records and resumes at the first incomplete directory.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.cluster_config import marin_temp_bucket

from scripts.ops.storage.recompress_parquet import (
    DEFAULT_WORKER_CPU,
    DEFAULT_WORKER_RAM,
    DEFAULT_WORKERS,
    RewriteOptions,
    run_migration,
)

logger = logging.getLogger(__name__)

ARTIFACT_TTL_DAYS = 30
ARTIFACT_PREFIX = "datakit-rewrite"
REWRITE_VERSION = "2026.08.18"
MARIN_DATA_PREFIX = "s3://marin-us-east-02a/marin"
HERO_DATA_PATHS = Path(__file__).with_name("hero_data_paths.json")
HERO_PARQUET_STAGES = (
    "exact_dups",
    "verified_fuzzy_dups",
    "cluster_assign",
    "minhash",
    "harrier",
    "tokenize.marin",
    "tokenize.nemotron",
)


@dataclass(frozen=True)
class RewritePrefix:
    """One quiescent directory selected for an in-place rewrite."""

    name: str
    source_glob: str


@dataclass(frozen=True)
class RewriteConfig:
    """Materialized inputs for one directory in the rewrite train."""

    source_glob: str
    workers: int
    worker_cpu: int
    worker_ram: str


class ParquetRewriteArtifact(Artifact):
    """Cached completion record and counters for one rewritten directory."""

    source_glob: str
    counters: dict[str, int | float]


SVG_REWRITE_PREFIXES = (
    RewritePrefix(
        name="svg-tokenize-a50a1068",
        source_glob="s3://marin-us-east-02a/marin/datakit/tokenize/svg_a50a1068/**/*.parquet",
    ),
)


def _rewrite_name(key: str, relative_path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", key).strip("-")[:80]
    path_digest = hashlib.sha256(relative_path.encode()).hexdigest()[:8]
    return f"{slug}-{path_digest}"


def _hero_rewrite_prefixes() -> tuple[RewritePrefix, ...]:
    paths: dict[str, str] = json.loads(HERO_DATA_PATHS.read_text())
    prefixes = []
    for stage in HERO_PARQUET_STAGES:
        stage_paths = (
            (key, relative_path) for key, relative_path in paths.items() if key == stage or key.startswith(f"{stage}/")
        )
        for key, relative_path in sorted(stage_paths):
            prefixes.append(
                RewritePrefix(
                    name=_rewrite_name(key, relative_path),
                    source_glob=f"{MARIN_DATA_PREFIX}/{relative_path}/**/*.parquet",
                )
            )
    return tuple(prefixes)


REWRITE_MANIFESTS = {
    "svg": SVG_REWRITE_PREFIXES,
    "hero": _hero_rewrite_prefixes(),
}


def _rewrite_prefix(config: RewriteConfig) -> ParquetRewriteArtifact:
    counters = run_migration(
        config.source_glob,
        workers=config.workers,
        worker_cpu=config.worker_cpu,
        worker_ram=config.worker_ram,
        options=RewriteOptions(apply=True),
    )
    return ParquetRewriteArtifact(source_glob=config.source_glob, counters=counters)


def rewrite_train(
    prefixes: Sequence[RewritePrefix],
    *,
    workers: int = DEFAULT_WORKERS,
    worker_cpu: int = DEFAULT_WORKER_CPU,
    worker_ram: str = DEFAULT_WORKER_RAM,
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
                    source_glob=prefix.source_glob,
                    workers=workers,
                    worker_cpu=worker_cpu,
                    worker_ram=worker_ram,
                ),
            )
        )
    return tuple(steps)


@contextlib.contextmanager
def _artifact_prefix(prefix: str) -> Iterator[None]:
    previous = os.environ.get("MARIN_PREFIX")
    os.environ["MARIN_PREFIX"] = prefix
    try:
        yield
    finally:
        if previous is None:
            del os.environ["MARIN_PREFIX"]
        else:
            os.environ["MARIN_PREFIX"] = previous


def run_rewrite_train(
    prefixes: Sequence[RewritePrefix],
    *,
    artifact_prefix: str,
    workers: int = DEFAULT_WORKERS,
    worker_cpu: int = DEFAULT_WORKER_CPU,
    worker_ram: str = DEFAULT_WORKER_RAM,
) -> ParquetRewriteArtifact:
    """Run the train sequentially and return the final completion artifact."""
    steps = rewrite_train(prefixes, workers=workers, worker_cpu=worker_cpu, worker_ram=worker_ram)
    logger.info("Caching Parquet rewrite records under %s", artifact_prefix)
    with _artifact_prefix(artifact_prefix):
        result = None
        for index, step in enumerate(steps, start=1):
            logger.info("Running Parquet rewrite directory %d/%d: %s", index, len(steps), step.name)
            result = run(step, max_concurrent=1)[0]
        assert result is not None
        return result


@click.command()
@click.option("--manifest", "manifest_name", type=click.Choice(sorted(REWRITE_MANIFESTS)), required=True)
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
    manifest_name: str,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    list_manifest: bool,
    apply_to_quiescent_prefixes: bool,
) -> None:
    """Rewrite every directory in a reviewed manifest in order."""
    prefixes = REWRITE_MANIFESTS[manifest_name]
    if list_manifest:
        for prefix in prefixes:
            click.echo(f"{prefix.name}\t{prefix.source_glob}")
        return
    if not apply_to_quiescent_prefixes:
        raise click.UsageError("pass --apply-to-quiescent-prefixes after confirming that every prefix is quiescent")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    artifact_prefix = marin_temp_bucket(
        ttl_days=ARTIFACT_TTL_DAYS,
        prefix=ARTIFACT_PREFIX,
        source_prefix=prefixes[0].source_glob,
    )
    result = run_rewrite_train(
        prefixes,
        artifact_prefix=artifact_prefix,
        workers=workers,
        worker_cpu=worker_cpu,
        worker_ram=worker_ram,
    )
    click.echo(f"completed {len(prefixes)} directories; final record: {result.path}")


if __name__ == "__main__":
    main()
