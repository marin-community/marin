# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue a capped issue #6854 connected-components run without recomputing MinHash.

The original marker artifact is first copied to an immutable snapshot. The
connected-components state under the original dedup path is then resumed to a
higher iteration ceiling, and a separate report is rendered from the converged
artifact.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact, read_record, write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
    compute_fuzzy_dups_attrs,
)
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.reports.dedup import dedup_report
from experiments.datakit.scripts.dedup_ab_run import MinhashCollection, _assert_variant

logger = logging.getLogger(__name__)


def _copy_file(item: tuple[str, str]) -> int:
    source, destination = item
    source_path = StoragePath(source)
    destination_path = StoragePath(destination)
    if (source_path.scheme, source_path.netloc) != (destination_path.scheme, destination_path.netloc):
        raise ValueError(f"Snapshot must stay on one storage backend: {source} -> {destination}")

    destination_path.parent.mkdirs()
    filesystem, source_key = url_to_fs(source)
    destination_filesystem, destination_key = url_to_fs(destination)
    if filesystem.protocol != destination_filesystem.protocol:
        raise ValueError(f"Snapshot filesystem mismatch: {source} -> {destination}")
    filesystem.copy(source_key, destination_key)
    source_size = int(filesystem.size(source_key))
    destination_size = int(destination_filesystem.size(destination_key))
    if source_size <= 0 or destination_size != source_size:
        raise AssertionError(
            f"Snapshot copy size mismatch: {source} ({source_size}) -> {destination} ({destination_size})"
        )
    return source_size


def snapshot_dedup_outputs(
    *,
    dedup_path: str,
    snapshot_path: str,
    copy_workers: int,
) -> FuzzyDupsAttrData:
    """Copy every marker shard and write a self-contained snapshot artifact."""
    if read_record(snapshot_path) is not None:
        snapshot = read_artifact(snapshot_path, FuzzyDupsAttrData)
        shard_counts = {
            source_main_dir: len((StoragePath(source.attr_dir) / "*.parquet").glob())
            for source_main_dir, source in snapshot.sources.items()
        }
        missing = sorted(source for source, count in shard_counts.items() if count == 0)
        if missing:
            raise FileNotFoundError(f"Existing snapshot has no marker shards for {missing[:5]}")
        logger.info("Reusing existing snapshot with %d marker shards at %s", sum(shard_counts.values()), snapshot_path)
        return snapshot

    dedup = read_artifact(dedup_path, FuzzyDupsAttrData)
    output_root = StoragePath(f"{dedup_path.rstrip('/')}/outputs")
    copy_items: list[tuple[str, str]] = []
    snapshot_sources: dict[str, FuzzyDupsPerSource] = {}
    for source_main_dir, source in dedup.sources.items():
        source_attr_dir = StoragePath(source.attr_dir)
        relative_dir = source_attr_dir.relative_to(output_root)
        destination_dir = StoragePath(f"{snapshot_path.rstrip('/')}/outputs") / relative_dir
        marker_files = sorted((source_attr_dir / "*.parquet").glob(), key=str)
        if not marker_files:
            raise FileNotFoundError(f"No marker shards under {source_attr_dir}")
        copy_items.extend((str(path), str(destination_dir / path.name)) for path in marker_files)
        snapshot_sources[source_main_dir] = FuzzyDupsPerSource(attr_dir=str(destination_dir))

    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        copied_bytes = sum(pool.map(_copy_file, copy_items))

    snapshot = dedup.model_copy(update={"sources": snapshot_sources})
    write_artifact(snapshot, snapshot_path)
    logger.info(
        "Snapshotted %d marker shards (%d bytes) from %s to %s",
        len(copy_items),
        copied_bytes,
        dedup_path,
        snapshot_path,
    )
    return snapshot


def continue_dedup(
    *,
    variant: str,
    code_ref: str,
    output_prefix: str,
    max_iterations: int,
    dedup_parallelism: int,
    snapshot_name: str,
    report_name: str,
    copy_workers: int,
) -> None:
    """Snapshot capped markers, resume CC, and write converged artifact/report records."""
    _assert_variant(variant)
    variant_root = f"{output_prefix.rstrip('/')}/{variant}"
    dedup_path = f"{variant_root}/dedup"
    snapshot_path = f"{variant_root}/{snapshot_name}"
    report_path = f"{variant_root}/{report_name}"
    snapshot = snapshot_dedup_outputs(
        dedup_path=dedup_path,
        snapshot_path=snapshot_path,
        copy_workers=copy_workers,
    )

    collection = read_artifact(f"{variant_root}/minhash-combined", MinhashCollection)
    worker = ResourceConfig(cpu=2, ram="8g", disk="16g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="16g", preemptible=False)
    converged = compute_fuzzy_dups_attrs(
        inputs=collection.inputs,
        output_path=dedup_path,
        cc_max_iterations=max_iterations,
        cc_resume=True,
        max_parallelism=dedup_parallelism,
        worker_resources=worker,
        coordinator_resources=coordinator,
    )
    write_artifact(converged, dedup_path)

    report = dedup_report(report_path, converged)
    write_artifact(report, report_path)
    manifest: dict[str, Any] = {
        "variant": variant,
        "code_ref": code_ref,
        "dedup_path": dedup_path,
        "capped_snapshot_path": snapshot_path,
        "capped_counters": snapshot.counters,
        "converged_counters": converged.counters,
        "report_path": report_path,
        "max_iterations": max_iterations,
        "dedup_parallelism": dedup_parallelism,
    }
    manifest_path = f"{variant_root}/continuation.json"
    StoragePath(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Wrote continuation manifest to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "treatment"), required=True)
    parser.add_argument("--code-ref", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--dedup-parallelism", type=int, default=512)
    parser.add_argument("--snapshot-name", default="dedup-cap50")
    parser.add_argument("--report-name", default="report-converged")
    parser.add_argument("--copy-workers", type=int, default=32)
    args = parser.parse_args()
    configure_logging(logging.INFO)
    continue_dedup(
        variant=args.variant,
        code_ref=args.code_ref,
        output_prefix=args.output_prefix,
        max_iterations=args.max_iterations,
        dedup_parallelism=args.dedup_parallelism,
        snapshot_name=args.snapshot_name,
        report_name=args.report_name,
        copy_workers=args.copy_workers,
    )


if __name__ == "__main__":
    main()
