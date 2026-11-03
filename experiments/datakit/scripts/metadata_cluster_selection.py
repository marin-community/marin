# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build restartable selection ledgers for one complete frozen Hero cluster."""

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.scripts.metadata_cluster_census import (
    aggregate_census_reports,
    inventory_shards,
)
from experiments.datakit.store.metadata import (
    METADATA_SCHEMA,
    ShardManifest,
    SparseDuplicateSource,
    duplicate_source_from_catalogs,
    export_cluster_metadata_shard,
)

logger = logging.getLogger(__name__)


def _read_json(path: str) -> dict:
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "r") as stream:
        return json.load(stream)


def _write_json(path: str, value: dict) -> None:
    fs, resolved = filesystem_for(path)
    fs.makedirs(filesystem_for(str(StoragePath(path).parent))[1], exist_ok=True)
    with fs.open(resolved, "w") as stream:
        json.dump(value, stream, sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _absolute(prefix: str, value: str) -> str:
    return value if "://" in value or value.startswith("/") else prefix_join(prefix, value)


def source_bindings(
    inventory: dict, exact_catalog_root: str, fuzzy_catalog_root: str
) -> dict[str, SparseDuplicateSource]:
    """Bind source directories to the exact frozen duplicate catalogs."""
    prefix = str(StoragePath(inventory["store_uri"]).parent.parent)
    exact = inventory["records"][exact_catalog_root]["record"]["result"]["sources"]
    fuzzy = inventory["records"][fuzzy_catalog_root]["record"]["result"]["sources"]
    basenames: dict[str, list[str]] = {}
    for shard in inventory_shards(inventory, None):
        basenames.setdefault(shard.source, []).append(StoragePath(shard.path).name)
    return {
        source["marin_source"]: duplicate_source_from_catalogs(
            artifact_prefix=prefix,
            source_key=source["source_key"],
            shard_basenames=tuple(basenames[source["marin_source"]]),
            exact_sources=exact,
            fuzzy_sources=fuzzy,
        )
        for source in inventory["sources"]
    }


def shard_manifest(inventory: dict, source: dict, cluster_path: str, binding: SparseDuplicateSource) -> ShardManifest:
    """Resolve one selected shard without materializing all per-shard manifests."""
    prefix = str(StoragePath(inventory["store_uri"]).parent.parent)
    basename = StoragePath(cluster_path).name
    roots = source["artifact_roots"]
    decontam = inventory["records"][roots["decontam"]]["record"]["result"]["main_output_dir"]
    return ShardManifest(
        marin_source=source["marin_source"],
        normalized=prefix_join(binding.normalized_dir, basename),
        decontam=prefix_join(_absolute(prefix, decontam), basename),
        cluster=cluster_path,
        quality=prefix_join(_absolute(prefix, roots["quality"]), basename),
        content_type=prefix_join(_absolute(prefix, roots["content_type"]), basename),
        exact_dedup=prefix_join(binding.exact_attr_dir, basename),
        dedup="" if source["fuzzy_exempt"] else prefix_join(binding.fuzzy_attr_dir, basename),
        bucket_edges=inventory["bucket_edges"],
    )


def census_target_counts(
    inventory: dict, census_inventory: dict, census_reports: list[dict], target_cluster: int
) -> dict:
    """Use only a full validated census with identical cluster shard inputs for pruning."""
    aggregate_census_reports(census_inventory, census_reports)
    current = [(shard.source, shard.path) for shard in inventory_shards(inventory, None)]
    original = [(shard.source, shard.path) for shard in inventory_shards(census_inventory, None)]
    if (
        current != original
        or inventory["store_config"]["cluster_view"] != census_inventory["store_config"]["cluster_view"]
    ):
        raise ValueError("Census cluster shards differ from the selection inventory")
    return {
        item["path"]: item["cluster_counts"].get(str(target_cluster), 0)
        for report in census_reports
        for item in report["completed"]
    }


def run_selection_task(
    inventory: dict,
    output_prefix: str,
    *,
    exact_catalog_root: str,
    fuzzy_catalog_root: str,
    target_cluster: int,
    task_index: int,
    task_count: int,
    workers: int,
    batch_size: int = 8192,
    checkpoint_interval: float = 60,
    census_counts: dict[str, int] | None = None,
    max_shards: int | None = None,
) -> dict:
    """Export one task's cluster ledger with immutable attempt outputs and resumable checkpoints."""
    if not 0 <= task_index < task_count or workers <= 0 or (max_shards is not None and max_shards <= 0):
        raise ValueError("Provide a valid task partition and positive worker count")
    bindings = source_bindings(inventory, exact_catalog_root, fuzzy_catalog_root)
    shards = inventory_shards(inventory, None)
    if census_counts is not None and set(census_counts) != {shard.path for shard in shards}:
        raise ValueError("Census pruning must cover the complete cluster shard inventory")
    sources = {source["marin_source"]: source for source in inventory["sources"]}
    config = {
        "inventory_sha256": _digest(inventory),
        "exact_catalog_root": exact_catalog_root,
        "fuzzy_catalog_root": fuzzy_catalog_root,
        "target_cluster": target_cluster,
        "task_index": task_index,
        "task_count": task_count,
        "census_counts_sha256": _digest(census_counts) if census_counts is not None else None,
        "max_shards": max_shards,
    }
    checkpoint = prefix_join(output_prefix, f"task-{task_index:05d}-of-{task_count:05d}.json")
    fs, path = filesystem_for(checkpoint)
    report: dict = _read_json(checkpoint) if fs.exists(path) else {"config": config, "complete": False, "completed": []}
    if report["config"] != config:
        raise ValueError("Selection checkpoint differs from the requested frozen inputs or task partition")
    completed = {item["index"] for item in report["completed"]}
    if len(completed) != len(report["completed"]):
        raise ValueError("Repeated shard indexes in selection checkpoint")
    assigned = [(index, shard) for index, shard in enumerate(shards) if index % task_count == task_index]
    if max_shards is not None:
        assigned = [item for item in assigned if census_counts is None or census_counts[item[1].path] > 0][:max_shards]
    assigned_indexes = {index for index, _ in assigned}
    for item in report["completed"]:
        index = item["index"]
        if index not in assigned_indexes or item["cluster_path"] != shards[index].path:
            raise ValueError("Selection checkpoint contains an unexpected shard")
        if item["status"] == "selected":
            confirmation = _read_json(item["completion_marker"])
            source = sources[shards[index].source]
            expected_manifest = shard_manifest(inventory, source, shards[index].path, bindings[shards[index].source])
            if (
                confirmation["ordered_occurrences_sha256"] != item["ordered_occurrences_sha256"]
                or confirmation["manifest"] != dataclasses.asdict(expected_manifest)
                or confirmation["target_cluster"] != target_cluster
            ):
                raise ValueError("Selection shard completion marker differs from its pinned inputs or checkpoint")
            output_fs, output = filesystem_for(confirmation["output"])
            if not output_fs.exists(output):
                raise FileNotFoundError(confirmation["output"])
        elif item["status"] != "census_zero" or census_counts is None or census_counts[item["cluster_path"]] != 0:
            raise ValueError("Selection checkpoint contains an unverified skipped shard")
    attempt = uuid.uuid4().hex
    remaining = iter((index, shard) for index, shard in assigned if index not in completed)
    last_checkpoint = time.monotonic()

    def export(item: tuple) -> dict:
        index, shard = item
        if census_counts is not None and census_counts[shard.path] == 0:
            return {"index": index, "cluster_path": shard.path, "source": shard.source, "status": "census_zero"}
        source = sources[shard.source]
        manifest = shard_manifest(inventory, source, shard.path, bindings[shard.source])
        destination = prefix_join(output_prefix, f"task={task_index:05d}/attempt={attempt}/shard={index:08d}")
        result = export_cluster_metadata_shard(
            manifest,
            destination,
            cluster_view=inventory["store_config"]["cluster_view"],
            target_cluster=target_cluster,
            fuzzy_exempt=source["fuzzy_exempt"],
            duplicate_source=bindings[shard.source],
            batch_size=batch_size,
        )
        if census_counts is not None and result["counters"]["datakit_store/records_in"] != census_counts[shard.path]:
            raise ValueError(f"Selection cluster count differs from the frozen census for {shard.path}")
        return {
            "index": index,
            "cluster_path": shard.path,
            "source": shard.source,
            "status": "selected",
            "completion_marker": prefix_join(destination, "_complete.json"),
            "output": result["output"],
            "selected_occurrences": result["selected_occurrences"],
            "quality_counts": {str(key): value for key, value in result["quality_counts"].items()},
            "ordered_occurrences_sha256": result["ordered_occurrences_sha256"],
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(remaining, workers)):
            futures = [executor.submit(export, item) for item in batch]
            try:
                for future in futures:
                    report["completed"].append(future.result())
            except Exception:
                _write_json(checkpoint, report)
                raise
            logger.info("Selection task %d: %d/%d shards", task_index, len(report["completed"]), len(assigned))
            if time.monotonic() - last_checkpoint >= checkpoint_interval:
                _write_json(checkpoint, report)
                last_checkpoint = time.monotonic()
    quality_counts: Counter[str] = Counter()
    for item in report["completed"]:
        quality_counts.update(item.get("quality_counts", {}))
    report["quality_counts"] = dict(quality_counts)
    report["selected_occurrences"] = sum(item.get("selected_occurrences", 0) for item in report["completed"])
    report["assigned_shards"] = len(assigned)
    report["sampled"] = max_shards is not None
    report["complete"] = len(report["completed"]) == len(assigned)
    report["validation"] = "selection ledger only; upstream reconstruction and embeddings not performed"
    _write_json(checkpoint, report)
    return report


def aggregate_selection_reports(
    inventory: dict,
    reports: list[dict],
    *,
    exact_catalog_root: str,
    fuzzy_catalog_root: str,
    target_cluster: int,
    census_counts: dict[str, int] | None,
    max_shards: int | None = None,
) -> dict:
    """Validate complete task coverage and aggregate internal ledgers without remote I/O."""
    if not reports:
        raise ValueError("No selection task reports supplied")
    shards = inventory_shards(inventory, None)
    if census_counts is not None and set(census_counts) != {shard.path for shard in shards}:
        raise ValueError("Selection census does not cover the frozen shard inventory")
    task_count = reports[0]["config"]["task_count"]
    if sorted(report["config"]["task_index"] for report in reports) != list(range(task_count)):
        raise ValueError("Selection reports do not cover every task exactly once")
    sources: dict[str, dict] = {}
    all_outputs: set[str] = set()
    inventory_digest = _digest(inventory)
    counts_digest = _digest(census_counts) if census_counts is not None else None
    selected = 0
    skipped = 0
    quality: Counter[str] = Counter()
    for report in reports:
        index = report["config"]["task_index"]
        expected_config = {
            "inventory_sha256": inventory_digest,
            "exact_catalog_root": exact_catalog_root,
            "fuzzy_catalog_root": fuzzy_catalog_root,
            "target_cluster": target_cluster,
            "task_index": index,
            "task_count": task_count,
            "census_counts_sha256": counts_digest,
            "max_shards": max_shards,
        }
        if report["config"] != expected_config or not report["complete"]:
            raise ValueError("Incomplete selection task or mismatched frozen inputs")
        assigned = [(i, shard) for i, shard in enumerate(shards) if i % task_count == index]
        if max_shards is not None:
            assigned = [item for item in assigned if census_counts is None or census_counts[item[1].path] > 0][
                :max_shards
            ]
        expected = {i: shard for i, shard in assigned}
        items = report["completed"]
        if len(items) != len(expected) or {item["index"] for item in items} != set(expected):
            raise ValueError("Selection task has missing, repeated, or unexpected shards")
        task_selected = 0
        task_quality: Counter[str] = Counter()
        for item in items:
            shard = expected[item["index"]]
            if item["cluster_path"] != shard.path or item["source"] != shard.source:
                raise ValueError("Selection shard identity differs from the frozen inventory")
            if item["status"] == "census_zero":
                if census_counts is None or census_counts[shard.path] != 0:
                    raise ValueError("Selection task skipped a shard without a census zero proof")
                skipped += 1
                continue
            if item["status"] != "selected":
                raise ValueError("Unknown selection shard status")
            count = item["selected_occurrences"]
            if count < 0 or sum(item["quality_counts"].values()) != count:
                raise ValueError("Selection shard quality counts do not reconcile")
            if census_counts is not None and count > census_counts[shard.path]:
                raise ValueError("Selected count exceeds the frozen cluster assignment count")
            output = StoragePath(item["output"])
            marker = StoragePath(item["completion_marker"])
            if (
                output.name != "metadata.parquet"
                or marker.name != "_complete.json"
                or output.parent != marker.parent
                or output.parent.name != f"shard={item['index']:08d}"
                or item["output"] in all_outputs
            ):
                raise ValueError("Selection output paths are repeated or do not name their shard")
            all_outputs.add(item["output"])
            source = sources.setdefault(
                shard.source, {"selected_occurrences": 0, "quality_counts": Counter(), "ledgers": []}
            )
            source["selected_occurrences"] += count
            source["quality_counts"].update(item["quality_counts"])
            source["ledgers"].append(item["output"])
            task_selected += count
            task_quality.update(item["quality_counts"])
        if task_selected != report["selected_occurrences"] or dict(task_quality) != report["quality_counts"]:
            raise ValueError("Selection task totals do not reconcile with its completed shards")
        selected += task_selected
        quality.update(task_quality)
    return {
        "inventory_sha256": _digest(inventory),
        "target_cluster": target_cluster,
        "task_count": task_count,
        "complete": True,
        "complete_cluster": max_shards is None,
        "selected_occurrences": selected,
        "quality_counts": dict(quality),
        "census_zero_shards": skipped,
        "completed_shards": skipped + len(all_outputs),
        "selection_shards": len(all_outputs),
        "sources": {
            name: {**value, "quality_counts": dict(value["quality_counts"])} for name, value in sorted(sources.items())
        },
        "validation": "selection reports and coverage only; output bytes and upstream reconstruction not verified",
    }


def verify_selection_outputs(
    inventory: dict,
    reports: list[dict],
    *,
    exact_catalog_root: str,
    fuzzy_catalog_root: str,
    target_cluster: int,
    batch_size: int = 8192,
) -> dict:
    """Read completion markers and stream ledger columns to verify counts and occurrence hashes.

    Call aggregate_selection_reports first to validate task coverage. This
    separate I/O pass checks the referenced output bytes, not upstream text.
    """
    bindings = source_bindings(inventory, exact_catalog_root, fuzzy_catalog_root)
    sources = {source["marin_source"]: source for source in inventory["sources"]}
    shards = inventory_shards(inventory, None)
    verified_shards = 0
    verified_rows = 0
    for report in reports:
        for item in report["completed"]:
            if item["status"] != "selected":
                continue
            shard = shards[item["index"]]
            source = sources[shard.source]
            binding = bindings[shard.source]
            manifest = shard_manifest(inventory, source, shard.path, binding)
            marker = _read_json(item["completion_marker"])
            duplicate_proof = {
                "source_key": binding.source_key,
                "normalized_dir": binding.normalized_dir,
                "exact_attr_dir": binding.exact_attr_dir,
                "fuzzy_attr_dir": binding.fuzzy_attr_dir,
                "shard_inventory_sha256": (
                    hashlib.sha256(json.dumps(binding.shard_basenames, separators=(",", ":")).encode()).hexdigest()
                ),
            }
            if (
                marker["duplicate_source"] != duplicate_proof
                or {str(key): value for key, value in marker["quality_counts"].items()} != item["quality_counts"]
                or marker["manifest"] != dataclasses.asdict(manifest)
                or marker["target_cluster"] != target_cluster
                or marker["cluster_view"] != inventory["store_config"]["cluster_view"]
                or marker["fuzzy_exempt"] != source["fuzzy_exempt"]
                or marker["output"] != item["output"]
                or marker["selected_occurrences"] != item["selected_occurrences"]
                or marker["ordered_occurrences_sha256"] != item["ordered_occurrences_sha256"]
            ):
                raise ValueError("Selection completion marker differs from the pinned shard or task report")
            fs, path = filesystem_for(item["output"])
            occurrence_hash = hashlib.sha256()
            counts: Counter[str] = Counter()
            previous_row = -1
            with fs.open(path, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                if not parquet.schema_arrow.equals(METADATA_SCHEMA):
                    raise ValueError("Selection ledger schema differs from the expected metadata schema")
                for batch in parquet.iter_batches(
                    columns=["id", "normalized_row", "cluster_id", "quality_bucket", "marin_source", "normalized_shard"],
                    batch_size=batch_size,
                ):
                    for row in batch.to_pylist():
                        if (
                            row["cluster_id"] != target_cluster
                            or row["marin_source"] != shard.source
                            or row["normalized_shard"] != manifest.normalized
                            or row["normalized_row"] <= previous_row
                        ):
                            raise ValueError(
                                "Selection ledger contains a wrong-cluster, wrong-source, or repeated occurrence"
                            )
                        previous_row = row["normalized_row"]
                        occurrence_hash.update(json.dumps([row["id"], previous_row], separators=(",", ":")).encode())
                        occurrence_hash.update(b"\n")
                        counts[str(row["quality_bucket"])] += 1
                if parquet.metadata.num_rows != item["selected_occurrences"]:
                    raise ValueError("Selection ledger row count differs from its completed report")
            if (
                occurrence_hash.hexdigest() != item["ordered_occurrences_sha256"]
                or dict(counts) != item["quality_counts"]
            ):
                raise ValueError("Selection ledger occurrence hash or quality counts do not match")
            verified_shards += 1
            verified_rows += sum(counts.values())
    return {"verified_shards": verified_shards, "verified_occurrences": verified_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--exact-catalog-root", required=True)
    parser.add_argument("--fuzzy-catalog-root", required=True)
    parser.add_argument("--cluster", type=int, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--checkpoint-interval", type=float, default=60)
    parser.add_argument("--max-shards", type=int, help="Bounded smoke only: candidate shards per task")
    parser.add_argument("--census-prefix")
    parser.add_argument("--census-inventory")
    args = parser.parse_args()
    info = get_job_info()
    task_index = args.task_index if args.task_index is not None else info.task_index if info is not None else None
    task_count = args.task_count if args.task_count is not None else info.num_tasks if info is not None else None
    if task_index is None or task_count is None:
        parser.error("Supply task index/count outside Iris")
    inventory = _read_json(args.inventory)
    counts = None
    if args.census_prefix or args.census_inventory:
        if not args.census_prefix or not args.census_inventory:
            parser.error("Supply both --census-prefix and --census-inventory")
        census_fs, census_path = filesystem_for(args.census_prefix)
        reports = [
            _read_json(census_fs.unstrip_protocol(path))
            for path in sorted(census_fs.glob(prefix_join(census_path, "task-*-of-*.json")))
        ]
        counts = census_target_counts(inventory, _read_json(args.census_inventory), reports, args.cluster)
    result = run_selection_task(
        inventory,
        args.output_prefix,
        exact_catalog_root=args.exact_catalog_root,
        fuzzy_catalog_root=args.fuzzy_catalog_root,
        target_cluster=args.cluster,
        task_index=task_index,
        task_count=task_count,
        workers=args.workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        census_counts=counts,
        max_shards=args.max_shards,
    )
    logger.info("Selected %d document occurrences", result["selected_occurrences"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
