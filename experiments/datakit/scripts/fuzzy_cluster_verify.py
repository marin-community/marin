# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solve every materialized fuzzy-duplicate cluster and write duplicate markers.

Reads the cluster-grouped candidate text, solves each cluster with the
containment rule, and writes the decisions back into a co-partitioned
attribute tree that the store can filter on. The text is already grouped, so
this stage is an embarrassingly parallel map: no memory store, no re-join, and
no cross-cluster communication.

Output rows follow the shape the store already consumes::

    {id, dup_doc, dup_cluster_id, dup_representative_id,
     dup_representative_source_tag, dup_containment, dup_jaccard,
     dup_novel_tokens, dup_comparisons}

One file per normalized shard, named after it, so the tree is co-partitioned
with the normalized data and with every other attribute tree.

    uv run iris --cluster=cw-us-east-02a job run --no-wait --priority batch \
        --cpu 4 --memory 16GB -- python experiments/datakit/scripts/fuzzy_cluster_verify.py \
            --cluster-text s3://.../user/rav/dedup/cluster_text/v1 \
            --out s3://.../user/rav/dedup/verified/v1
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    ClusterDocument,
    find_duplicates,
)
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_verify"
_SHARED_SHARDS_KEY = "fuzzy_cluster_verify_shards"

_MARKER_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("dup_representative_id", pa.string(), nullable=False),
        pa.field("dup_representative_source_tag", pa.string(), nullable=False),
        pa.field("dup_containment", pa.float32(), nullable=False),
        pa.field("dup_jaccard", pa.float32(), nullable=False),
        pa.field("dup_novel_tokens", pa.int32(), nullable=False),
        pa.field("dup_comparisons", pa.int32(), nullable=False),
    ]
)

_SIZE_BINS = (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)


def _size_bin(size: int) -> str:
    for edge in _SIZE_BINS:
        if size < edge:
            return f"{edge:06d}"
    return "999999"


def solve_text_shard(path: str, params: ClusterDedupParams) -> Iterator[dict[str, Any]]:
    """Solve every cluster in one grouped text file.

    The file is written sorted by ``cluster_key``, so a cluster is a contiguous
    run and the whole file streams without holding more than one cluster.
    """
    started = time.monotonic()
    columns = ["cluster_key", "dup_cluster_id", "id", "text", "file_idx", "source_tag"]
    clusters = 0
    duplicates = 0
    documents = 0
    chars = 0
    pending: list[dict[str, Any]] = []
    current: str | None = None

    def solve(members: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal clusters, duplicates
        clusters += 1
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/cluster_size/{_size_bin(len(members))}", 1)
        if len(members) < 2:
            counters.pipeline.update_counter(f"{COUNTER_PREFIX}/singleton_groups", 1)
            return
        cluster = [
            ClusterDocument(id=row["id"], text=row["text"], file_idx=row["file_idx"], source_tag=row["source_tag"])
            for row in members
        ]
        for removal in find_duplicates(cluster, params):
            member = members[removal.member_index]
            representative = members[removal.representative_index]
            duplicates += 1
            yield {
                "file_idx": member["file_idx"],
                "id": member["id"],
                "dup_doc": True,
                "dup_cluster_id": member["dup_cluster_id"],
                "dup_representative_id": representative["id"],
                "dup_representative_source_tag": representative["source_tag"],
                "dup_containment": removal.containment,
                "dup_jaccard": removal.jaccard,
                "dup_novel_tokens": removal.novel_tokens,
                "dup_comparisons": removal.comparisons,
            }

    # The file is written sorted by cluster_key, so a cluster is a contiguous
    # run and only one cluster is ever resident. Reading the whole file first
    # would put several gigabytes of text in Python objects at once.
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                documents += 1
                chars += len(row["text"])
                if row["cluster_key"] != current:
                    if pending:
                        yield from solve(pending)
                    pending = []
                    current = row["cluster_key"]
                pending.append(row)
    if pending:
        yield from solve(pending)

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/documents", documents)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/text_chars", chars)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/clusters", clusters)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicates", duplicates)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/solve_seconds_milli", int((time.monotonic() - started) * 1000))


def _write_markers(file_idx: int, records: Iterator[dict[str, Any]], output_path: str) -> dict[str, Any]:
    """Write one shard's markers into the co-partitioned attribute tree."""
    shards: dict[int, dict[str, str]] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
    shard = shards[file_idx]
    path = prefix_join(prefix_join(output_path, f"outputs/{shard['source_tag']}"), shard["basename"])
    written = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal written
        for record in records:
            written += 1
            yield {field.name: record[field.name] for field in _MARKER_SCHEMA}

    result = write_parquet_file(rows(), path, schema=_MARKER_SCHEMA)
    return {**result, "file_idx": file_idx, "markers": written}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-text", required=True, help="Root written by fuzzy_cluster_text.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-containment", type=float, default=0.98)
    parser.add_argument("--no-zero-novel-tokens", action="store_true")
    parser.add_argument("--probe-ngrams", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=32)
    parser.add_argument("--exact-scan-maximum", type=int, default=256)
    parser.add_argument("--limit-files", type=int, default=0, help="Solve only the first N text files")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="192g")
    parser.add_argument("--worker-disk", default="256g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="12g")
    parser.add_argument("--task-disk", default="48g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = json.loads(StoragePath(prefix_join(args.cluster_text, "manifest.json")).read_bytes())
    shards = {shard["file_idx"]: shard for shard in manifest["shards"]}

    text_dir = prefix_join(args.cluster_text, "text")
    fs, root = url_to_fs(text_dir)
    names = sorted(str(path).rsplit("/", 1)[-1] for path in fs.ls(root, detail=False) if str(path).endswith(".parquet"))
    paths = [prefix_join(text_dir, name) for name in names]
    if args.limit_files:
        paths = paths[: args.limit_files]

    params = ClusterDedupParams(
        minimum_containment=args.minimum_containment,
        accept_zero_novel_tokens=not args.no_zero_novel_tokens,
        probe_ngrams=args.probe_ngrams,
        maximum_candidates=args.maximum_candidates,
        exact_scan_maximum=args.exact_scan_maximum,
    )
    logger.info("Solving %d grouped text files with %s", len(paths), params.model_dump_json())

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    context = ZephyrContext(name="fuzzy-cluster-verify", resources=worker, max_workers=args.max_workers)
    context.put(_SHARED_SHARDS_KEY, shards)

    pipeline = (
        Dataset.from_list(paths)
        .flat_map(lambda path: solve_text_shard(path, params))
        .group_by(
            key=lambda record: record["file_idx"],
            reducer=lambda file_idx, records: _write_markers(file_idx, records, args.out),
            sort_by=lambda record: record["id"],
            num_output_shards=min(len(shards), 8192),
        )
    )
    outcome = context.execute(pipeline, verbose=True, map_task_resources=task, reduce_task_resources=task)

    # Consumers resolve an attribute tree through its source manifest, the same
    # way they resolve every other co-partitioned Datakit output.
    write_copartitioned_source_manifest(
        output_path=args.out,
        attr_dirs={
            shard["source_key"]: prefix_join(args.out, f"outputs/{shard['source_tag']}")
            for shard in manifest["shards"]
        },
    )
    markers = sum(result["markers"] for result in outcome.results)
    payload = {
        "cluster_text": args.cluster_text,
        "params": params.model_dump(mode="json"),
        "text_files": len(paths),
        "markers": markers,
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Wrote %d duplicate markers", markers)


if __name__ == "__main__":
    main()
