# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write duplicate markers from a saved scatter tree, skipping the map stage.

Zephyr deletes a run's scatter when its pipeline aborts, so a reduce failure
costs the whole map -- two hours of 410 workers, twice now. Snapshot the scatter
out of the TTL prefix while the run is alive and this reads it back, so a failed
reduce costs minutes instead of a rerun.

There is no resume inside Zephyr for this: the scatter path is derived from an
execution id minted at run time, and the reduce only exists as a stage of the
pipeline that produced it. So this rebuilds the reduce directly. Each task owns
one target shard, merges that shard's slice of every mapper chunk, and writes
the same co-partitioned markers ``verify_cluster_text`` would have written.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority production --cpu 16 --memory 64GB \
        -- python experiments/datakit/scripts/reduce_saved_scatter.py \
            --scatter s3://.../user/rav/dedup/scatter_backup/c075-r2 \
            --cluster-text s3://.../user/rav/dedup/cluster_text/v11 \
            --out s3://.../user/rav/dedup/verified/v11-c075
"""

import argparse
import itertools
import logging
from collections.abc import Iterator
from typing import Any

from fray.types import ResourceConfig
from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.execution.artifact import write_artifact
from marin.execution.step_status import STATUS_SUCCESS, StatusFile, worker_id
from marin.processing.classification.deduplication.cluster_dedup import ClusterDedupParams
from marin.processing.classification.deduplication.cluster_verify import (
    _SHARED_SHARDS_KEY,
    DEFAULT_MAX_SHARD_FAILURES,
    _attr_dir,
    _write_markers,
    read_cluster_text_manifest,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.shuffle import ScatterReader

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/reduce_saved_scatter"
SCATTER_STAGE = "stage0-Map-Scatter"


def reduce_target_shard(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Merge one target shard out of the saved scatter and write its markers.

    Records arrive in global sort order, and the sort key leads with the group
    key, so every record for one normalized shard is contiguous. That is what
    lets this hand ``_write_markers`` a plain iterator per shard rather than
    buffering the whole target shard in memory.
    """
    reader = ScatterReader.from_sidecars(spec["scatter_paths"], spec["target_shard"])
    merged = reader.merge_sorted_chunks(external_sort_dir=spec["external_sort_dir"])

    written = 0
    for file_idx, records in itertools.groupby(merged, key=lambda record: record["file_idx"]):
        result = _write_markers(file_idx, records, spec["output_path"])
        written += result["markers"]
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/markers", written)
    yield {"target_shard": spec["target_shard"], "markers": written}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scatter", required=True, help="Saved scatter root holding stage0-Map-Scatter/")
    parser.add_argument("--cluster-text", required=True, help="Root whose manifest names every normalized shard")
    parser.add_argument("--out", required=True, help="Attribute tree to write markers into")
    parser.add_argument("--reduce-shards", type=int, default=2048, help="Target shards the scatter was written for")
    parser.add_argument(
        "--minimum-containment",
        type=float,
        required=True,
        help="The threshold the saved scatter was solved under, recorded on the artifact",
    )
    parser.add_argument("--max-shard-failures", type=int, default=DEFAULT_MAX_SHARD_FAILURES)
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--worker-cpu", type=float, default=8)
    parser.add_argument("--worker-ram", default="144g")
    parser.add_argument("--worker-disk", default="256g")
    # Tasks share their pod's ephemeral port range, and a reduce task opens
    # thousands of short-lived connections, so this divides the ports as much
    # as it divides the CPU. At 1 the run put 8 tasks on a pod and measured
    # 27,612 sockets in TIME_WAIT against 55,296 ports; 4 gives 2 tasks a pod.
    parser.add_argument("--task-cpu", type=float, default=4)
    parser.add_argument("--task-ram", default="12g")
    parser.add_argument("--task-disk", default="48g")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = read_cluster_text_manifest(args.cluster_text)
    shards = {shard.file_idx: shard for shard in manifest.shards}

    stage = prefix_join(args.scatter, SCATTER_STAGE)
    scatter_paths = sorted(str(path) for path in StoragePath(prefix_join(stage, "shard-*")).glob())
    if not scatter_paths:
        raise FileNotFoundError(f"{stage} holds no mapper output")
    # The trailing slash matters: _scatter_meta_path concatenates the sidecar
    # name onto this string, so a missing slash asks for "scattermetadata.msgpack".
    scatter_paths = [f"{path.rstrip('/')}/scatter/" for path in scatter_paths]
    logger.info(
        "Reducing %d mapper chunks into %d target shards of %s",
        len(scatter_paths),
        args.reduce_shards,
        args.out,
    )

    specs = [
        {
            "target_shard": target,
            "scatter_paths": scatter_paths,
            "output_path": args.out,
            "external_sort_dir": f"/tmp/zephyr-external-sort/shard-{target:05d}",
        }
        for target in range(args.reduce_shards)
    ]
    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    context = ZephyrContext(
        name="reduce-saved-scatter",
        resources=worker,
        max_workers=args.max_workers,
        max_shard_failures=args.max_shard_failures,
    )
    context.put(_SHARED_SHARDS_KEY, shards)
    outcome = context.execute(
        Dataset.from_list(specs).flat_map(reduce_target_shard),
        verbose=True,
        map_task_resources=task,
    )

    markers = sum(result["markers"] for result in outcome.results if isinstance(result, dict))

    # Markers alone are not a readable tree. Seal it the way verify_cluster_text
    # does, or the store finds an attribute directory with no manifest, no
    # artifact and no SUCCESS, and refuses to schedule against it.
    source_tags = {shard.source_key: shard.source_tag for shard in manifest.shards}
    attr_dirs = {key: _attr_dir(args.out, tag) for key, tag in source_tags.items()}
    write_copartitioned_source_manifest(output_path=args.out, attr_dirs=attr_dirs)
    result = VerifiedFuzzyDupsAttrData(
        rule=ClusterDedupParams(minimum_containment=args.minimum_containment),
        sources={
            key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dirs[key], source_tag=tag) for key, tag in source_tags.items()
        },
        counters={f"{COUNTER_PREFIX}/markers": markers, f"{COUNTER_PREFIX}/target_shards": args.reduce_shards},
    )
    write_artifact(result, args.out)
    StatusFile(args.out, worker_id=worker_id()).write_status(STATUS_SUCCESS)
    logger.info("Wrote %d markers from the saved scatter and sealed %s", markers, args.out)


if __name__ == "__main__":
    main()
