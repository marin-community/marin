# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1/2: sample duplicate clusters from the dedup run's cc/it_10 output.

Scans the connected-components output (one row per input doc, with
``component_id`` = cluster id and ``adjacency_list`` of cluster peers),
filters non-singletons, deterministically samples ~N clusters by
``int(component_id) % SAMPLE_MOD``, and writes a small cluster-index
parquet.

cc/it_10 has 4,096 shards (vs the 100,810 shard pairs we'd need for the
attr + normalized join), so the fan-out is ~25x smaller. Singleton rows
have ``len(adjacency_list) == 1`` (the lone element is the node itself);
we drop them so the index only carries real duplicate clusters.

``component_id == id_norm`` is the natural canonical (CC's hash-to-min
keeps the min id_norm), so step 2 can derive ``is_canonical`` without
extra joins.

Output (parquet under OUTPUT_PATH): one row per sampled cluster member.

    component_id: string
    record_id:    string   # "source_NNN|<doc_id>"
    id_norm:      string
    file_idx:     int64    # global shard index across the dedup input

Pair with :mod:`fetch_cluster_texts` (step 2) to join in document text.

Submit on iris (us-central2 pinned by the worker's MARIN_PREFIX):

    uv run iris --cluster=marin job run --region us-central2 \\
        --priority interactive \\
        -- python experiments/datakit/dedup/ops/sample_clusters.py
"""

import logging

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)


DEDUP_ROOT = "gs://marin-us-central2/datakit/dedup_dabe67c2"
CC_GLOB = f"{DEDUP_ROOT}/metadata/cc/it_10/*.parquet"
OUTPUT_PATH = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/cluster_index"

# 1.42B unique clusters in this run → MOD=1_420_000 ⇒ ~1000 sampled clusters;
# avg 3.36 members/cluster ⇒ ~3,360 emitted rows.
SAMPLE_MOD = 1_420_000

WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3.5g", preemptible=False)
MAX_WORKERS = 256
NUM_OUTPUT_SHARDS = 4


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("component_id", pa.string()),
        pa.field("record_id", pa.string()),
        pa.field("id_norm", pa.string()),
        pa.field("file_idx", pa.int64()),
    ]
)


def _keep(r: dict) -> bool:
    adj = r.get("adjacency_list") or ()
    if len(adj) <= 1:
        return False
    try:
        return int(r["component_id"]) % SAMPLE_MOD == 0
    except (TypeError, ValueError):
        return False


def _emit(r: dict) -> dict:
    return {
        "component_id": r["component_id"],
        "record_id": r["record_id"],
        "id_norm": r["id_norm"],
        "file_idx": int(r["file_idx"]),
    }


def _output_path(shard_idx: int, total_shards: int) -> str:
    return f"{OUTPUT_PATH}/part-{shard_idx:05d}-of-{total_shards:05d}.parquet"


def main() -> None:
    configure_logging(logging.INFO)
    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=MAX_WORKERS,
        name="sample-cluster-index",
    )
    pipeline = (
        Dataset.from_files(CC_GLOB)
        .load_parquet(columns=["record_id", "id_norm", "adjacency_list", "component_id", "file_idx"])
        .filter(_keep)
        .map(_emit)
        .write_parquet(_output_path, schema=OUTPUT_SCHEMA, skip_existing=True)
    )
    outcome = ctx.execute(pipeline, verbose=True)
    logger.info("done; counters: %s", dict(outcome.counters))
    logger.info("cluster index written to: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
