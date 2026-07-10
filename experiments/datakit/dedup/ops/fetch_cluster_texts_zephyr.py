# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 (zephyr variant): fetch cluster member texts in parallel via iris.

Mirrors :mod:`fetch_cluster_texts` but instead of a single Python process with
a thread pool, fans the per-shard text fetches across a zephyr pipeline. The
launcher prepares the per-``file_idx`` task list (group sampled ids by which
normalized shard they live in), then workers stream that shard, pull out the
wanted ids, and emit one record per matched member. A ``group_by(component_id)``
assembles per-cluster member lists.

Per-shard zephyr counters give live progress visibility:
``fetch/shards_done``, ``fetch/members_found``, ``fetch/members_missing``.

Output: parquet under ``OUTPUT_PATH`` (distinct from the local-script path so
the two can race without overwriting each other).

Submit on iris (us-central2 pinned by the worker's MARIN_PREFIX):

    uv run iris --cluster=marin job run --region us-central2 \\
        --priority interactive \\
        -- python experiments/datakit/dedup/ops/fetch_cluster_texts_zephyr.py
"""

import concurrent.futures
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)


DEDUP_ROOT = "gs://marin-us-central2/datakit/dedup_dabe67c2"
INDEX_DIR = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/cluster_index"
OUTPUT_PATH = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/examples_zephyr"

MAX_MEMBERS_PER_CLUSTER = 10
MAX_TEXT_CHARS = 20_000

LIST_CONCURRENCY = 64
READ_BATCH_ROWS = 1000

WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="5g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3.5g", preemptible=False)
MAX_WORKERS = 256
NUM_OUTPUT_SHARDS = 4


_NORMALIZED_RE = re.compile(r"normalized/(.+)/outputs/main/?$")


def _source_label(norm_dir: str) -> str:
    m = _NORMALIZED_RE.search(norm_dir.rstrip("/"))
    return m.group(1) if m else norm_dir


def _build_file_idx_map() -> dict[int, dict]:
    """Reconstruct the dedup-time ``file_idx → {source_main_dir, basename}`` mapping.

    Critically, dedup's ``_build_shard_index`` appends entries in *caller* order
    (the order ``inputs`` was passed to ``compute_fuzzy_dups_attrs``), not in the
    sorted source_main_dir order that the persisted manifest's ``sources`` dict
    is keyed by. The two differ in this run.

    The caller order is preserved on disk in the dedup step's ``.executor_info``
    ``dependencies`` list — those are the minhash step paths in the exact order
    passed to dedup. We read each minhash artifact to recover its
    ``source_main_dir``, list its attr_dir parquets (sorted), and concatenate
    in caller order: ``file_idx`` is the global position. Basenames match
    between minhash, dedup attr, and normalized (all co-partitioned).
    """
    info = json.loads(StoragePath(f"{DEDUP_ROOT}/.executor_info").read_bytes())
    minhash_deps = [d for d in info["dependencies"] if "/datakit/minhash/" in d]
    logger.info("dedup was invoked with %d minhash inputs (caller order)", len(minhash_deps))

    def _resolve(minhash_dir: str) -> tuple[str, list[str]]:
        art = json.loads(StoragePath(f"{minhash_dir}/.artifact.json").read_bytes())
        source_main_dir = art["source_main_dir"]
        minhash_attr = art["attr_dir"]
        files = sorted(str(p) for p in StoragePath(minhash_attr).ls() if str(p).endswith(".parquet"))
        return source_main_dir, files

    resolved: list[tuple[str, list[str]]] = [(None, None)] * len(minhash_deps)  # type: ignore[list-item]
    with concurrent.futures.ThreadPoolExecutor(max_workers=LIST_CONCURRENCY) as ex:
        for i, r in enumerate(ex.map(_resolve, minhash_deps)):
            resolved[i] = r

    mapping: dict[int, dict] = {}
    fi = 0
    for source_main_dir, files in resolved:
        label = _source_label(source_main_dir)
        norm_root = source_main_dir.rstrip("/")
        for f in files:
            mapping[fi] = {
                "norm_path": f"{norm_root}/{os.path.basename(f)}",
                "source_label": label,
            }
            fi += 1
    logger.info("rebuilt file_idx map across %d global shards", fi)
    return mapping


def _read_index() -> list[dict]:
    files = sorted((p for p in StoragePath(INDEX_DIR).ls() if str(p).endswith(".parquet")), key=str)
    if not files:
        raise FileNotFoundError(f"no parquet files under {INDEX_DIR}")

    def _read_one(f: StoragePath) -> list[dict]:
        with f.open("rb") as fh:
            t = pq.read_table(fh)
        out: list[dict] = []
        cids = t.column("component_id").to_pylist()
        rids = t.column("record_id").to_pylist()
        idns = t.column("id_norm").to_pylist()
        fidx = t.column("file_idx").to_pylist()
        for c, r, i, fx in zip(cids, rids, idns, fidx, strict=True):
            out.append({"component_id": c, "record_id": r, "id_norm": i, "file_idx": int(fx)})
        return out

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=LIST_CONCURRENCY) as ex:
        for part in ex.map(_read_one, files):
            rows.extend(part)
    logger.info("loaded %d cluster-member rows from %d index shards", len(rows), len(files))
    return rows


def _split_record_id(record_id: str) -> tuple[str, str]:
    src, _, doc = record_id.partition("|")
    return src, doc


def _build_tasks(rows: list[dict], fi_map: dict[int, dict]) -> list[dict]:
    """Group sampled cluster-member rows by file_idx so each task reads one norm shard."""
    by_fi: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        _, doc_id = _split_record_id(r["record_id"])
        by_fi[r["file_idx"]].append(
            {
                "doc_id": doc_id,
                "component_id": r["component_id"],
                "id_norm": r["id_norm"],
            }
        )
    tasks: list[dict] = []
    for fi, members in by_fi.items():
        info = fi_map[fi]
        tasks.append(
            {
                "file_idx": fi,
                "norm_path": info["norm_path"],
                "source_label": info["source_label"],
                "members": members,
            }
        )
    return tasks


def _fetch_one(task: dict) -> Iterator[dict]:
    """Per-shard worker: stream the normalized parquet, emit one record per wanted member."""
    wanted_ids = {m["doc_id"]: m for m in task["members"]}
    found = 0
    target = len(wanted_ids)
    with StoragePath(task["norm_path"]).open("rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=READ_BATCH_ROWS):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for did, t in zip(ids, texts, strict=True):
                m = wanted_ids.get(did)
                if m is None:
                    continue
                full = t or ""
                yield {
                    "component_id": m["component_id"],
                    "source": task["source_label"],
                    "id": did,
                    "is_canonical": m["component_id"] == m["id_norm"],
                    "chars": len(full),
                    "text": full[:MAX_TEXT_CHARS],
                }
                found += 1
                if found == target:
                    break
            if found == target:
                break
    counters.pipeline.update_counter("fetch/shards_done", 1)
    counters.pipeline.update_counter("fetch/members_found", found)
    if found < target:
        counters.pipeline.update_counter("fetch/members_missing", target - found)


def _collect_members(cluster_id: str, items: Iterator[dict]) -> dict:
    members: list[dict] = []
    total = 0
    for it in items:
        total += 1
        if len(members) < MAX_MEMBERS_PER_CLUSTER:
            members.append(
                {
                    "source": it["source"],
                    "id": it["id"],
                    "is_canonical": it["is_canonical"],
                    "chars": it["chars"],
                    "text": it["text"],
                }
            )
    members.sort(key=lambda m: (not m["is_canonical"],))
    return {
        "cluster_id": cluster_id,
        "num_members": total,
        "members": members[:MAX_MEMBERS_PER_CLUSTER],
    }


_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.string()),
        pa.field("num_members", pa.int32()),
        pa.field(
            "members",
            pa.list_(
                pa.struct(
                    [
                        pa.field("source", pa.string()),
                        pa.field("id", pa.string()),
                        pa.field("is_canonical", pa.bool_()),
                        pa.field("chars", pa.int32()),
                        pa.field("text", pa.string()),
                    ]
                )
            ),
        ),
    ]
)


def _output_path(shard_idx: int, total_shards: int) -> str:
    return f"{OUTPUT_PATH}/part-{shard_idx:05d}-of-{total_shards:05d}.parquet"


def main() -> None:
    configure_logging(logging.INFO)
    rows = _read_index()
    fi_map = _build_file_idx_map()
    tasks = _build_tasks(rows, fi_map)
    logger.info("built %d fetch tasks (one per normalized shard)", len(tasks))

    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(MAX_WORKERS, len(tasks)),
        name="fetch-cluster-texts",
    )

    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(_fetch_one)
        .group_by(
            key=lambda r: r["component_id"],
            reducer=_collect_members,
            num_output_shards=NUM_OUTPUT_SHARDS,
        )
        .write_parquet(_output_path, schema=_OUTPUT_SCHEMA, skip_existing=True)
    )

    outcome = ctx.execute(pipeline, verbose=True)
    logger.info("done; counters: %s", dict(outcome.counters))
    logger.info("output: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
