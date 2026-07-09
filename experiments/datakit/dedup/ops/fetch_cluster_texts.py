# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2/2: join the cluster index with normalized text and emit per-cluster examples.

Reads the ``(component_id, record_id, id_norm, file_idx)`` index produced
by :mod:`sample_clusters` (step 1), reconstructs the dedup-time
``file_idx → (source_main_dir, basename)`` mapping, fetches text for every
sampled doc (streaming reads, capped concurrency), and writes per-cluster
member lists.

No zephyr — the work is tiny (~thousands of records across ~hundreds of
shards), and a single Python process with a thread pool is more robust
than the distributed pipeline we were fighting earlier.

Output (single parquet at OUTPUT_PATH):

    cluster_id:    string
    num_members:   int32
    members:       list<struct<source: string, id: string, is_canonical: bool,
                               chars: int32, text: string>>

Submit on iris (us-central2 pinned by the worker's MARIN_PREFIX):

    uv run iris --cluster=marin job run --region us-central2 \\
        --priority interactive \\
        -- python experiments/datakit/dedup/ops/fetch_cluster_texts.py
"""

import concurrent.futures
import json
import logging
import os
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


DEDUP_ROOT = "gs://marin-us-central2/datakit/dedup_dabe67c2"
INDEX_DIR = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/cluster_index"
OUTPUT_PATH = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/examples/part-00000.parquet"

MAX_MEMBERS_PER_CLUSTER = 10
MAX_TEXT_CHARS = 20_000
# Concurrency * per-shard streaming batch keeps peak memory bounded so this
# fits in the default 1 GB iris launcher: 64 threads * 1000-row batches * ~10 KB
# avg text ≈ 0.6 GB worst case.
READ_CONCURRENCY = 64
READ_BATCH_ROWS = 1000


_NORMALIZED_RE = re.compile(r"normalized/(.+)/outputs/main/?$")


def _source_label(norm_dir: str) -> str:
    m = _NORMALIZED_RE.search(norm_dir.rstrip("/"))
    return m.group(1) if m else norm_dir


def _build_file_idx_map() -> dict[int, dict]:
    """Reconstruct the dedup-time ``file_idx → {source_main_dir, basename}`` mapping.

    Mirrors ``fuzzy_dups._build_shard_index``: iterate the manifest's
    ``sources`` dict in insertion order (preserved through JSON), and for each
    source concatenate its lexically-sorted attr_dir parquets. ``file_idx`` is
    the global position. The normalized parquet path is
    ``{source_main_dir}/{basename}`` (attr and normalized are co-partitioned
    with matching basenames). The 114 directory listings are done in
    parallel; the global ordering is reconstructed from the manifest's
    insertion order afterwards.
    """
    art = json.loads(StoragePath(f"{DEDUP_ROOT}/.artifact.json").read_bytes())
    sources: dict[str, dict] = art["sources"]
    norm_dirs = list(sources.keys())  # insertion order = dedup caller order

    def _list_one(norm_dir: str) -> list[str]:
        attr_dir = sources[norm_dir]["attr_dir"]
        return sorted(str(p) for p in StoragePath(attr_dir).ls() if str(p).endswith(".parquet"))

    files_per_source: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as ex:
        for norm_dir, files in zip(norm_dirs, ex.map(_list_one, norm_dirs), strict=True):
            files_per_source[norm_dir] = files

    mapping: dict[int, dict] = {}
    fi = 0
    for norm_dir in norm_dirs:
        label = _source_label(norm_dir)
        norm_root = norm_dir.rstrip("/")
        for f in files_per_source[norm_dir]:
            basename = os.path.basename(f)
            mapping[fi] = {
                "norm_path": f"{norm_root}/{basename}",
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as ex:
        for part in ex.map(_read_one, files):
            rows.extend(part)
    logger.info("loaded %d cluster-member rows from %d index shards", len(rows), len(files))
    return rows


def _split_record_id(record_id: str) -> tuple[str, str]:
    src, _, doc = record_id.partition("|")
    return src, doc


def _fetch_texts_for_file(file_idx: int, norm_path: str, wanted_ids: set[str]) -> dict[str, str]:
    """Stream the normalized parquet, picking out only the wanted ids."""
    out: dict[str, str] = {}
    target = len(wanted_ids)
    with StoragePath(norm_path).open("rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=READ_BATCH_ROWS):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for did, t in zip(ids, texts, strict=True):
                if did in wanted_ids:
                    out[did] = t or ""
                    if len(out) == target:
                        return out
    return out


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


def main() -> None:
    configure_logging(logging.INFO)

    rows = _read_index()
    fi_map = _build_file_idx_map()

    # Group wanted ids per normalized shard so each shard is read once.
    wanted_by_file: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        _, doc_id = _split_record_id(r["record_id"])
        wanted_by_file[r["file_idx"]].add(doc_id)
    logger.info("fetching text from %d unique normalized shards", len(wanted_by_file))

    texts: dict[int, dict[str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as ex:
        future_to_fi = {
            ex.submit(_fetch_texts_for_file, fi, fi_map[fi]["norm_path"], wanted_by_file[fi]): fi
            for fi in wanted_by_file
        }
        for fut in concurrent.futures.as_completed(future_to_fi):
            fi = future_to_fi[fut]
            texts[fi] = fut.result()
    total_fetched = sum(len(v) for v in texts.values())
    logger.info("fetched %d texts (target %d)", total_fetched, sum(len(s) for s in wanted_by_file.values()))

    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        _, doc_id = _split_record_id(r["record_id"])
        full = texts.get(r["file_idx"], {}).get(doc_id, "")
        clusters[r["component_id"]].append(
            {
                "source": fi_map[r["file_idx"]]["source_label"],
                "id": doc_id,
                "is_canonical": r["component_id"] == r["id_norm"],
                "chars": len(full),
                "text": full[:MAX_TEXT_CHARS],
            }
        )

    records: list[dict] = []
    for cid, members in clusters.items():
        # Stable order: canonicals first, then the rest in arrival order; cap.
        members.sort(key=lambda m: (not m["is_canonical"],))
        records.append(
            {
                "cluster_id": cid,
                "num_members": len(members),
                "members": members[:MAX_MEMBERS_PER_CLUSTER],
            }
        )
    logger.info("assembled %d cluster examples", len(records))

    table = pa.Table.from_pylist(records, schema=_OUTPUT_SCHEMA)
    output_path = StoragePath(OUTPUT_PATH)
    parent = output_path.parent
    if parent.key:
        parent.mkdirs()
    with output_path.open("wb") as fh:
        pq.write_table(table, fh)
    logger.info("wrote %s (%d clusters)", OUTPUT_PATH, len(records))


if __name__ == "__main__":
    main()
