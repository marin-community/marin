# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify LSH candidate pairs and quantify nemotron math val contamination.

Inputs (produced upstream):
- gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_candidate_pairs/  (val_id, other_id)
- gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs/             (shard, row, id, text)
- scratch/nemotron_math_val_window_indices.npy
- scratch/nemotron_math_val_doc_indices.npy
- scratch/nemotron_math_doc_offsets.npy

For every candidate pair, computes exact Jaccard over 5-char shingles
(same normalization as dupekit: lowercase, collapse whitespace) between val
text and the other doc's text (fetched from the normalized parquet via
global-sorted xxh3 ids).

Outputs:
- per-pair verified Jaccard parquet (scratch/val_pair_jaccard.parquet)
- per-window contamination summary (scratch/val_window_contamination.json)
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gcsfs
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SCRATCH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
NORM = "marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
SEQ_LEN = 4096
JACCARD_THRESHOLD = 0.75
_WS = re.compile(r"\s+")


def shingles(text: str, n: int = 5) -> set[str]:
    cleaned = _WS.sub(" ", text.lower()).strip()
    return {cleaned[i : i + n] for i in range(max(1, len(cleaned) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fs = gcsfs.GCSFileSystem()

    # Candidate pairs.
    pair_files = fs.glob(f"{SCRATCH.removeprefix('gs://')}/val_candidate_pairs/*.parquet")
    pairs = []
    for path in pair_files:
        t = pq.read_table(fs.open(path))
        pairs.extend(zip(t.column("val_id").to_pylist(), t.column("other_id").to_pylist(), strict=True))
    pairs = sorted(set(pairs))
    logger.info("candidate pairs: %d", len(pairs))

    # Val texts.
    val_text: dict[str, str] = {}
    for path in fs.glob(f"{SCRATCH.removeprefix('gs://')}/val_docs/*.parquet"):
        t = pq.read_table(fs.open(path), columns=["id", "text"])
        val_text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))

    # Map other_id -> (shard, row) via the local all-ids dump. Shards are
    # individually sorted by id but NOT globally range-partitioned, so a
    # global doc index is required (scratch/nemotron_math_all_ids.npy).
    shard_paths = sorted(fs.glob(f"{NORM}/*.parquet"))
    all_ids = np.load("scratch/nemotron_math_all_ids.npy")  # (45M, 2) uint64
    order = np.lexsort((all_ids[:, 1], all_ids[:, 0]))
    sorted_ids = all_ids[order]

    shard_rows = json.loads(Path("scratch/nemotron_math_shard_rows.json").read_text())
    cum = np.concatenate([[0], np.cumsum([shard_rows[s] for s in sorted(shard_rows)])])

    others = sorted({other for _, other in pairs} - set(val_text))
    others_arr = np.array(
        [[int(oid[:16], 16), int(oid[16:], 16)] for oid in others],
        dtype=np.uint64,
    )
    pos = np.searchsorted(sorted_ids[:, 0], others_arr[:, 0], "left")
    by_shard: dict[int, list[str]] = {}
    for oid, p in zip(others, pos, strict=True):
        target = (int(oid[:16], 16), int(oid[16:], 16))
        idx = None
        while p < len(sorted_ids) and int(sorted_ids[p, 0]) == target[0]:
            if int(sorted_ids[p, 1]) == target[1]:
                idx = int(order[p])
                break
            p += 1
        if idx is None:
            continue
        shard_idx = int(np.searchsorted(cum, idx, "right") - 1)
        by_shard.setdefault(shard_idx, []).append(oid)

    def fetch(item: tuple[int, list[str]]) -> dict[str, str]:
        shard_idx, ids = item
        t = pq.read_table(fs.open(shard_paths[shard_idx]), columns=["id", "text"])
        idx = {i: j for j, i in enumerate(t.column("id").to_pylist())}
        return {i: t.column("text")[idx[i]].as_py() for i in ids if i in idx}

    other_text: dict[str, str] = {}
    with ThreadPoolExecutor(16) as ex:
        for chunk in ex.map(fetch, by_shard.items()):
            other_text.update(chunk)
    logger.info("fetched %d candidate texts", len(other_text))

    # Verify pairs.
    val_shingles = {vid: shingles(val_text[vid]) for vid in {v for v, _ in pairs} if vid in val_text}
    results = []
    for vid, oid in pairs:
        other = other_text.get(oid) or val_text.get(oid)
        if other is None or vid not in val_shingles:
            continue
        results.append({"val_id": vid, "other_id": oid, "jaccard": jaccard(val_shingles[vid], shingles(other))})

    contaminated = {r["val_id"] for r in results if r["jaccard"] >= JACCARD_THRESHOLD}
    logger.info("verified pairs: %d, contaminated val docs: %d", len(results), len(contaminated))

    # Window-level contamination.
    val_windows = np.load("scratch/nemotron_math_val_window_indices.npy")
    offsets = np.load("scratch/nemotron_math_doc_offsets.npy")
    val_doc_indices = np.load("scratch/nemotron_math_val_doc_indices.npy")

    # Recover id per val doc from extracted val_docs (shard,row → global index → id).
    with open("scratch/val_doc_manifest.json") as f:
        manifest = json.load(f)
    shard_names = sorted(manifest)
    shard_offsets = {}
    pos = 0
    for name, rows in ((s, manifest[s]) for s in shard_names):
        shard_offsets[name] = pos
        pos += len(rows)
    doc_id_order = []
    for s in shard_names:
        for r in sorted(manifest[s]):
            doc_id_order.append((s, r))
    id_by_docindex = {}
    for path in fs.glob(f"{SCRATCH.removeprefix('gs://')}/val_docs/*.parquet"):
        t = pq.read_table(fs.open(path), columns=["shard", "row", "id"])
        for shard, row, doc_id in zip(*(t.column(c).to_pylist() for c in ("shard", "row", "id")), strict=True):
            id_by_docindex[(shard, row)] = doc_id
    docindex_to_id = {val_doc_indices[i]: id_by_docindex[pair] for i, pair in enumerate(doc_id_order)}

    starts = val_windows * SEQ_LEN
    ends = starts + SEQ_LEN
    doc_lo = np.searchsorted(offsets, starts, "right")
    doc_hi = np.searchsorted(offsets, ends, "left")
    n_contam = 0
    for lo, hi in zip(doc_lo, doc_hi, strict=True):
        ids = {docindex_to_id.get(d) for d in range(lo, hi + 1)}
        if ids & contaminated:
            n_contam += 1
    summary = {
        "candidate_pairs": len(pairs),
        "verified_pairs": len(results),
        "contaminated_val_docs": len(contaminated),
        "total_val_docs": len(val_doc_indices),
        "contaminated_val_windows": n_contam,
        "total_val_windows": len(val_windows),
        "jaccard_threshold": JACCARD_THRESHOLD,
    }
    with open("scratch/val_window_contamination.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
