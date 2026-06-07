# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Union the verified near-dup drop list and count the surviving clean val set.

Runs on iris near the data (us-east5). Reads verified pairs of every finished
subset scan plus replay artifacts (45M id dump, doc offsets, val windows), then
reports clean docs and fully clean windows/tokens at several Jaccard cutoffs.

Launch:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 4 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name decon-val-summary \
        -- python scripts/analysis/decon_val_summary.py
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

SCRATCH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
SUBSETS = ["4plus_284x71", "3_284x71", "4plus_mind_284x71"]
REPLAY = f"{SCRATCH}/replay"
OUT = f"{SCRATCH}/decon_val_summary.json"
CUTOFFS = [0.5, 0.7, 0.75, 0.8, 0.9]
SEQ_LEN = 4096
VAL_DOCS_TOTAL = 57243


def _read_pairs(path: str) -> list[tuple[str, float]]:
    t = pq.read_table(path, columns=["val_id", "jaccard"])
    return list(zip(t.column("val_id").to_pylist(), t.column("jaccard").to_pylist(), strict=True))


def _load_npy(path: str) -> np.ndarray:
    with fsspec.open(path, "rb") as f:
        return np.load(f)


def _overlap_tokens_for_docs(
    doc_indices: set[int], offsets: np.ndarray, sorted_window_starts: np.ndarray, sorted_window_ends: np.ndarray
) -> int:
    dropped_tokens = 0
    for d in doc_indices:
        doc_start = int(offsets[d])
        doc_end = int(offsets[d + 1])
        window_index = int(np.searchsorted(sorted_window_ends, doc_start, "right"))
        while window_index < len(sorted_window_starts) and int(sorted_window_starts[window_index]) < doc_end:
            dropped_tokens += min(doc_end, int(sorted_window_ends[window_index])) - max(
                doc_start, int(sorted_window_starts[window_index])
            )
            window_index += 1
    return dropped_tokens


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    files = [f for sub in SUBSETS for f in fsspec_glob(f"{SCRATCH}/{sub}/verified_pairs/*.parquet")]
    logger.info("verified pair files: %d (missing subsets are skipped)", len(files))

    best: dict[str, float] = {}
    with ThreadPoolExecutor(32) as ex:
        for chunk in ex.map(_read_pairs, files):
            for val_id, j in chunk:
                if j > best.get(val_id, 0.0):
                    best[val_id] = j

    all_ids = _load_npy(f"{REPLAY}/nemotron_math_all_ids.npy")
    offsets = _load_npy(f"{REPLAY}/nemotron_math_doc_offsets.npy")
    windows = _load_npy(f"{REPLAY}/nemotron_math_val_window_indices.npy")
    order = np.lexsort((all_ids[:, 1], all_ids[:, 0]))
    sorted_ids = all_ids[order]

    ids = list(best)
    arr = np.array([[int(i[:16], 16), int(i[16:], 16)] for i in ids], dtype=np.uint64)
    pos = np.searchsorted(sorted_ids[:, 0], arr[:, 0], "left")
    doc_of: dict[str, int] = {}
    for i, hi, lo, p in zip(ids, arr[:, 0], arr[:, 1], pos, strict=True):
        while p < len(sorted_ids) and sorted_ids[p, 0] == hi:
            if sorted_ids[p, 1] == lo:
                doc_of[i] = int(order[p])
                break
            p += 1
    logger.info("mapped %d/%d drop ids to doc indices", len(doc_of), len(ids))

    # Doc-level dedupe: drop contaminated docs, keep every other val token
    # (token-exact window∩doc overlap; no whole-window pruning). Only
    # contaminated docs matter for the dropped-token count, so avoid scanning
    # every validation-window/doc span.
    window_starts = np.sort(windows * SEQ_LEN)
    window_ends = window_starts + SEQ_LEN

    summary = {"subsets": SUBSETS, "pair_files": len(files), "val_docs_total": VAL_DOCS_TOTAL, "cutoffs": {}}
    for cut in CUTOFFS:
        drop_docs = {doc_of[i] for i, j in best.items() if j >= cut and i in doc_of}
        dropped_tokens = _overlap_tokens_for_docs(drop_docs, offsets, window_starts, window_ends)
        total = len(windows) * SEQ_LEN
        summary["cutoffs"][str(cut)] = {
            "drop_docs": len(drop_docs),
            "clean_docs": VAL_DOCS_TOTAL - len(drop_docs),
            "clean_tokens": int(total - dropped_tokens),
            "dropped_tokens": int(dropped_tokens),
        }
        logger.info("J>=%.2f: drop=%d clean_tokens=%d", cut, len(drop_docs), total - dropped_tokens)

    with fsspec.open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
