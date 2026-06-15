# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantify val-set contamination against `4plus` ONLY (vs the union decon).

The verified_pairs store full (val_id, other_id, jaccard) triples — not just the
per-doc max — so we can recover exactly which val docs are contaminated w.r.t.
each source. p33m67 trains `4plus` only, but the shipped decon cutoff thresholds
on max over (3, 4plus, 4plus_mind). This reads the `4plus` verified_pairs, derives
the per-val-doc max-Jaccard-to-`4plus`, and contrasts contamination counts vs the
union inside the j090 paranoid universe — i.e. how over-decontaminated the sweep
is for a 4plus-only run (gotcha #1).
"""

import json
import logging
from collections import defaultdict

import fsspec
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

SCRATCH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
FOURP_PAIRS = f"{SCRATCH}/4plus_284x71/verified_pairs"
UNIVERSE = f"{SCRATCH}/decon_val_sets/keep_ids/keep_ids_j090.json"
OUT_MAX = f"{SCRATCH}/decon_val_sweep/fourplus_only_max_by_id.json"
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]


def fourplus_max_by_val_id() -> tuple[dict[str, float], int]:
    """Per-val-doc max exact Jaccard to any `4plus` doc, from the full pair list."""
    fourp_max: dict[str, float] = defaultdict(float)
    pairs = 0
    shards = sorted(fsspec_glob(f"{FOURP_PAIRS}/*.parquet"))
    for i, path in enumerate(shards):
        with fsspec.open(path, "rb") as f:
            table = pq.read_table(f, columns=["val_id", "jaccard"])
        if i == 0:
            with fsspec.open(path, "rb") as f:
                logger.info("verified_pairs schema: %s", pq.read_schema(f).names)
        for val_id, jac in zip(table.column("val_id").to_pylist(), table.column("jaccard").to_pylist(), strict=True):
            pairs += 1
            if jac > fourp_max[val_id]:
                fourp_max[val_id] = jac
    return dict(fourp_max), pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fourp_max, pairs = fourplus_max_by_val_id()
    logger.info("%d (val,4plus) verified pairs; %d distinct val docs with a >=0.5 4plus twin", pairs, len(fourp_max))

    with fsspec.open(UNIVERSE) as f:
        u = json.load(f)
    ids = u["ids"]
    union_by_docidx = {int(k): float(v) for k, v in u["max_jaccard_by_doc"].items()}
    union_max = {vid: union_by_docidx.get(int(d), 0.0) for vid, d in zip(ids, u["doc_indices"], strict=True)}
    n = len(ids)
    logger.info("j090 paranoid universe: %d val docs (all have union-max < 0.90 by construction)", n)

    logger.info("\n  tau | union>=t  4plus>=t  union-only  ||  union-keep  4plus-keep  recovered")
    logger.info("  ----+-------------------------------------++--------------------------------")
    for t in THRESHOLDS:
        union_c = sum(union_max[v] >= t for v in ids)
        fourp_c = sum(fourp_max.get(v, 0.0) >= t for v in ids)
        union_only = sum(union_max[v] >= t and fourp_max.get(v, 0.0) < t for v in ids)
        logger.info(
            "  %.2f | %7d  %7d  %9d  ||  %8d  %9d  %+8d",
            t,
            union_c,
            fourp_c,
            union_only,
            n - union_c,
            n - fourp_c,
            (n - fourp_c) - (n - union_c),
        )

    with fsspec.open(OUT_MAX, "w") as f:
        json.dump(fourp_max, f)
    logger.info("\nwrote per-id 4plus-only max -> %s", OUT_MAX)


if __name__ == "__main__":
    main()
