# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proper fuzzy-dedup a 10B-token `3` slice against all of `4plus`.

Standard marin/datakit fuzzy deduplication: MinHash at the DEFAULT 286x26 banding
(r=11, ~0.8 Jaccard threshold — a shared band-bucket already means a genuine
near-duplicate, so no candidate/verify step), then connected components across
both corpora. A `3` doc is a duplicate of the trained corpus iff its CC cluster
contains a `4plus` doc; those get dropped in stage 2.

(The earlier attempts wrongly reused the 284x71 minhash, which is the high-recall
*scan* banding — every math doc collides with ~35k others there, so it only works
with a per-pair verify. 286x26 is the dedup banding; no verify.)

The 286x26 `4plus` minhash already exists (the original contamination scan, 45M
docs). Only the `3` slice is signed here.

Stages:
1. Server-side copy the first N `3` normalized shards into the slice text dir
   (materializes the fixed 10B sample).
2. MinHash that slice at 286x26.
3. `compute_fuzzy_dups_attrs([3-slice, 4plus])` -> per-doc cluster markers.

Launch (us-east5, all preemptible):

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 8 --memory 64GB --disk 50GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name three-vs-fourplus-dedup \
        -- python scripts/analysis/crossdedup_three_vs_fourplus.py --shards 48
"""

import argparse
import json
import logging

import fsspec
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    MinHashParams,
    compute_minhash_attrs,
)
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

THREE_NORM = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/3_f8007d22/outputs/main"
FOURP_NORM = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
FOURP_MINHASH_286 = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/minhash/outputs"  # canonical 286x26, 45M docs
SLICE_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b"
SLICE_DOCS = f"{SLICE_ROOT}/docs"
SLICE_MINHASH = f"{SLICE_ROOT}/minhash_286x26"
OUT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_vs_fourplus_dedup"
PARAMS = MinHashParams(num_perms=286, num_bands=26, ngram_size=5, seed=42)


def stage_slice_docs(num_shards: int) -> str:
    """Server-side copy the first N `3` normalized shards into the slice docs dir."""
    three = sorted(fsspec_glob(f"{THREE_NORM}/*.parquet"))[:num_shards]
    if len(three) < num_shards:
        raise ValueError(f"only {len(three)} `3` shards available, need {num_shards}")
    fs = fsspec.filesystem("gcs")
    have = {p.rsplit("/", 1)[1] for p in fsspec_glob(f"{SLICE_DOCS}/*.parquet")}
    staged = 0
    for src in three:
        base = src.rsplit("/", 1)[1]
        if base not in have:
            fs.copy(src, f"{SLICE_DOCS}/{base}")
            staged += 1
    logger.info("staged %d new (of %d) `3`-slice doc shards in %s", staged, len(three), SLICE_DOCS)
    return SLICE_DOCS


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=48, help="First N `3` shards (~48 = 10B tokens).")
    parser.add_argument("--max-parallelism", type=int, default=512)
    args = parser.parse_args()

    slice_docs = stage_slice_docs(args.shards)

    slice_mh = compute_minhash_attrs(
        source=NormalizedData(main_output_dir=slice_docs, dup_output_dir="", counters={}),
        output_path=SLICE_MINHASH,
        num_perms=PARAMS.num_perms,
        num_bands=PARAMS.num_bands,
        ngram_size=PARAMS.ngram_size,
        seed=PARAMS.seed,
        worker_resources=ResourceConfig(cpu=4, ram="24g", disk="5g", preemptible=True, regions=("us-east5",)),
        max_workers=args.shards,
    )
    fourp_mh = MinHashAttrData(params=PARAMS, source_main_dir=FOURP_NORM, attr_dir=FOURP_MINHASH_286, counters={})

    logger.info(
        "fuzzy dedup `3`-slice (%d shards) + `4plus` (%d shards) at 286x26 -> %s",
        args.shards,
        len(fsspec_glob(f"{FOURP_MINHASH_286}/*.parquet")),
        OUT,
    )
    result = compute_fuzzy_dups_attrs(
        inputs=[slice_mh, fourp_mh],
        output_path=OUT,
        cc_max_iterations=15,
        cc_resume=True,
        max_parallelism=args.max_parallelism,
        worker_resources=ResourceConfig(cpu=2, ram="32g", disk="10g", preemptible=True, regions=("us-east5",)),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g", disk="20g", preemptible=True, regions=("us-east5",)),
    )
    manifest = {
        "three_slice_shards": args.shards,
        "slice_docs": SLICE_DOCS,
        "banding": "286x26 (r=11, ~0.8 Jaccard — canonical dedup)",
        "output": OUT,
        "sources": {k: v.attr_dir for k, v in result.sources.items()},
        "cluster_members": result.counters.get("dedup/fuzzy/document/cluster_members"),
    }
    with fsspec.open(f"{SLICE_ROOT}/dedup_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("done: %s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
