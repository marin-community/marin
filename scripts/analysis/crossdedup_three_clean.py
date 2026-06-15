# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2: label `3`-slice docs that are duplicates of `4plus` (cluster touches trained corpus).

Reads the fuzzy-dedup cluster markers from crossdedup_three_vs_fourplus.py:
source_000 = `4plus`, source_001 = `3`-slice. A `3` doc is a duplicate of the
trained corpus iff its `dup_cluster_id` also appears among `4plus` docs. Writes
the contaminated `3` id list + a summary (counts + clean-token estimate). Runs
in-region (just reads the small marker columns).

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 8 --memory 64GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name three-clean-label \
        -- python scripts/analysis/crossdedup_three_clean.py
"""

import argparse
import json
import logging

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

DEDUP_OUT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_vs_fourplus_dedup/outputs"
SLICE_DOCS = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b/docs"
SLICE_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b"
CONTAMINATED_IDS = f"{SLICE_ROOT}/contaminated_ids_286"
TOK_PER_DOC = 1487.9  # `3` calibrated tokens/doc
ID_SCHEMA = pa.schema([("three_id", pa.string())])


def fourplus_cluster_ids() -> set[str]:
    """Distinct dup_cluster_ids that contain a `4plus` doc (source_000)."""
    clusters: set[str] = set()
    for path in sorted(fsspec_glob(f"{DEDUP_OUT}/source_000/*.parquet")):
        with fsspec.open(path, "rb") as f:
            attrs = pq.read_table(f, columns=["attributes"]).column("attributes").to_pylist()
        clusters.update(a["dup_cluster_id"] for a in attrs)
    return clusters


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser().parse_args()

    s4 = fourplus_cluster_ids()
    logger.info("clusters containing a 4plus doc: %d", len(s4))

    contaminated: set[str] = set()
    for path in sorted(fsspec_glob(f"{DEDUP_OUT}/source_001/*.parquet")):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["id", "attributes"])
        for doc_id, attr in zip(t.column("id").to_pylist(), t.column("attributes").to_pylist(), strict=True):
            if attr["dup_cluster_id"] in s4:
                contaminated.add(doc_id)
    logger.info("contaminated 3-slice docs (cluster touches 4plus): %d", len(contaminated))

    total = sum(pq.read_metadata(fsspec.open(p, "rb").open()).num_rows for p in fsspec_glob(f"{SLICE_DOCS}/*.parquet"))
    clean = total - len(contaminated)

    ids = sorted(contaminated)
    n_per = 200_000
    for i in range(0, max(1, len(ids)), n_per):
        chunk = ids[i : i + n_per]
        with fsspec.open(f"{CONTAMINATED_IDS}/ids-{i // n_per:05d}.parquet", "wb") as f:
            pq.write_table(pa.table({"three_id": chunk}, schema=ID_SCHEMA), f)

    summary = {
        "banding": "286x26 (r=11, ~0.8 Jaccard — canonical fuzzy dedup)",
        "slice_docs_total": total,
        "contaminated_docs": len(contaminated),
        "contaminated_frac": round(len(contaminated) / total, 4),
        "clean_docs": clean,
        "slice_tokens_est_B": round(total * TOK_PER_DOC / 1e9, 2),
        "clean_tokens_est_B": round(clean * TOK_PER_DOC / 1e9, 2),
        "contaminated_ids": CONTAMINATED_IDS,
    }
    with fsspec.open(f"{SLICE_ROOT}/clean_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("clean summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
