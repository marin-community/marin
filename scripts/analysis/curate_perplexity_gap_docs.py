# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curate val documents spanning train-Jaccard bands (+ their train near-dups).

For the "why does 1e22 only get *worse* as you decontaminate?" investigation: we
want to look at concrete documents whose perplexity-vs-scale behavior differs by
how contaminated they are. This builds a small curated set the Delphi ladder can
be scored on per-token.

Selection (all from the already-built 4plus-only artifacts, in-region us-east5):

- **Banded val docs** from the `decon_val_4plus/j090` docs parquet, which carries
  `id, text, max_jaccard` for every fully-contained val doc with 4plus max-J < 0.90.
  Bands: clean (max-J == 0), ~0.60, ~0.75, and high [0.86, 0.90). A length window
  keeps the bands comparable.
- **Train near-duplicates** of the high-band val docs: for each, the best-matching
  `4plus` train doc id from `4plus_284x71/verified_pairs` (max jaccard), whose text
  is pulled from the `4plus` normalized parquet by a parallel Zephyr scan (filter by
  a small id set — most shards contribute nothing).

Output (GCS): a `curated_docs.jsonl` (one record per doc with `text` + provenance)
and a `curated_meta.json` (the same records minus text, in jsonl order) under
`gs://…/midtrain_dedup/perplexity_gap/`.

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 32GB --disk 40GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name curate-ppl-gap-docs \
        -- python scripts/analysis/curate_perplexity_gap_docs.py
"""

import json
import logging
from collections.abc import Iterator
from functools import partial

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

MIDTRAIN = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
J090_DOCS = f"{MIDTRAIN}/decon_val_4plus/j090/docs"
VERIFIED_4PLUS = f"{MIDTRAIN}/4plus_284x71/verified_pairs"
NORM_4PLUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
OUT = f"{MIDTRAIN}/perplexity_gap"

# (label, lo, hi) — half-open max-J bands. "clean" is exactly 0 (no verified pair;
# the verify floor is 0.5, so <0.5 is unrecorded and lumped at 0). j050 is the
# lowest *recorded* contamination band.
BANDS = [
    ("clean", 0.0, 1e-9),
    ("j050", 0.50, 0.55),
    ("j060", 0.58, 0.63),
    ("j075", 0.73, 0.78),
    ("j088", 0.86, 0.90),
]
PER_BAND = 5
TWIN_BANDS = {"j088"}  # pull train near-dups for these bands
MIN_CHARS = 1500
MAX_CHARS = 7000


def best_train_match() -> dict[str, tuple[str, float]]:
    """val_id -> (best 4plus train other_id, its jaccard), over all verified pairs."""
    best: dict[str, tuple[str, float]] = {}
    for path in fsspec_glob(f"{VERIFIED_4PLUS}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["val_id", "other_id", "jaccard"])
        for vid, oid, jac in zip(
            t.column("val_id").to_pylist(),
            t.column("other_id").to_pylist(),
            t.column("jaccard").to_pylist(),
            strict=True,
        ):
            cur = best.get(vid)
            if cur is None or jac > cur[1]:
                best[vid] = (oid, float(jac))
    return best


def load_banded_val_docs() -> list[dict]:
    """All fully-contained val docs (id, text, max_jaccard) from the j090 docs parquet."""
    docs: list[dict] = []
    for path in fsspec_glob(f"{J090_DOCS}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["id", "text", "max_jaccard"])
        for did, text, mj in zip(
            t.column("id").to_pylist(),
            t.column("text").to_pylist(),
            t.column("max_jaccard").to_pylist(),
            strict=True,
        ):
            docs.append({"id": did, "text": text, "max_jaccard": float(mj or 0.0)})
    return docs


def select_band(docs: list[dict], lo: float, hi: float) -> list[dict]:
    """Deterministic pick of PER_BAND length-windowed docs whose max-J is in [lo, hi)."""
    cands = [d for d in docs if lo <= d["max_jaccard"] < hi and MIN_CHARS <= len(d["text"]) <= MAX_CHARS]
    cands.sort(key=lambda d: d["id"])
    return cands[:PER_BAND]


TWIN_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])


def _scan_norm_for_ids(path: str, wanted: frozenset[str]) -> Iterator[dict]:
    with fsspec.open(path, "rb") as f:
        t = pq.read_table(f, columns=["id", "text"])
    for did, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True):
        if did in wanted:
            yield {"id": did, "text": text}


def fetch_train_text(twin_ids: frozenset[str]) -> dict[str, str]:
    """Parallel in-region scan of the 4plus normalized parquet for a small id set.

    Writes matches to a GCS parquet (most shards yield nothing) and reads them
    back — the proven collect pattern; the full 92 GB corpus never touches the
    driver.
    """
    if not twin_ids:
        return {}
    out_dir = f"{OUT}/train_twin_scan"
    shards = sorted(fsspec_glob(f"{NORM_4PLUS}/*.parquet"))
    ctx = ZephyrContext(
        name="curate-ppl-gap-train-scan",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
        # Preemptible+small coordinator: non-preemptible coordinators sit PENDING
        # for many minutes on this cluster (standing operational rule).
        coordinator_resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
    )
    ctx.execute(
        Dataset.from_list(shards)
        .flat_map(partial(_scan_norm_for_ids, wanted=twin_ids))
        .write_parquet(f"{out_dir}/twin-{{shard:05d}}-of-{{total:05d}}.parquet", schema=TWIN_SCHEMA)
    )
    result: dict[str, str] = {}
    for path in fsspec_glob(f"{out_dir}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["id", "text"])
        result.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    val_docs = load_banded_val_docs()
    logger.info("loaded %d fully-contained val docs", len(val_docs))
    best = best_train_match()
    logger.info("loaded best train match for %d val ids", len(best))

    records: list[dict] = []
    twin_ids: set[str] = set()
    for label, lo, hi in BANDS:
        picked = select_band(val_docs, lo, hi)
        logger.info("band %s [%.2f,%.2f): %d docs", label, lo, hi, len(picked))
        for d in picked:
            group = f"{label}-{d['id'][:8]}"
            records.append(
                {
                    "text": d["text"],
                    "doc_id": d["id"],
                    "role": "val",
                    "band": label,
                    "max_jaccard": d["max_jaccard"],
                    "group": group,
                    "train_twin_id": best.get(d["id"], (None, None))[0] if label in TWIN_BANDS else None,
                }
            )
            if label in TWIN_BANDS and d["id"] in best:
                twin_ids.add(best[d["id"]][0])

    train_text = fetch_train_text(frozenset(twin_ids))
    logger.info("fetched %d/%d train-twin texts", len(train_text), len(twin_ids))
    # Append train twins right after, tagged to their val group.
    val_by_twin = {r["train_twin_id"]: r for r in records if r.get("train_twin_id")}
    for tid, text in train_text.items():
        val_rec = val_by_twin.get(tid)
        records.append(
            {
                "text": text,
                "doc_id": tid,
                "role": "train_twin",
                "band": val_rec["band"] if val_rec else "twin",
                "max_jaccard": val_rec["max_jaccard"] if val_rec else None,
                "group": val_rec["group"] if val_rec else f"twin-{tid[:8]}",
                "train_twin_id": None,
            }
        )

    with fsspec.open(f"{OUT}/curated_docs.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with fsspec.open(f"{OUT}/curated_meta.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "text"} for r in records], f, indent=2)
    logger.info(
        "wrote %d curated docs (%d val + %d train_twin) to %s",
        len(records),
        sum(1 for r in records if r["role"] == "val"),
        sum(1 for r in records if r["role"] == "train_twin"),
        OUT,
    )


if __name__ == "__main__":
    main()
