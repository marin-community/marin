# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile the dedup drop at shard granularity to price a merge of the missing scores.

The coverage audit (``quality_score_coverage_audit.py``) already staged one parquet per
join-ready source under ``quality_scores_audit/_leaf_stage/``, carrying ``(side, path, rows,
shard_index)`` for the normalized, embed, token and score sides. Per-shard dedup drop is
``normalized_rows(k) - embed_rows(k)``: embed shard ``k`` is an order-preserving dedup filter of
normalized shard ``k``, and both sides share shard count and basename. ``--verify-pairs`` checks
that framing by reading the ``id`` column of a few shard pairs before the corpus-wide arithmetic
is trusted.

What this adds on top of the staged counts:

* the shard-level concentration curve and Gini of the 4.30B dropped documents, which decides
  whether a merge can skip files or has to touch essentially all of them;
* score-file byte sizes, from a listing (not a footer read), to price a read-modify-write;
* an explicit uniformity test per source -- the observed spread of per-shard drop rate against
  the binomial spread expected under hash partitioning.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --cpu 8 --memory 16g --disk 32g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.dedup_drop_shard_profile
"""

import argparse
import itertools
import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import fsspec
import numpy as np
import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
PREFIX = f"{BUCKET}/marin"
AUDIT_ROOT = f"{PREFIX}/user/muchanem/quality_scores_audit"
STAGE_ROOT = f"{AUDIT_ROOT}/_leaf_stage"

RUN_ROOT = f"{PREFIX}/user/muchanem/quality_scores_merge_profile"
PER_SHARD_URL = f"{RUN_ROOT}/per_shard.parquet"
PER_SOURCE_URL = f"{RUN_ROOT}/per_source.parquet"
REPORT_URL = f"{RUN_ROOT}/report.json"

# Duplicate ``id``s in normalized output make its row count overstate distinct documents, so this
# source's drop arithmetic is reported on its own rather than folded into corpus statistics.
DUPLICATE_ID_SOURCE = "common_crawl_focus_2026_22"

SPREAD_SOURCES = ("swe-zero-12m", "nemotron_cc_v2/medium_quality")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50)
LIST_THREADS = 64
VERIFY_PAIRS = 6


def _fs():
    return fsspec.filesystem("s3")


def _short(source_key: str) -> str:
    """``normalized/<source>/<subset>_<hash>/outputs/main`` -> ``<source>/<subset>``."""
    body = source_key.strip("/").removesuffix("/outputs/main")
    body = body.split("normalized/", 1)[-1]
    head, _, tail = body.rpartition("/")
    tail = tail.rsplit("_", 1)[0] if "_" in tail else tail
    return f"{head}/{tail}" if head else tail


def _read_stage(path: str) -> pl.DataFrame:
    with _fs().open(path, "rb") as fh:
        return pl.read_parquet(fh)


def _list_sizes(directory: str) -> dict[str, int]:
    """Object sizes under a directory, from a listing -- no footer reads."""
    try:
        found = _fs().find(directory.rstrip("/"), detail=True)
    except FileNotFoundError:
        return {}
    return {k: int(v.get("size") or 0) for k, v in found.items() if k.endswith(".parquet")}


def verify_filter_framing(per_shard: pl.DataFrame, stage: pl.DataFrame, seed: int) -> list[dict]:
    """Check that embed shard k is an order-preserving subset of normalized shard k.

    Only shards small enough to read cheaply are sampled, and only ones with a nonzero drop --
    a shard with zero drop would pass trivially.
    """
    fs = _fs()
    paths = stage.filter(pl.col("side").is_in(["normalized", "embed"])).select(
        "source_key", "side", "path", "shard_index"
    )
    wide = paths.pivot(on="side", index=["source_key", "shard_index"], values="path")
    cand = (
        per_shard.filter((pl.col("dropped") > 0) & (pl.col("normalized_rows").is_between(1, 200_000)))
        .join(wide, on=["source_key", "shard_index"], how="inner")
        .to_dicts()
    )
    rng = random.Random(seed)
    out = []
    for row in rng.sample(cand, min(VERIFY_PAIRS, len(cand))):
        with fs.open(row["normalized"], "rb") as fh:
            norm_ids = pq.ParquetFile(fh).read(columns=["id"]).column("id").to_pylist()
        with fs.open(row["embed"], "rb") as fh:
            emb_ids = pq.ParquetFile(fh).read(columns=["id"]).column("id").to_pylist()
        norm_pos = {}
        for i, v in enumerate(norm_ids):
            norm_pos.setdefault(v, i)
        positions = [norm_pos.get(v, -1) for v in emb_ids]
        out.append(
            {
                "source": _short(row["source_key"]),
                "shard_index": row["shard_index"],
                "normalized_rows": len(norm_ids),
                "embed_rows": len(emb_ids),
                "dropped": len(norm_ids) - len(emb_ids),
                "embed_ids_all_present_in_normalized": all(p >= 0 for p in positions),
                "embed_order_preserved": all(a < b for a, b in itertools.pairwise(positions)),
                "normalized_ids_distinct": len(norm_pos) == len(norm_ids),
                "embed_ids_distinct": len(set(emb_ids)) == len(emb_ids),
            }
        )
    return out


def gini(values: np.ndarray) -> float:
    """Gini over nonnegative per-shard drop counts; 0 = perfectly even, 1 = all in one shard."""
    v = np.sort(values.astype(np.float64))
    total = v.sum()
    if total <= 0:
        return 0.0
    n = v.size
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * total))


def concentration(dropped: np.ndarray) -> dict:
    """Share of total drops held by the top q fraction of shards, ranked by drop count."""
    order = np.sort(dropped.astype(np.float64))[::-1]
    total = order.sum()
    cum = np.cumsum(order)
    n = order.size
    return {
        f"top_{int(q * 100)}pct_shards_share_of_drops": float(cum[max(0, math.ceil(q * n) - 1)] / total)
        for q in QUANTILES
    }


def source_spread(per_shard: pl.DataFrame, source: str) -> dict:
    """Per-shard drop-rate spread for one source, against the binomial expectation.

    Under hash partitioning each shard is an independent uniform sample of the source, so the
    per-shard drop rate should sit within roughly ``sqrt(p(1-p)/rows)`` of the source mean.
    """
    sub = per_shard.filter(pl.col("source") == source).filter(pl.col("normalized_rows") > 0)
    if not sub.height:
        return {"source": source, "shards": 0}
    rate = (sub["dropped"] / sub["normalized_rows"]).to_numpy()
    rows = sub["normalized_rows"].to_numpy()
    p = float(sub["dropped"].sum() / sub["normalized_rows"].sum())
    expected_sd = float(np.mean(np.sqrt(p * (1 - p) / rows)))
    return {
        "source": source,
        "shards": sub.height,
        "normalized_rows": int(sub["normalized_rows"].sum()),
        "dropped": int(sub["dropped"].sum()),
        "source_drop_rate": p,
        "shard_rate_min": float(rate.min()),
        "shard_rate_p50": float(np.median(rate)),
        "shard_rate_max": float(rate.max()),
        "shard_rate_sd": float(rate.std(ddof=0)),
        "binomial_sd_expected": expected_sd,
        "sd_ratio_observed_over_binomial": float(rate.std(ddof=0) / expected_sd) if expected_sd else None,
        "median_shard_rows": float(np.median(rows)),
        "zero_drop_shards": int((sub["dropped"] == 0).sum()),
    }


def build_per_shard(stage: pl.DataFrame, sizes: dict[str, int]) -> pl.DataFrame:
    """One row per (source, shard_index) with row counts, drop, and score-file bytes."""
    counts = (
        stage.filter(pl.col("shard_index") >= 0)
        .pivot(on="side", index=["source_key", "shard_index"], values="rows", aggregate_function="sum")
        .fill_null(0)
    )
    score_paths = stage.filter(pl.col("side") == "scores").select(
        "source_key", "shard_index", pl.col("path").alias("score_path")
    )
    size_df = pl.DataFrame(
        {"score_path": list(sizes.keys()), "score_bytes": list(sizes.values())},
        schema={"score_path": pl.String, "score_bytes": pl.Int64},
    )
    return (
        counts.join(score_paths, on=["source_key", "shard_index"], how="left")
        .join(size_df, on="score_path", how="left")
        .with_columns(
            pl.col("score_bytes").fill_null(0),
            (pl.col("normalized") - pl.col("embed")).alias("dropped"),
            pl.col("source_key").map_elements(_short, return_dtype=pl.String).alias("source"),
        )
        .rename({"normalized": "normalized_rows", "embed": "embed_rows", "scores": "score_rows"})
        .drop("tokens")
        .sort("source_key", "shard_index")
    )


def summarize(per_shard: pl.DataFrame, label: str) -> dict:
    dropped = per_shard["dropped"].to_numpy()
    touched = per_shard.filter(pl.col("dropped") > 0)
    with_score_file = per_shard.filter(pl.col("score_path").is_not_null())
    touched_with_file = touched.filter(pl.col("score_path").is_not_null())
    total_drops = int(dropped.sum())
    return {
        "scope": label,
        "sources": per_shard["source"].n_unique(),
        "shards": per_shard.height,
        "score_files": with_score_file.height,
        "score_bytes": int(per_shard["score_bytes"].sum()),
        "normalized_rows": int(per_shard["normalized_rows"].sum()),
        "embed_rows": int(per_shard["embed_rows"].sum()),
        "score_rows": int(per_shard["score_rows"].sum()),
        "dropped": total_drops,
        "drop_rate": total_drops / max(1, int(per_shard["normalized_rows"].sum())),
        "shards_with_drops": touched.height,
        "shards_with_zero_drops": int((dropped == 0).sum()),
        "shards_with_negative_drops": int((dropped < 0).sum()),
        "score_files_touched": touched_with_file.height,
        "score_files_touched_frac": touched_with_file.height / max(1, with_score_file.height),
        "score_bytes_touched": int(touched_with_file["score_bytes"].sum()),
        "shards_with_drops_but_no_score_file": touched.height - touched_with_file.height,
        "gini_of_drops_over_shards": gini(dropped),
        "concentration": concentration(dropped),
        "drops_in_zero_drop_shards_share": 0.0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--near-zero-rate", type=float, default=0.001, help="drop rate below which a source is skippable")
    args = ap.parse_args()

    fs = _fs()
    t0 = time.monotonic()

    stage_files = sorted(p for p in fs.ls(STAGE_ROOT) if p.endswith(".parquet"))
    logger.info("reading %d staged source files", len(stage_files))
    with ThreadPoolExecutor(max_workers=LIST_THREADS) as pool:
        stage = pl.concat(list(pool.map(_read_stage, stage_files)))
    logger.info("stage rows=%d in %.1fs", stage.height, time.monotonic() - t0)

    score_dirs = sorted({p.rsplit("/", 1)[0] for p in stage.filter(pl.col("side") == "scores")["path"].to_list()})
    t1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=LIST_THREADS) as pool:
        sizes = {k: v for d in pool.map(_list_sizes, score_dirs) for k, v in d.items()}
    logger.info("sized %d score objects over %d dirs in %.1fs", len(sizes), len(score_dirs), time.monotonic() - t1)

    per_shard = build_per_shard(stage, sizes)
    logger.info("per-shard rows=%d", per_shard.height)

    verification = verify_filter_framing(per_shard, stage, args.seed)
    logger.info("VERIFY %s", json.dumps(verification, indent=2))

    outlier = per_shard.filter(pl.col("source").str.contains(DUPLICATE_ID_SOURCE, literal=True))
    clean = per_shard.filter(~pl.col("source").str.contains(DUPLICATE_ID_SOURCE, literal=True))

    per_source = (
        clean.group_by("source")
        .agg(
            pl.len().alias("shards"),
            pl.col("normalized_rows").sum(),
            pl.col("embed_rows").sum(),
            pl.col("dropped").sum(),
            pl.col("score_bytes").sum(),
            (pl.col("dropped") > 0).sum().alias("shards_with_drops"),
            (pl.col("dropped") == 0).sum().alias("shards_zero_drop"),
        )
        .with_columns((pl.col("dropped") / pl.col("normalized_rows")).alias("drop_rate"))
        .sort("dropped", descending=True)
    )

    near_zero = per_source.filter(pl.col("drop_rate") < args.near_zero_rate)
    fully_clean = per_source.filter(pl.col("shards_with_drops") == 0)

    report = {
        "verification": verification,
        "corpus_excluding_duplicate_id_source": summarize(clean, "corpus_excl_outlier"),
        "corpus_including_duplicate_id_source": summarize(per_shard, "corpus_all"),
        "duplicate_id_outlier": summarize(outlier, DUPLICATE_ID_SOURCE) if outlier.height else None,
        "sources_below_near_zero_rate": {
            "threshold": args.near_zero_rate,
            "count": near_zero.height,
            "normalized_rows": int(near_zero["normalized_rows"].sum()),
            "dropped": int(near_zero["dropped"].sum()),
            "score_bytes": int(near_zero["score_bytes"].sum()),
            "shards": int(near_zero["shards"].sum()),
            "sources": near_zero.select("source", "shards", "normalized_rows", "dropped", "drop_rate").to_dicts(),
        },
        "sources_with_no_dropped_shard": fully_clean.select("source", "shards", "normalized_rows").to_dicts(),
        "top_sources_by_dropped": per_source.head(20).to_dicts(),
        "top_sources_by_drop_rate": (
            per_source.filter(pl.col("normalized_rows") > 1_000_000)
            .sort("drop_rate", descending=True)
            .head(15)
            .to_dicts()
        ),
        "spread": [source_spread(clean, s) for s in SPREAD_SOURCES],
        "per_shard_path": f"s3://{PER_SHARD_URL}",
        "per_source_path": f"s3://{PER_SOURCE_URL}",
        "wall": time.monotonic() - t0,
    }

    for name, url in ((per_shard, PER_SHARD_URL), (per_source, PER_SOURCE_URL)):
        buf = BytesIO()
        name.write_parquet(buf)
        fs.pipe_file(url, buf.getvalue())
    fs.pipe_file(REPORT_URL, json.dumps(report, indent=2).encode())
    logger.info("REPORT %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
