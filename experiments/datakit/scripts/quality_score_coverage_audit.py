# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit corpus coverage of the nemotron-88k quality scores.

Answers one question: which documents ended up with no quality score, and how many.
Everything is measured from parquet footers and ``.artifact.json`` sidecars; nothing is
extrapolated. Three independent counts per join-ready source:

* ``normalized/<source>/<subset>_<hash>/outputs/main`` rows -- one row per corpus document,
  because normalize's ``group_by(id)`` emits canonical rows only. This is the corpus
  denominator, and it is the only side that is neither dedup-filtered nor chunk-inflated.
* ``datakit/embed/harrier/...`` rows -- one row per embedded document. The shortfall against
  normalized is the global-dedup drop, which is why those documents have no score.
* ``datakit/quality-scores/<tokens_leaf>/`` rows -- what the scoring run actually wrote.

The driver additionally inventories every tokenize and harrier sidecar to decide whether the
142 join-ready source_keys cover the whole corpus: a marin-tokenizer leaf whose
``dep_paths[0]`` points at ``normalized/...`` rather than ``datakit/sample_*/...`` is a real
source, and if its normalized upstream is absent from the 142 it is a source with no scores
at all.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --cpu 4 --memory 16g --disk 32g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.quality_score_coverage_audit
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import fsspec
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
PREFIX = f"{BUCKET}/marin"
TOKENIZE_ROOT = f"{PREFIX}/datakit/tokenize"
EMBED_ROOT = f"{PREFIX}/datakit/embed/harrier"
NORMALIZED_ROOT = f"{PREFIX}/normalized"
SCORES_ROOT = f"{PREFIX}/datakit/quality-scores"

RUN_ROOT = f"{PREFIX}/user/muchanem/quality_scores_audit"
STAGE_ROOT = f"{RUN_ROOT}/_leaf_stage"
INVENTORY_URL = f"{RUN_ROOT}/inventory.json"
REPORT_URL = f"{RUN_ROOT}/report.json"
PER_SOURCE_URL = f"{RUN_ROOT}/per_source.parquet"

NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MARIN_TOKENIZER = "marin-community/marin-tokenizer"

PART_RE = re.compile(r"part-(\d+)-of-(\d+)\.parquet$")
FOOTER_THREADS = 64
DISCOVERY_THREADS = 64
MAX_WORKERS = 48
WORKER_RESOURCES = ResourceConfig(cpu=8, ram="16g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)


def _fs():
    return fsspec.filesystem("s3")


def _key(path: str) -> str:
    """Normalize an artifact path (prefix-relative or ``s3://``) to a bucket-qualified key."""
    p = path.removeprefix("s3://").strip("/")
    return p if p.startswith(f"{BUCKET}/") else f"{PREFIX}/{p}"


def _sidecars(root: str) -> list[str]:
    """Glob leaf ``.artifact.json`` at both nesting depths.

    Sources with subsets nest as ``<source>/<subset>_<hash>``; a source with no subsets sits
    flat at ``<source>_<hash>``. Globbing only the nested depth drops whole sources silently.
    """
    fs = _fs()
    return sorted(set(fs.glob(f"{root}/*/.artifact.json")) | set(fs.glob(f"{root}/*/*/.artifact.json")))


def _read_artifact(path: str) -> tuple[str, dict | None]:
    try:
        return path, json.loads(_fs().cat(path))
    except Exception as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return path, None


def _read_many(paths: list[str]) -> list[tuple[str, dict | None]]:
    with ThreadPoolExecutor(max_workers=DISCOVERY_THREADS) as pool:
        return list(pool.map(_read_artifact, paths))


def _parquet_rows(url: str) -> int:
    """Row count from a parquet footer; -1 if the footer cannot be read."""
    try:
        with _fs().open(url, "rb") as fh:
            return pq.ParquetFile(fh).metadata.num_rows
    except Exception as exc:
        logger.warning("footer read failed %s: %s", url, exc)
        return -1


def _footer_sweep(paths: list[str]) -> list[int]:
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=FOOTER_THREADS) as pool:
        return list(pool.map(_parquet_rows, paths))


def _parquet_files(dir_key: str) -> list[str]:
    """Recursively list parquet shards under a directory key."""
    try:
        found = _fs().find(dir_key.rstrip("/"))
    except FileNotFoundError:
        return []
    return sorted(p for p in found if p.endswith(".parquet"))


def inventory() -> dict:
    """Inventory every tokenize and harrier leaf; decide what the 142 triples leave out."""
    t0 = time.monotonic()
    tok_arts = _read_many(_sidecars(TOKENIZE_ROOT))
    emb_arts = _read_many(_sidecars(EMBED_ROOT))
    logger.info("read %d tokenize + %d embed sidecars in %.1fs", len(tok_arts), len(emb_arts), time.monotonic() - t0)

    tokenize_leaves: list[dict] = []
    for path, art in tok_arts:
        if art is None:
            continue
        cfg = art.get("config") or {}
        res = art.get("result") or {}
        dep = (art.get("dep_paths") or [None])[0]
        tokenize_leaves.append(
            {
                "sidecar": path,
                "name": art.get("name"),
                "tokenizer": cfg.get("tokenizer"),
                "source_key": (res.get("source_keys") or {}).get("train"),
                "output_dir": (res.get("output_dirs") or {}).get("train"),
                "dep_path": dep,
                "counters": (res.get("counters") or {}).get("train") or {},
            }
        )

    embed_leaves: list[dict] = []
    for path, art in emb_arts:
        if art is None:
            continue
        res = art.get("result") or {}
        embed_leaves.append(
            {
                "sidecar": path,
                "name": art.get("name"),
                "source_key": res.get("source_key"),
                "output_dir": res.get("output_dir"),
                "dep_path": (art.get("dep_paths") or [None])[0],
                "dedup_attr_dir": res.get("dedup_attr_dir"),
                "counters": res.get("counters") or {},
            }
        )

    nemo = {leaf["source_key"]: leaf for leaf in tokenize_leaves if leaf["tokenizer"] == NEMOTRON and leaf["source_key"]}
    marin = defaultdict(list)
    for leaf in tokenize_leaves:
        if leaf["tokenizer"] == MARIN_TOKENIZER and leaf["source_key"]:
            marin[leaf["source_key"]].append(leaf)
    harrier = {leaf["source_key"]: leaf for leaf in embed_leaves if leaf["source_key"]}

    triples = sorted(set(nemo) & set(harrier))

    # A marin-tokenizer source_key outside the 142 matters only if it is a real normalized
    # source rather than a sample upstream. dep_paths[0] is the discriminator.
    marin_only = sorted(set(marin) - set(triples))
    marin_only_detail = []
    for key in marin_only:
        leaves = marin[key]
        deps = sorted({leaf["dep_path"] or "" for leaf in leaves})
        is_sample = all("/datakit/sample_" in d for d in deps)
        marin_only_detail.append(
            {
                "source_key": key,
                "dep_paths": deps,
                "is_sample_upstream": is_sample,
                "names": sorted({leaf["name"] or "" for leaf in leaves}),
                "output_dirs": sorted({leaf["output_dir"] or "" for leaf in leaves}),
            }
        )

    # For non-sample marin-only source_keys, is the underlying normalized source nonetheless
    # covered by one of the 142? Compare on the normalized leaf directory, stripping the
    # ``/outputs/main`` suffix so a differently-suffixed key still matches.
    def _norm_dir(source_key: str) -> str:
        k = (source_key or "").strip("/")
        return k.removesuffix("/outputs/main")

    triple_norm_dirs = {_norm_dir(k) for k in triples}
    uncovered = [
        d
        for d in marin_only_detail
        if not d["is_sample_upstream"] and _norm_dir(d["source_key"]) not in triple_norm_dirs
    ]

    # Independent check: which normalized leaves exist at all, and which have no triple.
    norm_dirs = sorted(
        {p.rsplit("/.artifact.json", 1)[0] for p in _sidecars(NORMALIZED_ROOT)}
        | {p.rsplit("/", 1)[0] for p in _fs().glob(f"{NORMALIZED_ROOT}/*/*/outputs")}
        | {p.rsplit("/", 1)[0] for p in _fs().glob(f"{NORMALIZED_ROOT}/*/outputs")}
    )
    norm_rel = {p.removeprefix(f"{PREFIX}/") for p in norm_dirs}
    normalized_without_triple = sorted(norm_rel - triple_norm_dirs)

    counters_present = sum(1 for leaf in embed_leaves if leaf["counters"])
    dedup_counter_present = sum(1 for leaf in embed_leaves if "embed/docs_dedup_dropped" in (leaf["counters"] or {}))

    inv = {
        "tokenize_leaves": len(tokenize_leaves),
        "embed_leaves": len(embed_leaves),
        "tokenizers": sorted({leaf["tokenizer"] or "" for leaf in tokenize_leaves}),
        "nemotron_source_keys": len(nemo),
        "marin_source_keys": len(marin),
        "harrier_source_keys": len(harrier),
        "triples": len(triples),
        "nemotron_without_harrier": sorted(set(nemo) - set(harrier)),
        "harrier_without_nemotron": sorted(set(harrier) - set(nemo)),
        "marin_only_source_keys": len(marin_only),
        "marin_only_sample_upstream": sum(1 for d in marin_only_detail if d["is_sample_upstream"]),
        "marin_only_normalized_upstream": sum(1 for d in marin_only_detail if not d["is_sample_upstream"]),
        "marin_only_normalized_uncovered": uncovered,
        "marin_only_detail": marin_only_detail,
        "normalized_leaf_dirs": len(norm_rel),
        "normalized_without_triple": normalized_without_triple,
        "embed_leaves_with_any_counters": counters_present,
        "embed_leaves_with_dedup_counter": dedup_counter_present,
        "dedup_attr_dirs": sorted({leaf["dedup_attr_dir"] or "" for leaf in embed_leaves if leaf["source_key"]}),
        "triple_source_keys": triples,
    }

    pairs = [
        {
            "source_key": key,
            "normalized_dir": _key(_norm_dir(key) + "/outputs/main"),
            "tokens_dir": _key(nemo[key]["output_dir"]),
            "embed_dir": _key(harrier[key]["output_dir"]),
            "tokens_leaf": _key(nemo[key]["output_dir"]).split("datakit/tokenize/", 1)[1].removesuffix("/train"),
            "dedup_attr_dir": harrier[key]["dedup_attr_dir"],
        }
        for key in triples
    ]
    return inv, pairs


def sweep_source(pair: dict) -> dict:
    """Footer-count normalized, embed, token and score shards for one source.

    Per-shard detail is staged to object storage; only the aggregate returns through the
    coordinator. Returning ~500k shard records through it OOM'd a previous sweep.
    """
    fs = _fs()
    staged = f"{STAGE_ROOT}/{pair['source_key'].replace('/', '__')}.parquet"
    if fs.exists(staged):
        with fs.open(staged, "rb") as fh:
            return _summarize(pair, pl.read_parquet(fh), resumed=True)

    t0 = time.monotonic()
    norm_files = _parquet_files(pair["normalized_dir"])
    embed_files = _parquet_files(pair["embed_dir"])
    token_files = _parquet_files(pair["tokens_dir"])
    score_files = _parquet_files(f"{SCORES_ROOT}/{pair['tokens_leaf']}")
    list_time = time.monotonic() - t0

    t1 = time.monotonic()
    counts = {
        "normalized": (norm_files, _footer_sweep(norm_files)),
        "embed": (embed_files, _footer_sweep(embed_files)),
        "tokens": (token_files, _footer_sweep(token_files)),
        "scores": (score_files, _footer_sweep(score_files)),
    }
    footer_time = time.monotonic() - t1

    frames = [
        pl.DataFrame(
            {
                "source_key": [pair["source_key"]] * len(files),
                "side": [side] * len(files),
                "path": files,
                "rows": rows,
                "shard_index": [int(m.group(1)) if (m := PART_RE.search(p)) else -1 for p in files],
            }
        )
        for side, (files, rows) in counts.items()
    ]
    df = (
        pl.concat([f for f in frames if f.height])
        if any(f.height for f in frames)
        else pl.DataFrame({"source_key": [], "side": [], "path": [], "rows": [], "shard_index": []})
    )
    buf = BytesIO()
    df.write_parquet(buf)
    fs.pipe_file(staged, buf.getvalue())

    out = _summarize(pair, df, resumed=False)
    out["list_time"] = list_time
    out["footer_time"] = footer_time
    return out


def _side(df: pl.DataFrame, side: str) -> tuple[int, int, int]:
    """Return ``(files, rows, failed_footers)`` for one side of a source."""
    if not df.height:
        return 0, 0, 0
    sub = df.filter(pl.col("side") == side)
    if not sub.height:
        return 0, 0, 0
    ok = sub.filter(pl.col("rows") >= 0)
    return sub.height, int(ok["rows"].sum()), int((sub["rows"] < 0).sum())


def _summarize(pair: dict, df: pl.DataFrame, resumed: bool) -> dict:
    norm_files, norm_rows, norm_bad = _side(df, "normalized")
    emb_files, emb_rows, emb_bad = _side(df, "embed")
    tok_files, tok_rows, tok_bad = _side(df, "tokens")
    sc_files, sc_rows, sc_bad = _side(df, "scores")
    scored_idx = set(df.filter((pl.col("side") == "scores") & (pl.col("rows") > 0))["shard_index"].to_list())
    token_idx = set(df.filter(pl.col("side") == "tokens")["shard_index"].to_list())
    return {
        "source_key": pair["source_key"],
        "tokens_leaf": pair["tokens_leaf"],
        "resumed": resumed,
        "normalized_files": norm_files,
        "normalized_rows": norm_rows,
        "normalized_footer_failed": norm_bad,
        "embed_files": emb_files,
        "embed_rows": emb_rows,
        "embed_footer_failed": emb_bad,
        "token_files": tok_files,
        "token_rows": tok_rows,
        "token_footer_failed": tok_bad,
        "score_files": sc_files,
        "score_rows": sc_rows,
        "score_footer_failed": sc_bad,
        "token_shards_without_score": len(token_idx - scored_idx),
        "dedup_dropped": norm_rows - emb_rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    fs = _fs()
    t_start = time.monotonic()

    inv, pairs = inventory()
    fs.pipe_file(INVENTORY_URL, json.dumps(inv, indent=2).encode())
    logger.info(
        "INVENTORY %s",
        json.dumps({k: v for k, v in inv.items() if k not in ("marin_only_detail", "triple_source_keys")}, indent=2),
    )

    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=args.max_workers,
        name="quality-score-coverage-audit",
    )
    outcome = ctx.execute(Dataset.from_list(pairs).map(sweep_source), verbose=True)
    rows = sorted(outcome.results, key=lambda r: r["source_key"])

    per_source = pl.DataFrame(rows).sort("source_key")
    buf = BytesIO()
    per_source.write_parquet(buf)
    fs.pipe_file(PER_SOURCE_URL, buf.getvalue())

    def _total(col: str) -> int:
        return int(per_source[col].sum())

    corpus = _total("normalized_rows")
    embedded = _total("embed_rows")
    scored = _total("score_rows")
    worst = (
        per_source.select(["source_key", "normalized_rows", "embed_rows", "dedup_dropped", "score_rows"])
        .sort("dedup_dropped", descending=True)
        .head(20)
        .to_dicts()
    )
    report = {
        "corpus_docs_normalized": corpus,
        "embedded_docs": embedded,
        "dedup_dropped": corpus - embedded,
        "score_rows": scored,
        "token_rows_chunk_inflated": _total("token_rows"),
        "normalized_files": _total("normalized_files"),
        "embed_files": _total("embed_files"),
        "token_files": _total("token_files"),
        "score_files": _total("score_files"),
        "footer_failures": {side: _total(f"{side}_footer_failed") for side in ("normalized", "embed", "token", "score")},
        "token_shards_without_score": _total("token_shards_without_score"),
        "sources_with_no_scores": [r["source_key"] for r in rows if r["score_rows"] == 0],
        "sources_with_no_normalized_rows": [r["source_key"] for r in rows if r["normalized_rows"] == 0],
        "sources_where_embed_exceeds_normalized": [
            {"source_key": r["source_key"], "normalized_rows": r["normalized_rows"], "embed_rows": r["embed_rows"]}
            for r in rows
            if r["embed_rows"] > r["normalized_rows"]
        ],
        "worst_dedup_drop": worst,
        "inventory_path": f"s3://{INVENTORY_URL}",
        "per_source_path": f"s3://{PER_SOURCE_URL}",
        "counters": {k: v for k, v in sorted(outcome.counters.items())},
        "total_wall": time.monotonic() - t_start,
    }
    fs.pipe_file(REPORT_URL, json.dumps(report, indent=2).encode())
    logger.info("REPORT %s", json.dumps({k: v for k, v in report.items() if k != "counters"}, indent=2))


if __name__ == "__main__":
    main()
