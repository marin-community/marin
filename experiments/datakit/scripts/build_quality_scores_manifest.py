# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the shard-level join manifest pairing Nemotron token shards with harrier embeddings.

The quality-scoring run fans out over ``(source_key, shard_index)`` tasks: each task
sorted-merge-joins one Nemotron token shard against the co-partitioned harrier embedding
shard on ``id`` and writes ``(id, score)`` to a mirrored output path. This script produces
the manifest that drives that fan-out.

Discovery is two ``fs.glob`` calls over the tokenize and harrier-embed ``.artifact.json``
sidecars; leaves pair on ``result.source_key`` (the normalized upstream), never on directory
name, because the step hash differs per stage. Sample upstreams (``sample_10pct``,
``sample_100b``, ``sample_0.1b``) drop out of the pairing on their own: their source_key
points at ``datakit/sample_*`` and no harrier leaf mirrors it.

Per-leaf listing and embed-side parquet footer reads fan out one Zephyr task per leaf; the
footers give the per-shard document count, since embeddings are one row per document while
the token side is inflated by chunking.

Before any of that the driver checks the load-bearing unverified assumption: that the ids in
embed ``part-k`` are a subset of the ids in token ``part-k``. The whole map-only plan rests
on it, so a failure aborts the job rather than producing a manifest nobody should trust.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --cpu 4 --memory 16g --disk 32g \\
        -- python -m experiments.datakit.scripts.build_quality_scores_manifest
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import fsspec
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

BUCKET = "marin-us-east-02a"
PREFIX = f"{BUCKET}/marin"
TOKENIZE_ROOT = f"{PREFIX}/datakit/tokenize"
EMBED_ROOT = f"{PREFIX}/datakit/embed/harrier"
SCORES_ROOT = f"s3://{PREFIX}/datakit/quality-scores"
OUT_ROOT = f"s3://{PREFIX}/user/muchanem/quality_scores_run/manifest"

MANIFEST_URL = f"{OUT_ROOT}/manifest.parquet"
SUMMARY_URL = f"{OUT_ROOT}/summary.json"
VALIDATION_URL = f"{OUT_ROOT}/validation.json"
SUCCESS_URL = f"{OUT_ROOT}/_SUCCESS"

NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
EXPECTED_LEAVES = 142

PART_RE = re.compile(r"part-(\d+)-of-(\d+)\.parquet$")

MAX_WORKERS = 48
WORKER_RESOURCES = ResourceConfig(cpu=8, ram="16g")
FOOTER_THREADS = 64
DISCOVERY_THREADS = 64


def _fs():
    return fsspec.filesystem("s3")


def _abs_key(path: str) -> str:
    """Normalize an artifact path to a bucket-qualified key.

    Stage artifacts record output dirs relative to ``MARIN_PREFIX``
    (``datakit/tokenize/...``); some callers hand in a full ``s3://`` URL instead.
    """
    p = path.removeprefix("s3://").strip("/")
    return p if p.startswith(f"{BUCKET}/") else f"{PREFIX}/{p}"


def _leaf_rel(output_dir: str, root_marker: str) -> str:
    """Return ``<source>/<subset>_<hash>`` from a stage output dir.

    Splits on the stage marker rather than counting path components: one source_key carries
    an anomalous ``data/datakit/normalized/...`` prefix, so nothing here may parse positionally.
    """
    idx = output_dir.find(root_marker)
    if idx < 0:
        raise ValueError(f"{output_dir!r} does not contain {root_marker!r}")
    rel = output_dir[idx + len(root_marker) :].strip("/")
    if rel.endswith("/train"):
        rel = rel[: -len("/train")]
    return rel


def _read_artifact(path: str) -> tuple[str, dict | None]:
    try:
        return path, json.loads(_fs().cat(path))
    except Exception as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return path, None


def discover() -> tuple[list[dict], dict]:
    """Pair Nemotron tokenize leaves with harrier embed leaves on ``result.source_key``."""
    fs = _fs()
    t0 = time.monotonic()
    tok_sidecars = fs.glob(f"{TOKENIZE_ROOT}/*/*/.artifact.json")
    emb_sidecars = fs.glob(f"{EMBED_ROOT}/*/*/.artifact.json")
    logger.info(
        "globbed %d tokenize + %d embed sidecars in %.1fs",
        len(tok_sidecars),
        len(emb_sidecars),
        time.monotonic() - t0,
    )

    with ThreadPoolExecutor(max_workers=DISCOVERY_THREADS) as pool:
        tok_arts = list(pool.map(_read_artifact, tok_sidecars))
        emb_arts = list(pool.map(_read_artifact, emb_sidecars))
    read_errors = sum(1 for _, a in tok_arts + emb_arts if a is None)

    nemo: dict[str, dict] = {}
    dup_tokens: list[str] = []
    for path, art in tok_arts:
        if art is None:
            continue
        cfg = art.get("config") or {}
        res = art.get("result") or {}
        if cfg.get("tokenizer") != NEMOTRON:
            continue
        source_key = (res.get("source_keys") or {}).get("train")
        out_dir = (res.get("output_dirs") or {}).get("train")
        if not source_key or not out_dir:
            logger.warning("nemotron leaf %s missing source_keys.train/output_dirs.train", path)
            continue
        if source_key in nemo:
            dup_tokens.append(source_key)
        nemo[source_key] = {"source_key": source_key, "tokens_dir": _abs_key(out_dir)}

    harrier: dict[str, dict] = {}
    dup_embeds: list[str] = []
    for path, art in emb_arts:
        if art is None:
            continue
        res = art.get("result") or {}
        source_key = res.get("source_key")
        out_dir = res.get("output_dir")
        if not source_key or not out_dir:
            logger.warning("harrier leaf %s missing source_key/output_dir", path)
            continue
        if source_key in harrier:
            dup_embeds.append(source_key)
        harrier[source_key] = {"source_key": source_key, "embed_dir": _abs_key(out_dir)}

    paired_keys = sorted(set(nemo) & set(harrier))
    pairs = []
    for key in paired_keys:
        tokens_dir = nemo[key]["tokens_dir"]
        embed_dir = harrier[key]["embed_dir"]
        tok_rel = _leaf_rel(tokens_dir, "datakit/tokenize/")
        emb_rel = _leaf_rel(embed_dir, "datakit/embed/harrier/")
        source, tok_leaf = tok_rel.split("/", 1)
        pairs.append(
            {
                "source_key": key,
                "source": source,
                "subset": tok_leaf.rsplit("_", 1)[0],
                "tokens_rel": tok_rel,
                "embed_rel": emb_rel,
                "tokens_dir": tokens_dir,
                "embed_dir": embed_dir,
            }
        )

    stats = {
        "tokenize_sidecars": len(tok_sidecars),
        "embed_sidecars": len(emb_sidecars),
        "artifact_read_errors": read_errors,
        "nemotron_source_keys": len(nemo),
        "harrier_source_keys": len(harrier),
        "paired_source_keys": len(pairs),
        "nemotron_without_harrier": sorted(set(nemo) - set(harrier)),
        "harrier_without_nemotron": sorted(set(harrier) - set(nemo)),
        "duplicate_nemotron_source_keys": sorted(set(dup_tokens)),
        "duplicate_harrier_source_keys": sorted(set(dup_embeds)),
    }
    return pairs, stats


def _shard_dir(prefix_url: str) -> str:
    return prefix_url.removeprefix("s3://").rstrip("/")


def list_shards(dir_url: str) -> dict[int, tuple[str, int, int]]:
    """Map shard index -> (basename, size, num_shards) for ``part-*.parquet`` in a leaf."""
    out: dict[int, tuple[str, int, int]] = {}
    for path, info in _fs().find(_shard_dir(dir_url), detail=True).items():
        m = PART_RE.search(path)
        if not m:
            continue
        out[int(m.group(1))] = (path.rsplit("/", 1)[-1], int(info.get("size") or 0), int(m.group(2)))
    return out


def probe_shard_naming(dir_url: str) -> tuple[int, int]:
    """Return ``(num_shards, index_width)`` from the leaf's ``part-0...`` basename.

    One prefix listing per leaf instead of a full enumeration: the declared ``-of-TTTTT``
    suffix carries the shard count, and its zero-padding fixes the index width.
    """
    hits = _fs().glob(f"{_shard_dir(dir_url)}/part-0*-of-*.parquet")
    if not hits:
        raise ValueError(f"no part files under {dir_url}")
    m = PART_RE.search(sorted(hits)[0])
    if not m:
        raise ValueError(f"unparseable shard basename under {dir_url}")
    return int(m.group(2)), len(m.group(2))


def _id_column(url: str) -> list[str]:
    with _fs().open(_shard_dir(url), "rb") as fh:
        return pq.read_table(fh, columns=["id"]).column("id").to_pylist()


def validate_containment(pairs: list[dict], num_pairs: int) -> dict:
    """Confirm embed ``part-k`` ids are a subset of token ``part-k`` ids on real shards.

    The map-only join plan is only valid if shard k on each side covers the same key set.
    Samples across sources of different sizes and checks both ``part-00000`` and a middle
    shard, reading actual id columns rather than trusting the co-partitioning docstring.
    """
    fs = _fs()
    with ThreadPoolExecutor(max_workers=DISCOVERY_THREADS) as pool:
        futures = {p["source_key"]: pool.submit(probe_shard_naming, p["tokens_dir"]) for p in pairs}
        naming = {k: f.result() for k, f in futures.items()}
    by_size = sorted(pairs, key=lambda p: naming[p["source_key"]][0])

    candidates = [
        *[p for p in pairs if p["tokens_rel"].startswith("nemotron_cc_v2/diverse_qa")][:1],
        by_size[0],
        by_size[len(by_size) // 2],
        by_size[-1],
        *by_size,
    ]
    picks: list[dict] = []
    seen: set[str] = set()
    for p in candidates:
        if len(picks) >= num_pairs:
            break
        if p["source_key"] not in seen:
            seen.add(p["source_key"])
            picks.append(p)

    checks = []
    for p in picks:
        total, width = naming[p["source_key"]]
        for k in sorted({0, total // 2}):
            basename = f"part-{k:0{width}d}-of-{total:0{width}d}.parquet"
            tok_url = f"{p['tokens_dir']}/{basename}"
            emb_url = f"{p['embed_dir']}/{basename}"
            if not fs.exists(_shard_dir(emb_url)):
                checks.append(
                    {
                        "source_key": p["source_key"],
                        "shard_index": k,
                        "embed_exists": False,
                        "contained": None,
                        "note": f"embed shard absent: {emb_url}",
                    }
                )
                continue
            tok_ids = _id_column(tok_url)
            emb_ids = _id_column(emb_url)
            tok_set = set(tok_ids)
            emb_set = set(emb_ids)
            missing = emb_set - tok_set
            checks.append(
                {
                    "source_key": p["source_key"],
                    "source": p["source"],
                    "subset": p["subset"],
                    "shard_index": k,
                    "num_shards": total,
                    "token_rows": len(tok_ids),
                    "token_unique_ids": len(tok_set),
                    "embed_rows": len(emb_ids),
                    "embed_unique_ids": len(emb_set),
                    "embed_exists": True,
                    "contained": not missing,
                    "n_missing_from_tokens": len(missing),
                    "example_missing": sorted(missing)[:5],
                    "token_ids_sorted": tok_ids == sorted(tok_ids),
                    "embed_ids_sorted": emb_ids == sorted(emb_ids),
                    "dedup_drop_fraction": 1.0 - len(emb_set) / len(tok_set) if tok_set else None,
                    "token_paths": tok_url,
                    "embed_paths": emb_url,
                }
            )
            logger.info("containment %s shard %d: %s", p["source_key"], k, checks[-1]["contained"])

    result = {
        "checked_pairs": len(picks),
        "checked_shards": len(checks),
        "all_contained": all(c["contained"] for c in checks if c["embed_exists"]),
        "all_sorted": all(c["token_ids_sorted"] and c["embed_ids_sorted"] for c in checks if c["embed_exists"]),
        "checks": checks,
    }
    fs.pipe_file(VALIDATION_URL.removeprefix("s3://"), json.dumps(result, indent=2).encode())
    return result


def sweep_leaf(pair: dict) -> dict:
    """List both sides of one leaf and read embed-side parquet footers for doc counts."""
    t0 = time.monotonic()
    tok = list_shards(pair["tokens_dir"])
    emb = list_shards(pair["embed_dir"])
    list_time = time.monotonic() - t0

    def _rows(item):
        k, (basename, _size, _n) = item
        url = f"{pair['embed_dir']}/{basename}"
        try:
            with _fs().open(_shard_dir(url), "rb") as fh:
                return k, pq.ParquetFile(fh).metadata.num_rows
        except Exception as exc:
            logger.warning("footer read failed %s: %s", url, exc)
            return k, -1

    t1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=FOOTER_THREADS) as pool:
        embed_rows = dict(pool.map(_rows, sorted(emb.items())))
    footer_time = time.monotonic() - t1

    num_shards = max((v[2] for v in tok.values()), default=0)
    shards = [
        [
            k,
            tok[k][0],
            tok[k][1],
            emb[k][0] if k in emb else "",
            emb[k][1] if k in emb else 0,
            embed_rows.get(k, -1) if k in emb else -1,
        ]
        for k in sorted(tok)
    ]

    stage = counters.pipeline
    stage.update_counter("manifest/leaves", 1)
    stage.update_counter("manifest/token_shards", len(tok))
    stage.update_counter("manifest/embed_shards", len(emb))
    stage.update_counter("manifest/embed_missing", len(set(tok) - set(emb)))
    stage.update_counter("manifest/embed_extra", len(set(emb) - set(tok)))

    return {
        "source_key": pair["source_key"],
        "source": pair["source"],
        "subset": pair["subset"],
        "tokens_rel": pair["tokens_rel"],
        "embed_rel": pair["embed_rel"],
        "tokens_dir": pair["tokens_dir"],
        "embed_dir": pair["embed_dir"],
        "num_shards": num_shards,
        "n_token_shards": len(tok),
        "n_embed_shards": len(emb),
        "embed_missing": sorted(set(tok) - set(emb)),
        "embed_extra": sorted(set(emb) - set(tok)),
        "shard_count_mismatch": len(tok) != len(emb),
        "declared_totals": sorted({v[2] for v in tok.values()} | {v[2] for v in emb.values()}),
        "list_time": list_time,
        "footer_time": footer_time,
        "shards": shards,
    }


def build_manifest(leaves: list[dict]) -> pl.DataFrame:
    rows = {
        c: []
        for c in (
            "source_key source subset tokens_leaf embed_leaf shard_index num_shards "
            "tokens_path embed_path output_path tokens_bytes embed_bytes embed_exists "
            "embed_rows n_docs"
        ).split()
    }
    for leaf in leaves:
        tokens_leaf = leaf["tokens_rel"]
        embed_leaf = leaf["embed_rel"]
        for k, tok_base, tok_size, emb_base, emb_size, emb_rows in leaf["shards"]:
            exists = bool(emb_base)
            basename = emb_base or tok_base
            rows["source_key"].append(leaf["source_key"])
            rows["source"].append(leaf["source"])
            rows["subset"].append(leaf["subset"])
            rows["tokens_leaf"].append(tokens_leaf)
            rows["embed_leaf"].append(embed_leaf)
            rows["shard_index"].append(k)
            rows["num_shards"].append(leaf["num_shards"])
            rows["tokens_path"].append(f"s3://{leaf['tokens_dir']}/{tok_base}")
            rows["embed_path"].append(f"s3://{leaf['embed_dir']}/{basename}")
            rows["output_path"].append(f"{SCORES_ROOT}/{tokens_leaf}/{tok_base}")
            rows["tokens_bytes"].append(tok_size)
            rows["embed_bytes"].append(emb_size)
            rows["embed_exists"].append(exists)
            rows["embed_rows"].append(emb_rows)
            rows["n_docs"].append(max(emb_rows, 0))
    return pl.DataFrame(rows).sort(["source_key", "shard_index"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-pairs", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    fs = _fs()
    t_start = time.monotonic()

    pairs, disc = discover()
    logger.info("discovery: %s", json.dumps({k: v for k, v in disc.items() if not isinstance(v, list) or v}))
    if not pairs:
        raise RuntimeError("no Nemotron/harrier source_key pairs found")
    if len(pairs) != EXPECTED_LEAVES:
        logger.warning("expected %d paired leaves, found %d", EXPECTED_LEAVES, len(pairs))

    validation = {"skipped": True}
    if not args.skip_validation:
        validation = validate_containment(pairs, args.validate_pairs)
        logger.info("validation: %s", json.dumps({k: v for k, v in validation.items() if k != "checks"}))
        if not validation["all_contained"]:
            bad = [c for c in validation["checks"] if c.get("contained") is False]
            raise RuntimeError(f"SHARD-K CONTAINMENT FAILED on {len(bad)} shard(s): {bad[:2]}")
        if not validation["all_sorted"]:
            raise RuntimeError("id columns are not ascending within a shard; merge join invalid")

    ds = Dataset.from_list(pairs).map(sweep_leaf)
    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        max_workers=args.max_workers,
        name="quality-scores-manifest",
    )
    t0 = time.monotonic()
    outcome = ctx.execute(ds, verbose=True)
    sweep_wall = time.monotonic() - t0
    leaves = sorted(outcome.results, key=lambda leaf: leaf["source_key"])
    logger.info("swept %d leaves in %.1fs", len(leaves), sweep_wall)

    df = build_manifest(leaves)
    buf = BytesIO()
    df.write_parquet(buf)
    payload = buf.getvalue()
    fs.pipe_file(MANIFEST_URL.removeprefix("s3://"), payload)

    missing_embed = int((~df["embed_exists"]).sum())
    zero_row = int(((df["embed_rows"] == 0) & df["embed_exists"]).sum())
    failed_footers = int((df["embed_rows"] < 0).sum())
    summary = {
        "manifest_path": MANIFEST_URL,
        "manifest_bytes": len(payload),
        "total_source_keys": len(leaves),
        "total_tasks": int(df.height),
        "total_docs": int(df["n_docs"].sum()),
        "total_tokens_bytes": int(df["tokens_bytes"].sum()),
        "total_embed_bytes": int(df["embed_bytes"].sum()),
        "total_tokens_tib": int(df["tokens_bytes"].sum()) / 2**40,
        "total_embed_tib": int(df["embed_bytes"].sum()) / 2**40,
        "shards_missing_embed": missing_embed,
        "shards_zero_row_embed": zero_row,
        "shards_footer_read_failed": failed_footers,
        "leaves_with_shard_count_mismatch": [leaf["source_key"] for leaf in leaves if leaf["shard_count_mismatch"]],
        "leaves_with_embed_extra_shards": [leaf["source_key"] for leaf in leaves if leaf["embed_extra"]],
        "leaves_with_inconsistent_declared_totals": [
            leaf["source_key"] for leaf in leaves if len(leaf["declared_totals"]) != 1
        ],
        "max_shards_in_leaf": int(df["num_shards"].max()),
        "min_shards_in_leaf": int(df["num_shards"].min()),
        "discovery": disc,
        "validation_summary": {k: v for k, v in validation.items() if k != "checks"},
        "validation_path": VALIDATION_URL,
        "sweep_wall": sweep_wall,
        "total_wall": time.monotonic() - t_start,
        "counters": {k: v for k, v in sorted(outcome.counters.items())},
        "per_source_key": [
            {
                "source_key": leaf["source_key"],
                "source": leaf["source"],
                "subset": leaf["subset"],
                "num_shards": leaf["num_shards"],
                "n_token_shards": leaf["n_token_shards"],
                "n_embed_shards": leaf["n_embed_shards"],
                "n_embed_missing": len(leaf["embed_missing"]),
            }
            for leaf in leaves
        ],
    }
    fs.pipe_file(SUMMARY_URL.removeprefix("s3://"), json.dumps(summary, indent=2).encode())
    fs.pipe_file(SUCCESS_URL.removeprefix("s3://"), b"")
    logger.info(
        "manifest written: %s",
        json.dumps({k: v for k, v in summary.items() if k not in ("per_source_key", "discovery", "counters")}, indent=2),
    )


if __name__ == "__main__":
    main()
