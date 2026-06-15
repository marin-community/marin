# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build paranoid short-doc math val sets decontaminated against `4plus` ONLY.

Why this exists
---------------
The canonical decon val caches (`nemotron_math_val_decon/j0{50..90}`) threshold
each val doc on its max verified train Jaccard across the **union** of the three
math-subset scans (`3`, `4plus`, `4plus_mind`). But the p33m67 / p50m50 / p67m33
midtrain runs train on **`4plus` only** as their math source (`3` and
`4plus_mind` appear in no mix — byte-identical val contract, only `train_weights`
differ). Thresholding on the union therefore **over-decontaminates**: it drops
val docs that merely resemble the *untrained* `3` / `4plus_mind` and are clean
for these runs (~6% of dropped docs; logbook 2026-06-11). A val set is only
"clean" relative to a specific training mix.

This driver rebuilds the same paranoid sets with the per-doc max Jaccard derived
from the **`4plus_284x71` verified_pairs alone**. Same paranoid filter (fully
contained in val windows AND max `4plus` train Jaccard < cutoff). Same 0.05 grid
(j050…j090). Fresh output roots so the union caches are untouched.

No rescan: reuses the existing `4plus` verified_pairs and the replay arrays
(doc offsets, val window indices, the 45M-doc id table). Runs as ONE in-region
us-east5 iris CPU job (all heavy inputs live in us-east5).

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 48GB --disk 80GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name decon-val-4plus-build \
        -- python scripts/analysis/build_decon_val_4plus_only.py

Verification gates (fail loud, never clobber):
- recomputed fully-contained set == 33,790 docs / 25,956,642 tokens (corrected
  2026-06-07 paranoid baseline);
- every union-j090 keep id is in the recomputed fully-contained id set (validates
  the fully-contained recomputation + the all_ids id formatting end-to-end);
- for every union-j090 doc, 4plus_max <= union_max (4plus pairs are a subset of
  the union — validates the verified_pairs aggregation + val_id format);
- per cutoff, 4plus-only keep >= union keep (4plus_max <= union_max => fewer
  drops => more kept);
- each tokenized cache's `.stats.json` doc + token counts match the derivation
  exactly before its manifest is written.
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from fray import ResourceConfig
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.utils import fsspec_exists, fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

MIDTRAIN = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
DECON4_ROOT = f"{MIDTRAIN}/decon_val_4plus"
CACHE4_ROOT = "gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus"
VAL_DOCS = f"{MIDTRAIN}/val_docs"
REPLAY = f"{MIDTRAIN}/replay"
ALL_IDS = f"{REPLAY}/nemotron_math_all_ids.npy"
DOC_OFFSETS = f"{REPLAY}/nemotron_math_doc_offsets.npy"
VAL_WINDOW_INDICES = f"{REPLAY}/nemotron_math_val_window_indices.npy"
VERIFIED_4PLUS = f"{MIDTRAIN}/4plus_284x71/verified_pairs"
UNION_J090 = f"{MIDTRAIN}/decon_val_sets/keep_ids/keep_ids_j090.json"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SEQ_LEN = 4096
WINDOW = 4096
ALL_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Corrected 2026-06-07 paranoid baseline (fully contained in val windows); these
# are decon-source-independent (the filter, not the Jaccard cutoff).
FULLY_CONTAINED_DOCS = 33_790
FULLY_CONTAINED_TOKENS = 25_956_642
UNION_J090_DOCS = 33_196  # cross-check anchor

FILTERED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("shard", pa.string()),
        ("row", pa.int64()),
        ("doc_index", pa.int64()),
        ("max_jaccard", pa.float64()),
    ]
)

_KEEP: dict[str, dict] | None = None
_KEEP_SETS: dict[str, set[str]] | None = None
_KEEP_META: dict[str, tuple[int, float]] | None = None


def cutoff_tag(cutoff: float) -> str:
    return f"j{round(cutoff * 100):03d}"


ALL_TAGS = [cutoff_tag(c) for c in ALL_CUTOFFS]
UNIVERSE_TAG = cutoff_tag(max(ALL_CUTOFFS))  # j090 — every cutoff <= 0.90 is a subset
STATS_SCHEMA = pa.schema([("input_shard", pa.string())] + [(f"kept_{tag}", pa.int64()) for tag in ALL_TAGS])


# ----------------------------------------------------------------------------
# Driver-side derivation (runs on the main container; reads the big replay arrays)
# ----------------------------------------------------------------------------
def _load_npy(uri: str) -> np.ndarray:
    with fsspec.open(uri, "rb") as f:
        return np.load(f)


def compute_fully_contained(ends: np.ndarray, val_windows: np.ndarray) -> np.ndarray:
    """Doc indices whose entire token span lies inside validation windows.

    ``ends`` is the per-doc END offset array (``ends[d]`` = exclusive end of doc
    d; doc d spans ``[ends[d-1], ends[d])`` with start 0 for d == 0). A doc is
    fully contained iff every 4096-token window it touches is a validation
    window — exact and vectorized via a cumulative count of val windows.
    """
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    first_win = starts // WINDOW
    last_win = (ends - 1) // WINDOW
    n_win = int(last_win.max()) + 1
    is_val = np.zeros(n_win, dtype=bool)
    is_val[val_windows] = True
    # cum[w] = number of val windows in [0, w-1]; val windows in [a, b] inclusive
    # is cum[b+1] - cum[a].
    cum = np.concatenate([[0], np.cumsum(is_val)]).astype(np.int64)
    touched = last_win - first_win + 1
    val_in_range = cum[last_win + 1] - cum[first_win]
    fully_mask = val_in_range == touched
    return np.nonzero(fully_mask)[0].astype(np.int64)


def max_jaccard_4plus_by_id() -> dict[str, float]:
    """val_id -> max verified Jaccard against `4plus` (the >=0.5 reported pairs)."""
    out: dict[str, float] = {}
    for path in fsspec_glob(f"{VERIFIED_4PLUS}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["val_id", "jaccard"])
        for vid, jac in zip(t.column("val_id").to_pylist(), t.column("jaccard").to_pylist(), strict=True):
            if jac > out.get(vid, 0.0):
                out[vid] = jac
    return out


def format_ids(rows: np.ndarray) -> list[str]:
    """(hi, lo) uint64 rows -> 32-hex xxh3_128 ids (the normalized-parquet id)."""
    return [f"{int(hi):016x}{int(lo):016x}" for hi, lo in rows]


def derive_payloads() -> dict[str, dict]:
    """Build the 4plus-only keep-id payloads for every cutoff, with hard gates."""
    ends = _load_npy(DOC_OFFSETS).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    doc_tokens = ends - starts
    val_windows = _load_npy(VAL_WINDOW_INDICES).astype(np.int64)

    fully = compute_fully_contained(ends, val_windows)
    fc_tokens = int(doc_tokens[fully].sum())
    assert len(fully) == FULLY_CONTAINED_DOCS, f"fully-contained {len(fully)} != {FULLY_CONTAINED_DOCS}"
    assert fc_tokens == FULLY_CONTAINED_TOKENS, f"fully-contained tokens {fc_tokens} != {FULLY_CONTAINED_TOKENS}"
    logger.info("fully-contained: %d docs / %d tokens (gate passed)", len(fully), fc_tokens)

    all_ids = _load_npy(ALL_IDS)
    fully_ids = format_ids(np.asarray(all_ids[fully]))
    id_by_doc = dict(zip(fully.tolist(), fully_ids, strict=True))

    j4 = max_jaccard_4plus_by_id()
    aligned = np.array([j4.get(id_by_doc[int(d)], 0.0) for d in fully], dtype=np.float64)
    n_hit = int((aligned >= 0.5).sum())
    logger.info(
        "4plus pairs: %d val_ids with a 4plus near-dup; within fully-contained %d docs have 4plus_max>=0.5 (max %.4f)",
        len(j4),
        n_hit,
        float(aligned.max()),
    )
    assert n_hit > 5_000, f"implausibly few 4plus near-dups in fully-contained set ({n_hit}) — check val_id format"
    # Short fully-contained docs rarely reach J>=0.99 (a small formatting diff moves
    # a short doc's Jaccard a lot); >=0.9 confirms the verified_pairs join produced
    # real high-similarity values rather than an all-near-zero failure.
    assert aligned.max() >= 0.9, "no high-Jaccard 4plus pair present — check verified_pairs join"

    # Cross-check against the union j090 universe.
    with fsspec.open(UNION_J090) as f:
        union = json.load(f)
    union_ids = set(union["ids"])
    union_max = {int(k): float(v) for k, v in union["max_jaccard_by_doc"].items()}
    fc_id_set = set(fully_ids)
    assert len(union_ids) == UNION_J090_DOCS, f"union j090 has {len(union_ids)} ids != {UNION_J090_DOCS}"
    assert union_ids <= fc_id_set, "union j090 keep ids are not a subset of recomputed fully-contained ids"
    # 4plus_max <= union_max for every union doc (4plus pairs subset of union pairs).
    fourplus_by_doc = {int(d): aligned[i] for i, d in enumerate(fully)}
    bad = [d for d, um in union_max.items() if fourplus_by_doc.get(d, 0.0) > um + 1e-9]
    assert not bad, f"{len(bad)} docs have 4plus_max > union_max (e.g. {bad[:3]}) — aggregation/id-format bug"
    logger.info("cross-checks vs union j090 passed (subset + 4plus<=union on %d docs)", len(union_max))

    payloads: dict[str, dict] = {}
    previous: set[str] | None = None
    # Sanity bound: 4plus keep >= union keep (per-doc 4plus_max <= union_max).
    union_universe = {int(d) for d in union["doc_indices"]}
    union_keep_per_cutoff = _union_keep_counts(union_max, union_universe, fully)
    # The helper must reproduce the published union keep counts exactly.
    union_anchor = {0.50: 13_947, 0.75: 28_089, 0.90: 33_196}
    for c, expected in union_anchor.items():
        got = union_keep_per_cutoff[c]
        assert got == expected, f"union keep recompute {got} != published {expected} at {c}"
    for cutoff in ALL_CUTOFFS:
        keep_pos = np.nonzero(aligned < cutoff)[0]
        keep_docs = fully[keep_pos]
        keep_ids = [fully_ids[i] for i in keep_pos]
        tokens = int(doc_tokens[keep_docs].sum())
        id_set = set(keep_ids)
        assert len(id_set) == len(keep_ids), f"duplicate ids at cutoff {cutoff}"
        if previous is not None:
            assert previous < id_set, f"keep sets must strictly nest (cutoff {cutoff})"
        previous = id_set
        u = union_keep_per_cutoff[cutoff]
        assert len(keep_ids) >= u, f"4plus keep {len(keep_ids)} < union keep {u} at {cutoff} (impossible)"
        logger.info(
            "cutoff %.2f: keep %d docs / %d tokens (union kept %d; +%d added back)",
            cutoff,
            len(keep_ids),
            tokens,
            u,
            len(keep_ids) - u,
        )
        tag = cutoff_tag(cutoff)
        payloads[tag] = {
            "name": f"nemotron_math_paranoid_val_4plus_{tag}",
            "filter": "fully_contained_in_val_windows_and_max_4plus_train_jaccard_lt_cutoff",
            "decon_against": "4plus_only",
            "cutoff": cutoff,
            "scan_subsets": ["4plus_284x71"],
            "expected_docs": len(keep_ids),
            "expected_tokens": tokens,
            "keep_ids_xxh3": xxhash.xxh3_128_hexdigest("\n".join(sorted(keep_ids)).encode()),
            "doc_indices": keep_docs.tolist(),
            "max_jaccard_by_doc": {str(int(d)): float(fourplus_by_doc[int(d)]) for d in keep_docs},
            "ids": keep_ids,
        }
    return payloads


def _union_keep_counts(union_max: dict[int, float], union_universe: set[int], fully: np.ndarray) -> dict[float, int]:
    """How many fully-contained docs the UNION decon keeps per cutoff (sanity bound).

    The j090 universe file lists docs with union_max < 0.90 (``union_universe``)
    and a ``max_jaccard_by_doc`` covering only those with a union pair >=0.5. A
    fully-contained doc's effective union Jaccard is therefore:
    its listed value if present; 0.0 if in the universe but pair-free (clean);
    else 1.0 (it has union_max >= 0.90 and is dropped at every cutoff <= 0.90).
    """

    def union_jaccard(doc: int) -> float:
        if doc in union_max:
            return union_max[doc]
        return 0.0 if doc in union_universe else 1.0

    um = np.array([union_jaccard(int(d)) for d in fully], dtype=np.float64)
    return {c: int((um < c).sum()) for c in ALL_CUTOFFS}


# ----------------------------------------------------------------------------
# Worker-side filter (ships by value — keep everything it needs in this module)
# ----------------------------------------------------------------------------
def _load_keep_payloads() -> dict[str, dict]:
    global _KEEP
    if _KEEP is None:
        _KEEP = {}
        for tag in ALL_TAGS:
            with fsspec.open(f"{DECON4_ROOT}/keep_ids/keep_ids_{tag}.json") as f:
                _KEEP[tag] = json.load(f)
    return _KEEP


def _keep_lookup() -> tuple[dict[str, set[str]], dict[str, tuple[int, float]]]:
    global _KEEP_SETS, _KEEP_META
    if _KEEP_SETS is None or _KEEP_META is None:
        keep = _load_keep_payloads()
        _KEEP_SETS = {tag: set(keep[tag]["ids"]) for tag in ALL_TAGS}
        universe = keep[UNIVERSE_TAG]
        max_by_doc = universe["max_jaccard_by_doc"]
        _KEEP_META = {
            doc_id: (doc_index, float(max_by_doc.get(str(doc_index), 0.0)))
            for doc_id, doc_index in zip(universe["ids"], universe["doc_indices"], strict=True)
        }
    return _KEEP_SETS, _KEEP_META


def _filter_shard(path: str) -> Iterator[dict]:
    keep_sets, meta = _keep_lookup()
    basename = path.rsplit("/", 1)[1]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text", "shard", "row"])

    rows: dict[str, list[dict]] = {tag: [] for tag in ALL_TAGS}
    for record in table.to_pylist():
        doc_id = record["id"]
        if doc_id not in keep_sets[UNIVERSE_TAG]:
            continue
        doc_index, max_jaccard = meta[doc_id]
        record |= {"doc_index": doc_index, "max_jaccard": max_jaccard}
        for tag in ALL_TAGS:
            if doc_id in keep_sets[tag]:
                rows[tag].append(record)

    for tag in ALL_TAGS:
        out = f"{DECON4_ROOT}/{tag}/docs/{basename}"
        filtered = pa.Table.from_pylist(rows[tag], schema=FILTERED_SCHEMA)
        with fsspec.open(out, "wb") as f:
            pq.write_table(filtered, f)
        counters.increment(f"filter/kept_{tag}", len(rows[tag]))

    yield {"input_shard": basename} | {f"kept_{tag}": len(rows[tag]) for tag in ALL_TAGS}


def run_filter() -> None:
    shard_files = sorted(fsspec_glob(f"{VAL_DOCS}/*.parquet"))
    logger.info("filtering %d val_docs shards into %s", len(shard_files), DECON4_ROOT)
    ctx = ZephyrContext(
        name="decon-val-4plus-filter",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", disk="10g", preemptible=False),
    )
    outcome = ctx.execute(
        Dataset.from_list(shard_files)
        .flat_map(_filter_shard)
        .write_parquet(
            f"{DECON4_ROOT}/filter_stats/stats-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=STATS_SCHEMA,
            skip_existing=True,
        )
    )
    logger.info("filter counters: %s", dict(outcome.counters))


# ----------------------------------------------------------------------------
# Cache build (one tokenize pass per cutoff; resume-safe, token-exact gated)
# ----------------------------------------------------------------------------
def actual_doc_count(tag: str) -> int:
    total = 0
    for path in fsspec_glob(f"{DECON4_ROOT}/{tag}/docs/*.parquet"):
        with fsspec.open(path, "rb") as f:
            total += pq.read_metadata(f).num_rows
    return total


def cache_stats(tag: str) -> dict:
    with fsspec.open(f"{CACHE4_ROOT}/{tag}/validation/.stats.json") as f:
        return json.load(f)


def cache_status(tag: str, expected: dict) -> str:
    if fsspec_exists(f"{CACHE4_ROOT}/{tag}/validation/.stats.json"):
        stats = cache_stats(tag)
        if (
            stats.get("total_elements") == expected["expected_docs"]
            and stats.get("total_tokens") == expected["expected_tokens"]
        ):
            return "complete"
        return "mismatch"
    if fsspec_glob(f"{CACHE4_ROOT}/{tag}/validation/*"):
        return "partial"
    return "absent"


def delete_cache(tag: str) -> None:
    path = f"{CACHE4_ROOT}/{tag}/validation"
    fs, _ = fsspec.core.url_to_fs(path)
    if fs.exists(path):
        fs.rm(path, recursive=True)


def build_one_cutoff(tag: str, expected: dict, *, force: bool) -> None:
    status = "absent" if force else cache_status(tag, expected)
    if status == "complete":
        logger.info(
            "%s already complete (%d docs, %d tokens); skipping",
            tag,
            expected["expected_docs"],
            expected["expected_tokens"],
        )
        return
    if status == "mismatch":
        raise RuntimeError(
            f"{tag} cache exists but counts disagree with expected {expected}; investigate, do not clobber"
        )
    if status == "partial" or force:
        logger.warning("%s cache is partial/forced; deleting and rebuilding", tag)
        delete_cache(tag)

    docs = actual_doc_count(tag)
    assert docs == expected["expected_docs"], f"{tag}: filtered {docs} docs != {expected['expected_docs']}"
    tokenize(
        TokenizeConfig(
            train_paths=[],
            validation_paths=[f"{DECON4_ROOT}/{tag}/docs/*.parquet"],
            cache_path=f"{CACHE4_ROOT}/{tag}",
            tokenizer=TOKENIZER,
        )
    )
    stats = cache_stats(tag)
    assert stats["total_elements"] == expected["expected_docs"], f"{tag}: cache rows {stats}"
    assert (
        stats["total_tokens"] == expected["expected_tokens"]
    ), f"{tag}: cache tokens {stats['total_tokens']} != {expected['expected_tokens']}"

    manifest = {
        "name": f"nemotron_math_paranoid_val_4plus_{tag}",
        "definition": (
            "paranoid short-doc val set, decontaminated against 4plus ONLY: fully contained in "
            "val windows AND max 4plus train Jaccard < cutoff"
        ),
        **{
            k: expected[k]
            for k in ("cutoff", "filter", "decon_against", "expected_docs", "expected_tokens", "keep_ids_xxh3")
        },
        "actual_docs": docs,
        "actual_tokens": stats["total_tokens"],
        "eval_sequences": stats["total_tokens"] // SEQ_LEN,
        "sources": {
            "val_docs": VAL_DOCS,
            "doc_offsets": DOC_OFFSETS,
            "val_window_indices": VAL_WINDOW_INDICES,
            "all_ids": ALL_IDS,
            "scan_verified_pairs": [VERIFIED_4PLUS],
        },
        "tokenizer": TOKENIZER,
        "cache_path": f"{CACHE4_ROOT}/{tag}",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with fsspec.open(f"{DECON4_ROOT}/{tag}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("%s ready: %d docs, %d tokens at %s", tag, docs, stats["total_tokens"], manifest["cache_path"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", default=None, help="Comma-separated cutoffs to build; default = all nine.")
    parser.add_argument("--skip-filter", action="store_true", help="Docs already filtered; only (re)tokenize.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild even a complete cache.")
    args = parser.parse_args()

    cutoffs = [float(c) for c in args.cutoffs.split(",")] if args.cutoffs else list(ALL_CUTOFFS)
    invalid = [c for c in cutoffs if c not in ALL_CUTOFFS]
    if invalid:
        raise ValueError(f"unknown cutoffs {invalid}; valid: {ALL_CUTOFFS}")

    if args.skip_filter:
        payloads = {}
        for cutoff in cutoffs:
            tag = cutoff_tag(cutoff)
            with fsspec.open(f"{DECON4_ROOT}/keep_ids/keep_ids_{tag}.json") as f:
                payloads[tag] = json.load(f)
        logger.info("skip-filter: loaded keep-id payloads for %s", [cutoff_tag(c) for c in cutoffs])
    else:
        payloads = derive_payloads()
        for tag, payload in payloads.items():
            with fsspec.open(f"{DECON4_ROOT}/keep_ids/keep_ids_{tag}.json", "w") as f:
                json.dump(payload, f)
        logger.info("wrote %d keep-id payloads to %s/keep_ids", len(payloads), DECON4_ROOT)
        run_filter()

    for cutoff in cutoffs:
        tag = cutoff_tag(cutoff)
        build_one_cutoff(tag, payloads[tag], force=args.force)


if __name__ == "__main__":
    main()
