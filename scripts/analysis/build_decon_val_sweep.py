# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fill in the intermediate-cutoff decon val sets for the val-loss-vs-Jaccard sweep.

The canonical build (``build_decon_val_sets.py``) materialized three paranoid
short-doc val caches at Jaccard cutoffs 0.50 / 0.75 / 0.90. This driver fills
the 0.05-spaced grid between them — j055, j060, j065, j070, j080, j085 — so a
checkpoint's math val loss can be plotted against a finely swept
decontamination threshold (one curve per compute budget).

Same "paranoid" filter per cutoff: keep a val doc iff it is fully contained in
validation windows (no window-split spill into train) AND its max verified
train Jaccard (union of the 3 / 4plus / 4plus_mind scans) is < cutoff. Every
cutoff <= 0.90 keep-set is a subset of the j090 universe, so all keep lists are
derived from the existing j090 keep-id list plus the doc-offsets replay array
(for exact token counts) — no rescan. The three already-built caches are reused
as-is; only the six new cutoffs are tokenized, into the same cache root.

Self-verification: the derivation recomputes the five published anchor counts
(0.50/0.70/0.75/0.80/0.90) from the universe and hard-asserts doc + token
counts before building anything, so a drifted universe or offsets array fails
loudly.

Resume + parallelism: each cutoff is built independently and is skipped if its
cache is already complete (``.stats.json`` matches the expected doc/token
counts); a partial cache from a preempted attempt is deleted and rebuilt. The
six small tokenize passes are single-worker (small sets bundle into one file
group), so the fast path is one job per cutoff with ``--skip-filter`` (the
filter is shared and only needs to run once):

    # one-shot full build (filter once, then all six cutoffs sequentially)
    python scripts/analysis/build_decon_val_sweep.py

    # fast path: filter once, then one job per remaining cutoff in parallel
    python scripts/analysis/build_decon_val_sweep.py --cutoffs 0.6 --skip-filter
    python scripts/analysis/build_decon_val_sweep.py --cutoffs 0.65 --skip-filter
    ...

Runs ON in-region us-east5 iris CPU jobs (doc_offsets + val_docs stay in
region).
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

DECON_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets"
SWEEP_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sweep"
CACHE_ROOT = "gs://marin-us-east5/tokenized/nemotron_math_val_decon"
VAL_DOCS = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs"
UNIVERSE_KEEP_IDS = f"{DECON_ROOT}/keep_ids/keep_ids_j090.json"
DOC_OFFSETS = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/replay/nemotron_math_doc_offsets.npy"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SEQ_LEN = 4096
SCAN_SUBSETS = ["3_284x71", "4plus_284x71", "4plus_mind_284x71"]

# Full 0.05 grid. The three EXISTING cutoffs are reused from the canonical
# build; the rest are tokenized here, into the same CACHE_ROOT. Building a
# cutoff in EXISTING_CUTOFFS is refused — those caches are verified and must not
# be deleted/rebuilt by this driver.
ALL_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
EXISTING_CUTOFFS = {0.50, 0.75, 0.90}
UNIVERSE_TAG = "j090"

# Published anchor counts the derivation must reproduce before any cache is
# written. Doc counts come from the corrected 2026-06-07 paranoid matrix (all
# five locally re-verified from the universe). Token counts are asserted only
# against the three cutoffs whose caches were actually built and whose
# .stats.json token totals are therefore exact (0.50/0.75/0.90); reproducing
# those proves the doc-offsets token method, so the in-between cutoffs need no
# separate token anchor.
KNOWN_DOCS = {0.50: 13_947, 0.70: 25_868, 0.75: 28_089, 0.80: 30_038, 0.90: 33_196}
KNOWN_TOKENS = {0.50: 10_282_799, 0.75: 20_782_728, 0.90: 25_346_090}

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


def cutoff_tag(cutoff: float) -> str:
    return f"j{round(cutoff * 100):03d}"


BUILD_CUTOFFS = [c for c in ALL_CUTOFFS if c not in EXISTING_CUTOFFS]
BUILD_TAGS = [cutoff_tag(c) for c in BUILD_CUTOFFS]
STATS_SCHEMA = pa.schema([("input_shard", pa.string())] + [(f"kept_{tag}", pa.int64()) for tag in BUILD_TAGS])

_KEEP: dict[str, dict] | None = None
_KEEP_SETS: dict[str, set[str]] | None = None
_KEEP_META: dict[str, tuple[int, float | None]] | None = None


def derive_cutoff_keepsets(
    ids: list[str],
    doc_indices: np.ndarray,
    aligned_max_jaccard: np.ndarray,
    doc_tokens: np.ndarray,
    cutoffs: list[float],
) -> dict[float, dict]:
    """Pure derivation of the kept ids / doc indices / token sum per cutoff.

    ``aligned_max_jaccard[i]`` is the max train Jaccard of ``doc_indices[i]``
    (0.0 for a doc with no >=0.5 near-dup). ``doc_tokens`` is the full per-doc
    token-length array (indexed by doc index). Asserts strict nesting across
    ascending cutoffs and no duplicate ids.
    """
    result: dict[float, dict] = {}
    previous: set[str] | None = None
    for cutoff in cutoffs:
        keep_idx = np.nonzero(aligned_max_jaccard < cutoff)[0]
        keep_ids = [ids[i] for i in keep_idx]
        keep_doc_indices = doc_indices[keep_idx]
        tokens = int(doc_tokens[keep_doc_indices].sum())
        id_set = set(keep_ids)
        assert len(id_set) == len(keep_ids), f"duplicate ids at cutoff {cutoff}"
        if previous is not None:
            assert previous < id_set, f"keep sets must strictly nest (cutoff {cutoff})"
        previous = id_set
        result[cutoff] = {"ids": keep_ids, "doc_indices": keep_doc_indices, "tokens": tokens}
    return result


def _payload(cutoff: float, keep_ids: list[str], keep_doc_indices: np.ndarray, tokens: int, max_jaccard: dict) -> dict:
    tag = cutoff_tag(cutoff)
    return {
        "name": f"nemotron_math_paranoid_val_{tag}",
        "filter": "fully_contained_in_val_windows_and_max_train_jaccard_lt_cutoff",
        "cutoff": cutoff,
        "scan_subsets": SCAN_SUBSETS,
        "expected_docs": len(keep_ids),
        "expected_tokens": tokens,
        "keep_ids_xxh3": xxhash.xxh3_128_hexdigest("\n".join(sorted(keep_ids)).encode()),
        "doc_indices": keep_doc_indices.tolist(),
        "max_jaccard_by_doc": {str(int(d)): max_jaccard[int(d)] for d in keep_doc_indices if int(d) in max_jaccard},
        "ids": keep_ids,
    }


def derive_keep_payloads() -> dict[str, dict]:
    """All-cutoff keep-id payloads from the j090 universe + doc-offsets lengths.

    Hard-asserts the five published anchor doc/token counts so a drifted input
    fails before any cache is built.
    """
    with fsspec.open(UNIVERSE_KEEP_IDS) as f:
        universe = json.load(f)
    ids: list[str] = universe["ids"]
    doc_indices = np.asarray(universe["doc_indices"], dtype=np.int64)
    max_jaccard = {int(k): float(v) for k, v in universe["max_jaccard_by_doc"].items()}

    # nemotron_math_doc_offsets.npy holds doc END offsets (len == num docs,
    # offsets[-1] == total tokens); doc d spans [ends[d-1], ends[d]) with start
    # 0 for d == 0.
    with fsspec.open(DOC_OFFSETS, "rb") as f:
        ends = np.load(f).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    doc_tokens = ends - starts
    aligned_max_jaccard = np.array([max_jaccard.get(int(d), 0.0) for d in doc_indices], dtype=np.float64)

    keepsets = derive_cutoff_keepsets(ids, doc_indices, aligned_max_jaccard, doc_tokens, ALL_CUTOFFS)

    payloads: dict[str, dict] = {}
    for cutoff, keep in keepsets.items():
        keep_ids = keep["ids"]
        tokens = keep["tokens"]
        if cutoff in KNOWN_DOCS:
            assert len(keep_ids) == KNOWN_DOCS[cutoff], f"J<{cutoff}: {len(keep_ids)} docs != {KNOWN_DOCS[cutoff]}"
        if cutoff in KNOWN_TOKENS:
            assert tokens == KNOWN_TOKENS[cutoff], f"J<{cutoff}: {tokens} tokens != {KNOWN_TOKENS[cutoff]}"
        payloads[cutoff_tag(cutoff)] = _payload(cutoff, keep_ids, keep["doc_indices"], tokens, max_jaccard)
    return payloads


def load_payloads_from_keep_ids(cutoffs: list[float]) -> dict[str, dict]:
    """Read already-written, self-verified keep-id payloads (the --skip-filter path)."""
    payloads: dict[str, dict] = {}
    for cutoff in cutoffs:
        tag = cutoff_tag(cutoff)
        uri = f"{SWEEP_ROOT}/keep_ids/keep_ids_{tag}.json"
        if not fsspec_exists(uri):
            raise FileNotFoundError(f"--skip-filter set but keep-id list missing: {uri}")
        with fsspec.open(uri) as f:
            payloads[tag] = json.load(f)
    return payloads


def _load_keep_payloads() -> dict[str, dict]:
    """Per-worker cache of the build-tag + universe keep-id payloads (from GCS)."""
    global _KEEP
    if _KEEP is None:
        _KEEP = {}
        for tag in [*BUILD_TAGS, UNIVERSE_TAG]:
            with fsspec.open(f"{SWEEP_ROOT}/keep_ids/keep_ids_{tag}.json") as f:
                _KEEP[tag] = json.load(f)
    return _KEEP


def _keep_lookup() -> tuple[dict[str, set[str]], dict[str, tuple[int, float | None]]]:
    """Per-worker derived lookups: keep-id set per tag, id -> (doc_index, max_jaccard)."""
    global _KEEP_SETS, _KEEP_META
    if _KEEP_SETS is None or _KEEP_META is None:
        keep = _load_keep_payloads()
        _KEEP_SETS = {tag: set(keep[tag]["ids"]) for tag in [*BUILD_TAGS, UNIVERSE_TAG]}
        universe = keep[UNIVERSE_TAG]
        _KEEP_META = {
            doc_id: (doc_index, universe["max_jaccard_by_doc"].get(str(doc_index)))
            for doc_id, doc_index in zip(universe["ids"], universe["doc_indices"], strict=True)
        }
    return _KEEP_SETS, _KEEP_META


def _filter_shard(path: str) -> Iterator[dict]:
    """Filter one val_docs shard into the six new cutoff docs/ dirs."""
    keep_sets, meta = _keep_lookup()
    basename = path.rsplit("/", 1)[1]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text", "shard", "row"])

    rows: dict[str, list[dict]] = {tag: [] for tag in BUILD_TAGS}
    for record in table.to_pylist():
        doc_id = record["id"]
        if doc_id not in keep_sets[UNIVERSE_TAG]:
            continue
        doc_index, max_jaccard = meta[doc_id]
        record |= {"doc_index": doc_index, "max_jaccard": max_jaccard}
        for tag in BUILD_TAGS:
            if doc_id in keep_sets[tag]:
                rows[tag].append(record)

    for tag in BUILD_TAGS:
        out = f"{SWEEP_ROOT}/{tag}/docs/{basename}"
        filtered = pa.Table.from_pylist(rows[tag], schema=FILTERED_SCHEMA)
        with fsspec.open(out, "wb") as f:
            pq.write_table(filtered, f)
        counters.increment(f"filter/kept_{tag}", len(rows[tag]))

    yield {"input_shard": basename} | {f"kept_{tag}": len(rows[tag]) for tag in BUILD_TAGS}


def run_filter() -> None:
    shard_files = sorted(fsspec_glob(f"{VAL_DOCS}/*.parquet"))
    logger.info("filtering %d val_docs shards into %s", len(shard_files), SWEEP_ROOT)
    ctx = ZephyrContext(
        name="decon-val-sweep-filter",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", disk="10g", preemptible=False),
    )
    outcome = ctx.execute(
        Dataset.from_list(shard_files)
        .flat_map(_filter_shard)
        .write_parquet(
            f"{SWEEP_ROOT}/filter_stats/stats-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=STATS_SCHEMA,
            skip_existing=True,
        )
    )
    logger.info("filter counters: %s", dict(outcome.counters))


def actual_doc_count(tag: str) -> int:
    total = 0
    for path in fsspec_glob(f"{SWEEP_ROOT}/{tag}/docs/*.parquet"):
        with fsspec.open(path, "rb") as f:
            total += pq.read_metadata(f).num_rows
    return total


def cache_stats(tag: str) -> dict:
    with fsspec.open(f"{CACHE_ROOT}/{tag}/validation/.stats.json") as f:
        return json.load(f)


def cache_status(tag: str, expected: dict) -> str:
    """complete (stats match) / mismatch / partial (cache dir, no stats) / absent."""
    if fsspec_exists(f"{CACHE_ROOT}/{tag}/validation/.stats.json"):
        stats = cache_stats(tag)
        if (
            stats.get("total_elements") == expected["expected_docs"]
            and stats.get("total_tokens") == expected["expected_tokens"]
        ):
            return "complete"
        return "mismatch"
    if fsspec_glob(f"{CACHE_ROOT}/{tag}/validation/*"):
        return "partial"
    return "absent"


def delete_cache(tag: str) -> None:
    path = f"{CACHE_ROOT}/{tag}/validation"
    fs, _ = fsspec.core.url_to_fs(path)
    if fs.exists(path):
        fs.rm(path, recursive=True)


def build_one_cutoff(tag: str, expected: dict, *, force: bool) -> None:
    """Resume-safe tokenize of one cutoff: skip if complete, rebuild if partial."""
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
            validation_paths=[f"{SWEEP_ROOT}/{tag}/docs/*.parquet"],
            cache_path=f"{CACHE_ROOT}/{tag}",
            tokenizer=TOKENIZER,
        )
    )
    stats = cache_stats(tag)
    assert stats["total_elements"] == expected["expected_docs"], f"{tag}: cache rows {stats}"
    assert (
        stats["total_tokens"] == expected["expected_tokens"]
    ), f"{tag}: cache tokens {stats['total_tokens']} != expected {expected['expected_tokens']}"

    manifest = {
        "name": f"nemotron_math_paranoid_val_{tag}",
        "definition": "paranoid short-doc val set: fully contained in val windows AND max train Jaccard < cutoff",
        **{k: expected[k] for k in ("cutoff", "filter", "expected_docs", "expected_tokens", "keep_ids_xxh3")},
        "actual_docs": docs,
        "actual_tokens": stats["total_tokens"],
        "eval_sequences": stats["total_tokens"] // SEQ_LEN,
        "sources": {
            "val_docs": VAL_DOCS,
            "universe_keep_ids": UNIVERSE_KEEP_IDS,
            "doc_offsets": DOC_OFFSETS,
            "scan_verified_pairs": [
                f"gs://marin-us-east5/scratch/ahmed/midtrain_dedup/{sub}/verified_pairs" for sub in SCAN_SUBSETS
            ],
        },
        "tokenizer": TOKENIZER,
        "cache_path": f"{CACHE_ROOT}/{tag}",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with fsspec.open(f"{SWEEP_ROOT}/{tag}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("%s ready: %d docs, %d tokens at %s", tag, docs, stats["total_tokens"], manifest["cache_path"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoffs", default=None, help="Comma-separated cutoffs to build; default = the six new cutoffs."
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Docs already filtered; only (re)tokenize. Reads keep-id lists from GCS.",
    )
    parser.add_argument("--force", action="store_true", help="Delete and rebuild even a complete cache.")
    args = parser.parse_args()

    cutoffs = [float(c) for c in args.cutoffs.split(",")] if args.cutoffs else list(BUILD_CUTOFFS)
    invalid = [c for c in cutoffs if c not in ALL_CUTOFFS or c in EXISTING_CUTOFFS]
    if invalid:
        raise ValueError(
            f"refusing to build {invalid}: not a new sweep cutoff (existing caches must not be rebuilt here)"
        )

    if args.skip_filter:
        payloads = load_payloads_from_keep_ids(cutoffs)
        logger.info("skip-filter: loaded keep-id payloads for %s", [cutoff_tag(c) for c in cutoffs])
    else:
        payloads = derive_keep_payloads()
        logger.info("derived %d cutoff payloads (self-verified)", len(payloads))
        for tag, payload in payloads.items():
            with fsspec.open(f"{SWEEP_ROOT}/keep_ids/keep_ids_{tag}.json", "w") as f:
                json.dump(payload, f)
        run_filter()

    for cutoff in cutoffs:
        tag = cutoff_tag(cutoff)
        build_one_cutoff(tag, payloads[tag], force=args.force)


if __name__ == "__main__":
    main()
