# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the paranoid short-doc math val sets (J>=0.5 / 0.75 / 0.9 cutoffs).

For each cutoff, keep a val doc iff it is fully contained in validation
windows (no window-split spill into train) AND its max verified train Jaccard
across the 3 / 4plus / 4plus_mind scans is < cutoff. Keep-id lists come from
scripts/analysis/build_paranoid_val_keep_ids.py. Docs are short by
construction (must fit one 4096-token val window).

Stages, one driver run:
1. One zephyr pass over the 231 val_docs shards filters into three docs/
   dirs (id, text, shard, row, doc_index, max_jaccard).
2. Three validation-only tokenize() calls write Levanter caches; eval configs
   point at the cache root (data lands under <root>/validation).
3. Per-cutoff manifest.json with expected vs actual doc/token counts.

Overwrite policy: hard-fail if any target exists. --resume continues a
matching build (intent.json must agree on keep-id hashes, sources, and
expected counts); anything else requires explicitly deleting the targets.
Submit WITH --resume so iris preemption retries can continue their own build.

Launch (us-east5; all reads and writes in-region):

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 4 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name decon-val-build-all \
        -- python scripts/analysis/build_decon_val_sets.py --resume
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.utils import fsspec_exists, fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

DECON_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets"
CACHE_ROOT = "gs://marin-us-east5/tokenized/nemotron_math_val_decon"
VAL_DOCS = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs"
INTENT_URI = f"{DECON_ROOT}/build_intent.json"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
CUTOFF_TAGS = ("j050", "j075", "j090")
SEQ_LEN = 4096

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
STATS_SCHEMA = pa.schema([("input_shard", pa.string())] + [(f"kept_{tag}", pa.int64()) for tag in CUTOFF_TAGS])

_KEEP: dict[str, dict] | None = None
_KEEP_SETS: dict[str, set[str]] | None = None
_KEEP_META: dict[str, tuple[int, float | None]] | None = None


def _load_keep_payloads() -> dict[str, dict]:
    """Per-worker cache of the three keep-id payloads (loaded from GCS, not closures)."""
    global _KEEP
    if _KEEP is None:
        _KEEP = {}
        for tag in CUTOFF_TAGS:
            with fsspec.open(f"{DECON_ROOT}/keep_ids/keep_ids_{tag}.json") as f:
                _KEEP[tag] = json.load(f)
    return _KEEP


def _keep_lookup() -> tuple[dict[str, set[str]], dict[str, tuple[int, float | None]]]:
    """Per-worker derived lookups: keep-id set per tag, id -> (doc_index, max_jaccard)."""
    global _KEEP_SETS, _KEEP_META
    if _KEEP_SETS is None or _KEEP_META is None:
        keep = _load_keep_payloads()
        _KEEP_SETS = {tag: set(p["ids"]) for tag, p in keep.items()}
        widest = keep["j090"]
        _KEEP_META = {
            doc_id: (doc_index, widest["max_jaccard_by_doc"].get(str(doc_index)))
            for doc_id, doc_index in zip(widest["ids"], widest["doc_indices"], strict=True)
        }
    return _KEEP_SETS, _KEEP_META


def build_intent(keep_payloads: dict[str, dict]) -> dict:
    """The identity of this build; --resume requires an exact match."""
    return {
        "val_docs": VAL_DOCS,
        "cache_root": CACHE_ROOT,
        "tokenizer": TOKENIZER,
        "cutoffs": {
            tag: {
                "cutoff": p["cutoff"],
                "filter": p["filter"],
                "keep_ids_xxh3": p["keep_ids_xxh3"],
                "expected_docs": p["expected_docs"],
                "expected_tokens": p["expected_tokens"],
            }
            for tag, p in keep_payloads.items()
        },
    }


def existing_targets() -> list[str]:
    """Targets that already exist; non-empty without --resume is fatal."""
    found = [INTENT_URI] if fsspec_exists(INTENT_URI) else []
    if fsspec_glob(f"{DECON_ROOT}/filter_stats/*.parquet"):
        found.append(f"{DECON_ROOT}/filter_stats/")
    for tag in CUTOFF_TAGS:
        if fsspec_glob(f"{DECON_ROOT}/{tag}/docs/*.parquet"):
            found.append(f"{DECON_ROOT}/{tag}/docs/")
        if fsspec_exists(f"{DECON_ROOT}/{tag}/manifest.json"):
            found.append(f"{DECON_ROOT}/{tag}/manifest.json")
        if fsspec_glob(f"{CACHE_ROOT}/{tag}/validation/*"):
            found.append(f"{CACHE_ROOT}/{tag}/validation/")
    return found


def check_resume_allowed(existing_intent: dict | None, current_intent: dict) -> None:
    """Raise unless an existing build's intent matches this run exactly."""
    if existing_intent is None:
        raise RuntimeError(
            "--resume was set but no build_intent.json exists; partial outputs are unverified. "
            f"Delete the targets under {DECON_ROOT} and {CACHE_ROOT} and rerun fresh."
        )
    if existing_intent != current_intent:
        raise RuntimeError(
            "--resume intent mismatch: existing build_intent.json does not match this run "
            "(keep-id hash, sources, or expected counts changed). Delete the targets under "
            f"{DECON_ROOT} and {CACHE_ROOT} explicitly before rebuilding."
        )


def _filter_shard(path: str) -> Iterator[dict]:
    """Filter one val_docs shard into the three cutoff docs/ dirs."""
    keep_sets, meta = _keep_lookup()
    basename = path.rsplit("/", 1)[1]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text", "shard", "row"])

    rows = {tag: [] for tag in CUTOFF_TAGS}
    for record in table.to_pylist():
        doc_id = record["id"]
        if doc_id not in keep_sets["j090"]:
            continue
        doc_index, max_jaccard = meta[doc_id]
        record |= {"doc_index": doc_index, "max_jaccard": max_jaccard}
        for tag in CUTOFF_TAGS:
            if doc_id in keep_sets[tag]:
                rows[tag].append(record)

    for tag in CUTOFF_TAGS:
        out = f"{DECON_ROOT}/{tag}/docs/{basename}"
        filtered = pa.Table.from_pylist(rows[tag], schema=FILTERED_SCHEMA)
        with fsspec.open(out, "wb") as f:
            pq.write_table(filtered, f)
        counters.increment(f"filter/kept_{tag}", len(rows[tag]))

    yield {"input_shard": basename} | {f"kept_{tag}": len(rows[tag]) for tag in CUTOFF_TAGS}


def run_filter() -> None:
    shard_files = sorted(fsspec_glob(f"{VAL_DOCS}/*.parquet"))
    logger.info("filtering %d val_docs shards into %s", len(shard_files), DECON_ROOT)
    ctx = ZephyrContext(
        name="decon-val-filter",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", disk="10g", preemptible=False),
    )
    # skip_existing checkpoints per input shard: the stats record is written
    # only after that shard's three docs/ files are finalized.
    outcome = ctx.execute(
        Dataset.from_list(shard_files)
        .flat_map(_filter_shard)
        .write_parquet(
            f"{DECON_ROOT}/filter_stats/stats-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=STATS_SCHEMA,
            skip_existing=True,
        )
    )
    logger.info("filter counters: %s", dict(outcome.counters))


def actual_doc_count(tag: str) -> int:
    total = 0
    for path in fsspec_glob(f"{DECON_ROOT}/{tag}/docs/*.parquet"):
        with fsspec.open(path, "rb") as f:
            total += pq.read_metadata(f).num_rows
    return total


def cache_stats(tag: str) -> dict:
    with fsspec.open(f"{CACHE_ROOT}/{tag}/validation/.stats.json") as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Continue a build whose intent matches exactly.")
    args = parser.parse_args()

    keep = _load_keep_payloads()
    intent = build_intent(keep)
    found = existing_targets()
    if found and not args.resume:
        raise RuntimeError(f"targets already exist (pass --resume or delete them explicitly): {found}")
    if args.resume and found:
        existing_intent = None
        if fsspec_exists(INTENT_URI):
            with fsspec.open(INTENT_URI) as f:
                existing_intent = json.load(f)
        check_resume_allowed(existing_intent, intent)
        logger.info("resuming build; intent matches")
    if not fsspec_exists(INTENT_URI):
        with fsspec.open(INTENT_URI, "w") as f:
            json.dump(intent, f, indent=2)

    run_filter()

    for tag in CUTOFF_TAGS:
        expected = intent["cutoffs"][tag]
        docs = actual_doc_count(tag)
        assert docs == expected["expected_docs"], f"{tag}: filtered {docs} docs != {expected['expected_docs']}"

        tokenize(
            TokenizeConfig(
                train_paths=[],
                validation_paths=[f"{DECON_ROOT}/{tag}/docs/*.parquet"],
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
            **expected,
            "actual_docs": docs,
            "actual_tokens": stats["total_tokens"],
            "eval_sequences": stats["total_tokens"] // SEQ_LEN,
            "sources": {
                "val_docs": VAL_DOCS,
                "keep_ids": f"{DECON_ROOT}/keep_ids/keep_ids_{tag}.json",
                "scan_verified_pairs": [
                    f"gs://marin-us-east5/scratch/ahmed/midtrain_dedup/{sub}/verified_pairs"
                    for sub in ("3_284x71", "4plus_284x71", "4plus_mind_284x71")
                ],
            },
            "tokenizer": TOKENIZER,
            "cache_path": f"{CACHE_ROOT}/{tag}",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with fsspec.open(f"{DECON_ROOT}/{tag}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            "%s ready: %d docs, %d tokens, %d eval sequences at %s",
            tag,
            docs,
            stats["total_tokens"],
            manifest["eval_sequences"],
            manifest["cache_path"],
        )


if __name__ == "__main__":
    main()
