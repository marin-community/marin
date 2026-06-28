# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a math validation cache clean against actually seen contaminated docs.

This is the reverse-use artifact for
``decon-seen-delphi-math-val-v2``. That Datakit run marked the actually seen
1e22 p33m67 K=0.20 math training docs against the math val docs. This driver
uses the completed Datakit attributes to remove validation docs implicated by
those contaminated seen docs, then tokenizes the remaining validation docs.

Drop semantics
--------------
Datakit scored seen training docs by paragraph, but this script filters at
validation-record granularity: keep or drop the whole validation document. A val
doc is dropped if any of its indexed 13-word ngram hashes appears in the
``matched_hashes`` of a contaminated seen training record, as attributed through
``_bloom/eval_hash_index.parquet``. This is stricter than a reverse Datakit
``>=0.5`` threshold and avoids a very large reverse scan over 6.6M docs.

Launch in us-east5 so all heavy reads stay in-region:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 64GB --disk 50GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name clean-val-against-seen-datakit \
        -e PYTHONUNBUFFERED 1 \
        -- python scripts/analysis/build_clean_val_against_seen_datakit.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash
from fray import ResourceConfig
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.utils import fsspec_exists, fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

MIDTRAIN = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
SEEN_DECON_ROOT = f"{MIDTRAIN}/seen_docs/1e22_p33m67_k020_math/val_decon"
DATAKIT_ATTRS = f"{SEEN_DECON_ROOT}/datakit_decon_attrs"
EVAL_HASH_INDEX = f"{DATAKIT_ATTRS}/_bloom/eval_hash_index.parquet"
VAL_DOCS = f"{MIDTRAIN}/val_docs"
CLEAN_ROOT = f"{SEEN_DECON_ROOT}/clean_val_against_contaminated_seen_docs"
CACHE_ROOT = "gs://marin-us-east5/tokenized/nemotron_math_val_clean_seen_1e22_p33m67_k020"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SEQ_LEN = 4096

DROP_IDS_FILENAME = "drop_val_ids.json"
INTENT_FILENAME = "build_intent.json"
MANIFEST_FILENAME = "manifest.json"

FILTERED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("shard", pa.string()),
        ("row", pa.int64()),
    ]
)
FILTER_STATS_SCHEMA = pa.schema(
    [
        ("input_shard", pa.string()),
        ("input_docs", pa.int64()),
        ("kept_docs", pa.int64()),
        ("dropped_docs", pa.int64()),
    ]
)

_DROP_IDS: set[str] | None = None
_DROP_IDS_URI: str | None = None


@dataclass(frozen=True)
class HitHashStats:
    hit_hashes: set[int]
    attr_files: int
    attr_records: int
    contaminated_records: int
    raw_matched_hashes: int


@dataclass(frozen=True)
class EvalIdStats:
    drop_ids: set[str]
    eval_index_rows: int
    matched_eval_index_rows: int
    eval_index_row_groups: int


def write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> dict[str, Any]:
    with fsspec.open(path) as f:
        return json.load(f)


def write_parquet_table(path: str, table: pa.Table) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "wb") as f:
        pq.write_table(table, f)


def uri_has_children(path: str) -> bool:
    return bool(fsspec_glob(f"{path.rstrip('/')}/*"))


def remove_uri(path: str) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if fs.exists(paths[0]):
        fs.rm(paths[0], recursive=True)


def fail_if_existing(paths: list[str], *, resume: bool, force: bool) -> list[str]:
    existing = [path for path in paths if fsspec_exists(path) or uri_has_children(path)]
    if not existing:
        return existing
    if force:
        for path in existing:
            logger.warning("removing existing target because --force was set: %s", path)
            remove_uri(path)
        return existing
    if resume:
        return existing
    raise RuntimeError(f"targets already exist; pass --resume or --force: {existing}")


def _unique_uint64_values(values: pa.Array | pa.ChunkedArray) -> set[int]:
    if len(values) == 0:
        return set()
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if len(array) == 0:
        return set()
    numpy_values = array.to_numpy(zero_copy_only=False)
    if numpy_values.size == 0:
        return set()
    return {int(value) for value in np.unique(numpy_values)}


def collect_hit_hashes_from_contaminated_attrs(attr_paths: list[str]) -> HitHashStats:
    """Collect unique matched eval hashes from contaminated seen-doc rows."""
    hit_hashes: set[int] = set()
    attr_records = 0
    contaminated_records = 0
    raw_matched_hashes = 0

    for index, path in enumerate(attr_paths):
        logger.info("reading Datakit attrs %d/%d: %s", index + 1, len(attr_paths), path)
        with fsspec.open(path, "rb") as f:
            table = pq.read_table(f, columns=["contaminated", "matched_hashes"])
        attr_records += table.num_rows

        contaminated = table.filter(table["contaminated"])
        contaminated_records += contaminated.num_rows
        if contaminated.num_rows == 0:
            continue

        flat_hashes = pc.list_flatten(contaminated["matched_hashes"])
        raw_matched_hashes += len(flat_hashes)
        before = len(hit_hashes)
        hit_hashes.update(_unique_uint64_values(flat_hashes))
        logger.info(
            "attrs %d/%d: contaminated=%d raw_hashes=%d unique_hashes=%d (+%d)",
            index + 1,
            len(attr_paths),
            contaminated.num_rows,
            len(flat_hashes),
            len(hit_hashes),
            len(hit_hashes) - before,
        )

    return HitHashStats(
        hit_hashes=hit_hashes,
        attr_files=len(attr_paths),
        attr_records=attr_records,
        contaminated_records=contaminated_records,
        raw_matched_hashes=raw_matched_hashes,
    )


def collect_eval_ids_for_hashes(hit_hashes: set[int], eval_hash_index_path: str) -> EvalIdStats:
    """Map matched hashes through Datakit's hash -> eval_id sidecar."""
    if not hit_hashes:
        return EvalIdStats(drop_ids=set(), eval_index_rows=0, matched_eval_index_rows=0, eval_index_row_groups=0)

    value_set = pa.array(np.fromiter(hit_hashes, dtype=np.uint64, count=len(hit_hashes)), type=pa.uint64())
    drop_ids: set[str] = set()
    eval_index_rows = 0
    matched_eval_index_rows = 0

    with fsspec.open(eval_hash_index_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        row_groups = parquet_file.num_row_groups
        for row_group in range(row_groups):
            logger.info("joining eval hash index row group %d/%d", row_group + 1, row_groups)
            table = parquet_file.read_row_group(row_group, columns=["hash", "eval_id"])
            eval_index_rows += table.num_rows
            matched = table.filter(pc.is_in(table["hash"], value_set=value_set))
            matched_eval_index_rows += matched.num_rows
            drop_ids.update(str(eval_id) for eval_id in matched["eval_id"].to_pylist())
            logger.info(
                "eval row group %d/%d: matched_rows=%d drop_ids=%d",
                row_group + 1,
                row_groups,
                matched.num_rows,
                len(drop_ids),
            )

    return EvalIdStats(
        drop_ids=drop_ids,
        eval_index_rows=eval_index_rows,
        matched_eval_index_rows=matched_eval_index_rows,
        eval_index_row_groups=row_groups,
    )


def derive_drop_ids_payload(
    *,
    datakit_attrs: str,
    eval_hash_index_path: str,
    max_attr_files: int | None,
) -> dict[str, Any]:
    attr_paths = sorted(fsspec_glob(f"{datakit_attrs.rstrip('/')}/*.parquet"))
    if not attr_paths:
        raise FileNotFoundError(f"No Datakit attribute parquet files found under {datakit_attrs}")
    if max_attr_files is not None:
        attr_paths = attr_paths[:max_attr_files]
        logger.warning("debug mode: deriving drop ids from only %d attr files", len(attr_paths))

    hit_stats = collect_hit_hashes_from_contaminated_attrs(attr_paths)
    eval_stats = collect_eval_ids_for_hashes(hit_stats.hit_hashes, eval_hash_index_path)
    drop_ids = sorted(eval_stats.drop_ids)

    return {
        "method": "drop_whole_val_doc_if_any_eval_hash_is_matched_by_a_contaminated_seen_doc",
        "granularity": {
            "datakit_scoring": "seen_training_document_labeled_by_max_paragraph_score",
            "validation_filtering": "whole_validation_document",
        },
        "strictness_note": (
            "This is stricter than a reverse Datakit >=0.5 paragraph threshold: any attributed 13-word "
            "ngram hit from the contaminated seen-doc set drops the whole validation document."
        ),
        "datakit_attrs": datakit_attrs,
        "eval_hash_index": eval_hash_index_path,
        "max_attr_files": max_attr_files,
        "source_attr_files": hit_stats.attr_files,
        "source_attr_records": hit_stats.attr_records,
        "source_contaminated_records": hit_stats.contaminated_records,
        "raw_matched_hashes_from_contaminated_records": hit_stats.raw_matched_hashes,
        "unique_matched_hashes_from_contaminated_records": len(hit_stats.hit_hashes),
        "eval_index_rows": eval_stats.eval_index_rows,
        "matched_eval_index_rows": eval_stats.matched_eval_index_rows,
        "eval_index_row_groups": eval_stats.eval_index_row_groups,
        "drop_ids_xxh3": xxhash.xxh3_128_hexdigest("\n".join(drop_ids).encode()),
        "drop_count": len(drop_ids),
        "drop_ids": drop_ids,
    }


def _load_drop_ids(drop_ids_uri: str) -> set[str]:
    global _DROP_IDS, _DROP_IDS_URI
    if _DROP_IDS is None or _DROP_IDS_URI != drop_ids_uri:
        payload = read_json(drop_ids_uri)
        _DROP_IDS = set(payload["drop_ids"])
        _DROP_IDS_URI = drop_ids_uri
    return _DROP_IDS


def _filter_val_shard(task: dict[str, str]) -> Iterator[dict[str, Any]]:
    path = task["path"]
    output_root = task["output_root"].rstrip("/")
    drop_ids = _load_drop_ids(task["drop_ids_uri"])
    basename = path.rsplit("/", 1)[1]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text", "shard", "row"])

    kept_rows = []
    dropped = 0
    for record in table.to_pylist():
        if record["id"] in drop_ids:
            dropped += 1
            continue
        kept_rows.append(record)

    out = f"{output_root}/docs/{basename}"
    write_parquet_table(out, pa.Table.from_pylist(kept_rows, schema=FILTERED_SCHEMA))
    counters.increment("filter/input_docs", table.num_rows)
    counters.increment("filter/kept_docs", len(kept_rows))
    counters.increment("filter/dropped_docs", dropped)
    yield {
        "input_shard": basename,
        "input_docs": table.num_rows,
        "kept_docs": len(kept_rows),
        "dropped_docs": dropped,
    }


def run_filter(*, val_docs: str, output_root: str, drop_ids_uri: str, max_workers: int) -> None:
    global _DROP_IDS, _DROP_IDS_URI
    _DROP_IDS = None
    _DROP_IDS_URI = drop_ids_uri

    shard_files = sorted(fsspec_glob(f"{val_docs.rstrip('/')}/*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No validation parquet files found under {val_docs}")
    tasks = [
        {"path": path, "output_root": output_root.rstrip("/"), "drop_ids_uri": drop_ids_uri} for path in shard_files
    ]

    logger.info("filtering %d val_docs shards into %s/docs", len(shard_files), output_root)
    ctx = ZephyrContext(
        name="clean-val-against-seen-filter",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", disk="10g", preemptible=False),
    )
    outcome = ctx.execute(
        Dataset.from_list(tasks)
        .flat_map(_filter_val_shard)
        .write_parquet(
            f"{output_root.rstrip('/')}/filter_stats/stats-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=FILTER_STATS_SCHEMA,
            skip_existing=True,
        )
    )
    logger.info("filter counters: %s", dict(outcome.counters))


def filter_counts(output_root: str) -> dict[str, int]:
    stats_files = sorted(fsspec_glob(f"{output_root.rstrip('/')}/filter_stats/*.parquet"))
    if not stats_files:
        raise FileNotFoundError(f"No filter stats found under {output_root}/filter_stats")
    totals = {"input_docs": 0, "kept_docs": 0, "dropped_docs": 0, "stats_files": len(stats_files)}
    for path in stats_files:
        with fsspec.open(path, "rb") as f:
            table = pq.read_table(f, columns=["input_docs", "kept_docs", "dropped_docs"])
        for key in ("input_docs", "kept_docs", "dropped_docs"):
            totals[key] += sum(int(value) for value in table[key].to_pylist())
    return totals


def actual_doc_count(docs_root: str) -> int:
    total = 0
    for path in fsspec_glob(f"{docs_root.rstrip('/')}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            total += pq.read_metadata(f).num_rows
    return total


def cache_stats(cache_root: str) -> dict[str, Any]:
    return read_json(f"{cache_root.rstrip('/')}/validation/.stats.json")


def cache_status(cache_root: str, expected_docs: int) -> str:
    stats_path = f"{cache_root.rstrip('/')}/validation/.stats.json"
    if fsspec_exists(stats_path):
        stats = cache_stats(cache_root)
        return "complete" if stats.get("total_elements") == expected_docs else "mismatch"
    if fsspec_glob(f"{cache_root.rstrip('/')}/validation/*"):
        return "partial"
    return "absent"


def delete_validation_cache(cache_root: str) -> None:
    path = f"{cache_root.rstrip('/')}/validation"
    fs, resolved = fsspec.core.url_to_fs(path)
    if fs.exists(resolved):
        fs.rm(resolved, recursive=True)


def build_tokenized_cache(*, docs_root: str, cache_root: str, expected_docs: int, force: bool) -> dict[str, Any]:
    status = "absent" if force else cache_status(cache_root, expected_docs)
    if status == "complete":
        logger.info("%s already complete with %d docs; skipping tokenize", cache_root, expected_docs)
        return cache_stats(cache_root)
    if status == "mismatch":
        raise RuntimeError(f"{cache_root} has .stats.json but doc count does not match {expected_docs}")
    if status == "partial" or force:
        logger.warning("%s validation cache is partial/forced; deleting and rebuilding", cache_root)
        delete_validation_cache(cache_root)

    tokenize(
        TokenizeConfig(
            train_paths=[],
            validation_paths=[f"{docs_root.rstrip('/')}/*.parquet"],
            cache_path=cache_root,
            tokenizer=TOKENIZER,
        )
    )
    stats = cache_stats(cache_root)
    if stats["total_elements"] != expected_docs:
        raise AssertionError(f"cache rows {stats['total_elements']} != expected docs {expected_docs}: {stats}")
    return stats


def build_intent(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "datakit_attrs": args.datakit_attrs,
        "eval_hash_index": args.eval_hash_index,
        "val_docs": args.val_docs,
        "output_root": args.output_root,
        "cache_root": args.cache_root,
        "tokenizer": TOKENIZER,
        "seq_len": SEQ_LEN,
        "max_attr_files": args.max_attr_files,
        "drop_granularity": "whole_validation_document",
        "drop_criterion": "any Datakit eval hash matched by a contaminated seen training document",
    }


def check_resume_allowed(intent_uri: str, current_intent: dict[str, Any]) -> None:
    if not fsspec_exists(intent_uri):
        raise RuntimeError("--resume was set but build_intent.json is missing; use --force or delete partial outputs")
    existing = read_json(intent_uri)
    if existing != current_intent:
        raise RuntimeError("--resume intent mismatch; use --force or a fresh output root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datakit-attrs", default=DATAKIT_ATTRS)
    parser.add_argument("--eval-hash-index", default=EVAL_HASH_INDEX)
    parser.add_argument("--val-docs", default=VAL_DOCS)
    parser.add_argument("--output-root", default=CLEAN_ROOT)
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--resume", action="store_true", help="Reuse outputs whose build intent matches.")
    parser.add_argument("--force", action="store_true", help="Remove output/cache targets before running.")
    parser.add_argument("--skip-derive-drop-ids", action="store_true", help="Use existing drop_val_ids.json.")
    parser.add_argument("--skip-filter", action="store_true", help="Use existing filtered docs and filter stats.")
    parser.add_argument("--skip-tokenize", action="store_true", help="Stop after writing filtered docs.")
    parser.add_argument("--max-attr-files", type=int, default=None, help="Debug only: derive from first N attr files.")
    parser.add_argument("--filter-max-workers", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.output_root = args.output_root.rstrip("/")
    args.cache_root = args.cache_root.rstrip("/")
    output_root = args.output_root
    cache_root = args.cache_root
    drop_ids_uri = f"{output_root}/{DROP_IDS_FILENAME}"
    intent_uri = f"{output_root}/{INTENT_FILENAME}"
    manifest_uri = f"{output_root}/{MANIFEST_FILENAME}"
    intent = build_intent(args)

    existing_targets = fail_if_existing([output_root, cache_root], resume=args.resume, force=args.force)
    if args.resume and existing_targets and not args.force:
        check_resume_allowed(intent_uri, intent)
    if not fsspec_exists(intent_uri):
        write_json(intent_uri, intent)

    if args.skip_derive_drop_ids:
        if not fsspec_exists(drop_ids_uri):
            raise FileNotFoundError(f"--skip-derive-drop-ids set but {drop_ids_uri} does not exist")
        drop_payload = read_json(drop_ids_uri)
    elif args.resume and fsspec_exists(drop_ids_uri):
        drop_payload = read_json(drop_ids_uri)
        logger.info("using existing drop-id payload: %s", drop_ids_uri)
    else:
        drop_payload = derive_drop_ids_payload(
            datakit_attrs=args.datakit_attrs,
            eval_hash_index_path=args.eval_hash_index,
            max_attr_files=args.max_attr_files,
        )
        write_json(drop_ids_uri, drop_payload)
        logger.info("wrote %d drop ids to %s", drop_payload["drop_count"], drop_ids_uri)

    if not args.skip_filter:
        run_filter(
            val_docs=args.val_docs,
            output_root=output_root,
            drop_ids_uri=drop_ids_uri,
            max_workers=args.filter_max_workers,
        )

    counts = filter_counts(output_root)
    if counts["dropped_docs"] != drop_payload["drop_count"]:
        raise AssertionError(
            f"filtered drop count {counts['dropped_docs']} != derived drop ids {drop_payload['drop_count']}"
        )
    docs = actual_doc_count(f"{output_root}/docs")
    if docs != counts["kept_docs"]:
        raise AssertionError(f"docs parquet row count {docs} != filter kept_docs {counts['kept_docs']}")

    if args.skip_tokenize:
        stats: dict[str, Any] | None = None
    else:
        stats = build_tokenized_cache(
            docs_root=f"{output_root}/docs",
            cache_root=cache_root,
            expected_docs=counts["kept_docs"],
            force=args.force,
        )

    manifest = {
        "name": "nemotron_math_val_clean_seen_1e22_p33m67_k020",
        "definition": (
            "math validation docs after dropping whole val docs with any Datakit 13-word ngram hash "
            "attributed to a contaminated actually-seen 1e22 p33m67 K=0.20 math training document"
        ),
        "drop_granularity": "whole_validation_document",
        "datakit_scoring_granularity": "seen_training_document_labeled_by_max_paragraph_score",
        "drop_count": drop_payload["drop_count"],
        "input_docs": counts["input_docs"],
        "actual_docs": counts["kept_docs"],
        "actual_tokens": None if stats is None else stats["total_tokens"],
        "eval_sequences": None if stats is None else stats["total_tokens"] // SEQ_LEN,
        "drop_ids_xxh3": drop_payload["drop_ids_xxh3"],
        "sources": {
            "datakit_attrs": args.datakit_attrs,
            "eval_hash_index": args.eval_hash_index,
            "drop_ids": drop_ids_uri,
            "val_docs": args.val_docs,
        },
        "tokenizer": TOKENIZER,
        "cache_path": cache_root,
        "filter_counts": counts,
        "drop_derivation": {key: value for key, value in drop_payload.items() if key not in {"drop_ids"}},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(manifest_uri, manifest)
    logger.info("clean val manifest written to %s", manifest_uri)


if __name__ == "__main__":
    main()
