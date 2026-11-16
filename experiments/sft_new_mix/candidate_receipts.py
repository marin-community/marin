# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve expected shard receipts and build catalogs from complete source inventories."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheCatalog, CacheCatalogEntry, CacheLedger, CacheMetadata
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_filters import MAX_TOKENS
from experiments.sft_new_mix.exp3_runtime import configure_storage


def component_name(source: str, loss_mode: str) -> str:
    if loss_mode == "chat_assistant_only":
        return "sft/source/" + source
    if loss_mode == "full_token":
        return "pretrain/source/" + source.removeprefix("replay/")
    raise ValueError("Every source needs an explicit supported loss mode")


def expected_receipt(job: dict) -> str:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    source = hashlib.sha256(job["source"].encode()).hexdigest()
    return str(StoragePath(job["output_prefix"]) / "components" / source / "shards" / identity / "receipt.json")


def read_receipt(job: dict) -> dict:
    uri = expected_receipt(job)
    try:
        content = StoragePath(uri).read_bytes()
    except FileNotFoundError:
        return {"source": job["source"], "receipt_uri": uri, "complete": False}
    value = json.loads(content)
    if value["job"] != job or value["complete"] is not True:
        raise ValueError("Candidate receipt identity or completion mismatch")
    if value["output_rows"]:
        if sum(value["lengths"]) != value["output_tokens"] or len(value["lengths"]) != value["output_rows"]:
            raise ValueError("Candidate receipt lengths disagree with retained population")
        if max(value["lengths"]) > MAX_TOKENS:
            raise ValueError("Candidate receipt includes an oversized document")
    value.pop("lengths", None)
    return {**value, "receipt_uri": uri, "receipt_sha256": hashlib.sha256(content).hexdigest()}


def collect_inventory(plan: dict, workers: int) -> dict:
    """Include every expected input; partial collection is explicitly non-final."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(read_receipt, plan["jobs"]))
    by_source = defaultdict(list)
    for job, receipt in zip(plan["jobs"], receipts, strict=True):
        by_source[job["source"]].append((job, receipt))
    sources = []
    transforms = []
    missing = []
    for name, shards in sorted(by_source.items()):
        loss = plan["source_loss"][name]
        component = component_name(name, loss["loss_mode"])
        totals = Counter()
        stages = defaultdict(Counter)
        outputs = []
        source_complete = all(receipt["complete"] for _, receipt in shards)
        for job, receipt in shards:
            totals["input_documents"] += job["expected_rows"]
            totals["input_tokens"] += job["expected_tokens"]
            if not receipt["complete"]:
                missing.append(receipt["receipt_uri"])
                continue
            totals["completed_input_documents"] += job["expected_rows"]
            totals["completed_input_tokens"] += job["expected_tokens"]
            totals["output_documents"] += receipt["output_rows"]
            totals["output_tokens"] += receipt["output_tokens"]
            totals["supervised_prediction_positions"] += receipt.get("supervised_prediction_positions", 0)
            for stage, counts in receipt["stats"].items():
                stages[stage].update(counts)
            if receipt["output_rows"]:
                outputs.append(
                    {
                        "cache_uri": receipt["output_path"],
                        "rows": receipt["output_rows"],
                        "tokens": receipt["output_tokens"],
                        "supervised_prediction_positions": receipt["supervised_prediction_positions"],
                        "receipt_uri": receipt["receipt_uri"],
                        "receipt_sha256": receipt["receipt_sha256"],
                        "original_input_path": job["input_path"],
                        "decisions_uri": receipt["receipt_uri"].removesuffix("receipt.json") + "decisions.jsonl",
                    }
                )
        sources.append(
            {
                "source": name,
                "component": component,
                **loss,
                "complete": source_complete,
                "expected_shards": len(shards),
                "completed_shards": sum(value["complete"] for _, value in shards),
                "counts": dict(totals),
                "shards": outputs,
            }
        )
        for stage, counts in stages.items():
            transforms.append(
                {
                    "source": name,
                    "component": component,
                    "loss_mode": loss["loss_mode"],
                    "transform": stage,
                    "complete": source_complete,
                    **counts,
                }
            )
    first = plan["jobs"][0]
    return {
        "complete": not missing,
        "readiness": "intermediate; overlays, lexical receipts, weights and resolve pending",
        "expected_shards": len(receipts),
        "completed_shards": sum(receipt["complete"] for receipt in receipts),
        "tokenizer_uri": first["tokenizer_uri"],
        "tokenizer_sha256": first["tokenizer_sha256"],
        "sources": sources,
        "transforms": transforms,
        "missing_receipts": missing,
        "source_replacements": plan["source_replacements"],
    }


def catalog_from_inventory(inventory: dict, cache_root: str) -> tuple[CacheCatalog, dict]:
    """Embed complete per-source ledgers without copying immutable cache arrays.

    Sources share a storage ancestor, while their catalog ledgers select disjoint
    relative shard paths. No shared top-level cache ledger is overwritten.
    """
    if inventory["complete"] is not True or any(source["complete"] is not True for source in inventory["sources"]):
        raise ValueError("Cannot build a catalog from incomplete candidate sources")
    entries = {}
    metadata = {}
    for source in inventory["sources"]:
        if not source["shards"]:
            continue
        name = source["component"]
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            layout=CACHE_LAYOUT_SHARDED,
            metadata=CacheMetadata({"source": source["source"], "loss_mode": source["loss_mode"]}),
        )
        for shard in source["shards"]:
            child = CacheLedger.load(shard["cache_uri"])
            if not child.is_finished or child.layout == CACHE_LAYOUT_SHARDED:
                raise ValueError("Candidate inventory requires complete flat leaf caches")
            if child.total_num_rows != shard["rows"] or child.field_counts["input_ids"] != shard["tokens"]:
                raise ValueError("Candidate leaf ledger disagrees with its receipt")
            if child.field_counts["input_ids"] != child.field_counts["loss_weights"]:
                raise ValueError("Candidate token and prediction-weight position counts differ")
            relative = StoragePath(shard["cache_uri"]).relative_to(StoragePath(cache_root))
            if not relative or relative in ledger.shard_rows:
                raise ValueError("Repeated candidate shard path")
            ledger.finished_shards.append(relative)
            ledger.shard_rows[relative] = child.total_num_rows
            ledger.field_counts_by_shard[relative] = child.field_counts
            ledger.total_num_rows += child.total_num_rows
            for field, count in child.field_counts.items():
                ledger.field_counts[field] = ledger.field_counts.get(field, 0) + count
        entries[name] = CacheCatalogEntry(cache_dir=cache_root, ledger=ledger)
        metadata[name] = {
            key: source[key]
            for key in ("loss_mode", "loss_weights_key", "loss_weights_indexing", "teacher_only", "source_revision")
        }
    if not entries:
        raise ValueError("Candidate catalog has no retained sources")
    return CacheCatalog(splits={"train": entries}), {"version": 1, "sources": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    configure_storage()
    content = StoragePath(args.plan).read_bytes()
    inventory = collect_inventory(json.loads(content), args.workers)
    inventory["input_plan_uri"] = args.plan
    inventory["input_plan_sha256"] = hashlib.sha256(content).hexdigest()
    StoragePath(args.output).write_text(json.dumps(inventory, sort_keys=True))
    print(
        json.dumps({"expected_shards": inventory["expected_shards"], "completed_shards": inventory["completed_shards"]})
    )


if __name__ == "__main__":
    main()
