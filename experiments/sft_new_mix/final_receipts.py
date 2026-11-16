# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect pinned final-shard receipts into source inventories and lexical inputs."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.candidate_receipts import component_name
from experiments.sft_new_mix.exp3_filters import MAX_TOKENS
from experiments.sft_new_mix.exp3_runtime import configure_storage


def read_final_receipt(job: dict) -> dict:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    path = StoragePath(job["output_prefix"]) / "shards" / identity / "receipt.json"
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {"complete": False, "receipt_uri": str(path)}
    receipt = json.loads(content)
    if receipt["complete"] is not True or receipt["job"] != job:
        raise ValueError("Final shard receipt changed its job or completion status")
    lengths = receipt.pop("lengths")
    if len(lengths) != receipt["output_rows"] or sum(lengths) != receipt["output_tokens"]:
        raise ValueError("Final shard lengths disagree with its retained population")
    if lengths and (min(lengths) <= 0 or max(lengths) > MAX_TOKENS):
        raise ValueError("Final shard contains a document outside the context limit")
    return {**receipt, "receipt_uri": str(path), "receipt_sha256": hashlib.sha256(content).hexdigest()}


def collect_final_inventory(plan: dict, source_loss: dict, workers: int) -> dict:
    """Account for every planned shard, preserving explicit incomplete status."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(read_final_receipt, plan["jobs"]))
    grouped = defaultdict(list)
    for job, receipt in zip(plan["jobs"], receipts, strict=True):
        grouped[job["source"]].append((job, receipt))
    sources, transforms, missing = [], [], []
    for name, shards in sorted(grouped.items()):
        loss = source_loss[name]
        counts = Counter()
        stages = defaultdict(Counter)
        outputs = []
        complete = all(receipt["complete"] for _, receipt in shards)
        for job, receipt in shards:
            if (loss["loss_mode"], loss["teacher_only"]) != (job["loss_mode"], job["teacher_only"]):
                raise ValueError("Final shard and pinned source supervision disagree")
            counts["input_documents"] += job["base_shard"]["rows"]
            counts["input_tokens"] += job["base_shard"]["tokens"]
            if not receipt["complete"]:
                missing.append(receipt["receipt_uri"])
                continue
            counts["output_documents"] += receipt["output_rows"]
            counts["output_tokens"] += receipt["output_tokens"]
            counts["supervised_prediction_positions"] += receipt["supervised_prediction_positions"]
            counts["changed_documents"] += receipt["changed_documents"]
            for stage, values in receipt["stats"].items():
                stages[stage].update(values)
            if receipt["output_rows"]:
                outputs.append(
                    {
                        "cache_uri": receipt["output_path"],
                        "rows": receipt["output_rows"],
                        "tokens": receipt["output_tokens"],
                        "supervised_prediction_positions": receipt["supervised_prediction_positions"],
                        "receipt_uri": receipt["receipt_uri"],
                        "receipt_sha256": receipt["receipt_sha256"],
                        "original_input_path": job["base_shard"]["cache_uri"],
                        "decisions_uri": str(StoragePath(receipt["receipt_uri"]).parent / "decisions.jsonl"),
                    }
                )
        component = component_name(name, loss["loss_mode"])
        sources.append(
            {
                "source": name,
                "component": component,
                **loss,
                "complete": complete,
                "expected_shards": len(shards),
                "completed_shards": sum(receipt["complete"] for _, receipt in shards),
                "counts": dict(counts),
                "shards": outputs,
            }
        )
        transforms.extend(
            {"source": name, "component": component, "loss_mode": loss["loss_mode"], "transform": stage, **values}
            for stage, values in stages.items()
        )
    first = plan["jobs"][0]
    return {
        "complete": not missing,
        "scope": plan["scope"],
        "expected_shards": len(receipts),
        "completed_shards": sum(receipt["complete"] for receipt in receipts),
        "tokenizer_uri": first["tokenizer_uri"],
        "tokenizer_sha256": first["tokenizer_sha256"],
        "sources": sources,
        "transforms": transforms,
        "missing_receipts": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--source-loss", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    configure_storage()
    content = StoragePath(args.plan).read_bytes()
    inventory = collect_final_inventory(
        json.loads(content), json.loads(StoragePath(args.source_loss).read_text()), args.workers
    )
    inventory.update(input_plan_uri=args.plan, input_plan_sha256=hashlib.sha256(content).hexdigest())
    StoragePath(args.output).write_text(json.dumps(inventory, sort_keys=True))


if __name__ == "__main__":
    main()
