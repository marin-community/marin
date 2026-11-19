# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parallelize unfinished small-document caches without altering their population."""

import argparse
import json
import logging

import numpy as np
from fray.types import ResourceConfig
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.candidate_receipts import collect_inventory, expected_receipt
from experiments.sft_new_mix.exp3_build import process_shard
from experiments.sft_new_mix.exp3_runtime import (
    BUILD_IMPLEMENTATION_FILES,
    configure_storage,
    implementation_identity,
    read_pinned,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--rows-per-task", required=True, type=int)
    parser.add_argument("--max-input-document-tokens", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    content = read_pinned(args.plan, args.plan_sha256)
    plan = json.loads(content)
    inventory = collect_inventory(plan, workers=args.workers)
    root = StoragePath(args.output_prefix)
    (root / "sealed-parent-inventory.json").write_text(json.dumps(inventory, sort_keys=True))
    missing = set(inventory["missing_receipts"])
    fresh, combined, retired = [], [], []
    for job in plan["jobs"]:
        if expected_receipt(job) not in missing:
            combined.append(job)
            continue
        if not (job["source"] == "identity-data" or job["source"].startswith("replay/")):
            raise ValueError("Remainder partitioning is restricted to replay and existing identity")
        store = JaggedArrayStore.open(prefix_join(job["input_path"], "input_ids"), mode="r", dtype=np.int32)
        if len(store) != job["expected_rows"]:
            raise ValueError("Parent cache row count changed")
        ends = np.asarray(store.offsets[1 : len(store) + 1].read().result(), dtype=np.int64)
        lengths = np.diff(np.concatenate(([0], ends)))
        if int(lengths.sum()) != job["expected_tokens"] or max(lengths) > args.max_input_document_tokens:
            raise ValueError("Parent token counts or small-document partition contract changed")
        children = []
        for start in range(0, len(store), args.rows_per_task):
            stop = min(len(store), start + args.rows_per_task)
            child = {
                **job,
                "row_start": start,
                "row_stop": stop,
                "expected_rows": stop - start,
                "expected_tokens": int(lengths[start:stop].sum()),
                "output_prefix": args.output_prefix,
                "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
            }
            children.append(child)
        if sum(child["expected_rows"] for child in children) != job["expected_rows"]:
            raise ValueError("Range partition lost or repeated source rows")
        if sum(child["expected_tokens"] for child in children) != job["expected_tokens"]:
            raise ValueError("Range partition changed the source token budget")
        fresh.extend(children)
        combined.extend(children)
        retired.append({"job": job, "uncompleted_receipt_uri": expected_receipt(job), "range_tasks": len(children)})
    combined_plan = {
        **plan,
        "jobs": combined,
        "output_prefix": args.output_prefix,
        "range_partition_lineage": {
            "parent_plan_uri": args.plan,
            "parent_plan_sha256": args.plan_sha256,
            "retired_unfinished_jobs": retired,
            "population_policy": "All original rows and tokens, exactly once; no sampling or trimming",
        },
    }
    (root / "combined-plan.json").write_text(json.dumps(combined_plan, sort_keys=True))
    (root / "plan.json").write_text(json.dumps({**combined_plan, "jobs": fresh}, sort_keys=True))
    with ZephyrContext(
        name="exp3-small-document-remainder",
        max_workers=args.workers,
        resources=ResourceConfig(cpu=2, ram="16g"),
        chunk_storage_prefix=prefix_join(args.output_prefix, "zephyr"),
        max_execution_retries=2,
    ) as context:
        result = context.execute(Dataset.from_list(fresh).map(process_shard))
    (root / "execution.json").write_text(json.dumps({"results": result.results}, default=str))
    final = collect_inventory(combined_plan, workers=args.workers)
    if final["complete"] is not True:
        raise ValueError("Range remainder did not complete every original input population")
    (root / "inventory.json").write_text(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    main()
