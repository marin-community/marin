# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage frozen Mini840 inputs and a complete model mirror on RNO2A CPUs."""

import argparse
import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.inference.model_preparation import resolve_model_path
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.mini_policy_runtime.environment_inventory import publish
from experiments.evaluation.mini_policy_runtime.profile import DATASET, DESTINATION, MODEL, MODEL_REVISION, REVISION
from experiments.evaluation.mini_policy_runtime.repair_release import repaired_table


def main(excluded: list[str], wheels_revision: str):
    logging.basicConfig(level=logging.INFO)
    unique = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    table = repaired_table(pq.read_table(unique), wheels_revision)
    seen = set()
    chosen = []
    for i, row in enumerate(table.to_pylist()):
        if row["benchmark"] not in seen and row["benchmark"] not in excluded:
            seen.add(row["benchmark"])
            chosen.append(i)
    smoke_name = "smoke-runtime-v3" + ("-without-" + "-".join(sorted(excluded)) if excluded else "")
    expanded_path = hf_hub_download(DATASET, "data/validation-rollouts.parquet", repo_type="dataset", revision=REVISION)
    expanded = repaired_table(pq.read_table(expanded_path), wheels_revision)
    if table.num_rows != 840 or expanded.num_rows != 1626:
        raise ValueError("Published dataset cardinality changed")
    for name, rows in ((smoke_name, table.take(pa.array(chosen))), ("full-runtime-v3", expanded)):
        destination = StoragePath(DESTINATION) / name / "rollouts.parquet"
        fs, key = filesystem_for(str(destination))
        with fs.open(key, "wb") as output:
            pq.write_table(rows, output)
    logging.info("Dataset staged; resolving pinned model mirror")
    model_uri = resolve_model_path(MODEL, cache_ttl_days=30, revision=MODEL_REVISION)
    report = {
        "dataset": DATASET,
        "source_revision": REVISION,
        "wheels_revision": wheels_revision,
        "runtime_repair": "v3",
        "smoke_task_ids": [table.to_pylist()[i]["task_id"] for i in chosen],
        "smoke_benchmarks": sorted(seen),
        "smoke_data_uri": str(StoragePath(DESTINATION) / smoke_name),
        "smoke_excluded_benchmarks": excluded,
        "full_data_uri": str(StoragePath(DESTINATION) / "full-runtime-v3"),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_uri": model_uri,
    }
    publish(smoke_name + "-staged-inputs.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-exclude", action="append", default=[])
    parser.add_argument("--wheels-revision", required=True)
    args = parser.parse_args()
    main(args.smoke_exclude, args.wheels_revision)
