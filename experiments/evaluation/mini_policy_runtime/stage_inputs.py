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

from experiments.evaluation.mini_policy_runtime.artifacts import publish
from experiments.evaluation.mini_policy_runtime.cohort import CohortModel, cohort_model
from experiments.evaluation.mini_policy_runtime.profile import DATASET, DESTINATION, REVISION
from experiments.evaluation.mini_policy_runtime.repair_release import repaired_table


def main(excluded: list[str], wheels_revision: str, included: list[str], output_prefix: str, model: CohortModel):
    logging.basicConfig(level=logging.INFO)
    unique = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    table = repaired_table(pq.read_table(unique), wheels_revision)
    seen = set()
    chosen = []
    for i, row in enumerate(table.to_pylist()):
        if (
            row["benchmark"] not in seen
            and row["benchmark"] not in excluded
            and (not included or row["benchmark"] in included)
        ):
            seen.add(row["benchmark"])
            chosen.append(i)
    if included and set(included) != seen:
        raise ValueError("A requested smoke benchmark is absent from the release")
    smoke_name = ("smoke-" + "-".join(sorted(included)) + "-runtime-v4" if included else "smoke-runtime-v4") + (
        "-without-" + "-".join(sorted(excluded)) if excluded else ""
    )
    expanded_path = hf_hub_download(DATASET, "data/validation-rollouts.parquet", repo_type="dataset", revision=REVISION)
    expanded = repaired_table(pq.read_table(expanded_path), wheels_revision)
    if table.num_rows != 840 or expanded.num_rows != 1626:
        raise ValueError("Published dataset cardinality changed")
    for name, rows in ((smoke_name, table.take(pa.array(chosen))), ("full-runtime-v4", expanded)):
        destination = StoragePath(output_prefix) / name / "rollouts.parquet"
        fs, key = filesystem_for(str(destination))
        with fs.open(key, "wb") as output:
            pq.write_table(rows, output)
    logging.info("Dataset staged; resolving pinned model mirror")
    model_uri = resolve_model_path(model.repo, cache_ttl_days=30, revision=model.revision)
    report = {
        "dataset": DATASET,
        "source_revision": REVISION,
        "wheels_revision": wheels_revision,
        "runtime_repair": "v4",
        "smoke_task_ids": [table.to_pylist()[i]["task_id"] for i in chosen],
        "smoke_benchmarks": sorted(seen),
        "smoke_data_uri": str(StoragePath(output_prefix) / smoke_name),
        "smoke_excluded_benchmarks": excluded,
        "smoke_included_benchmarks": included,
        "full_data_uri": str(StoragePath(output_prefix) / "full-runtime-v4"),
        "model": model.repo,
        "model_revision": model.revision,
        "cohort_key": model.key,
        "model_uri": model_uri,
    }
    publish(f"smoke-runtime-v4-{model.key}-staged-inputs.json", report, output_prefix)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-exclude", action="append", default=[])
    parser.add_argument("--smoke-include", action="append", default=[])
    parser.add_argument("--wheels-revision", required=True)
    parser.add_argument("--output-prefix", default=DESTINATION)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.smoke_exclude, args.wheels_revision, args.smoke_include, args.output_prefix, cohort_model(args.model))
