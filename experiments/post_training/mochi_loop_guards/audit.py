# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind frozen source-order IDs to published TaskSpecs, then audit one saved panel."""

import argparse
import hashlib
import io
import json
import subprocess

import pyarrow.parquet as pq
from marin.rl.skyrl import skyrl_cpu_command
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec

from experiments.post_training.mochi_loop_guards.presets import VERSION
from experiments.post_training.mochi_loop_guards.selection import COHORT_FILES, MANIFEST_COHORTS

REGIONAL_SELECTION = (
    "s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/panel/" + VERSION
)


def resolved_manifest(manifest: dict, selection: str) -> dict:
    """Resolve each frozen source index without changing membership or trial multiplicity."""
    manifest = json.loads(json.dumps(manifest))
    manifest["frozen_manifest_sha256"] = manifest.pop("manifest_sha256")
    for key, filename in COHORT_FILES.items():
        cohort = manifest["cohorts"][MANIFEST_COHORTS[key]]
        table = pq.read_table(io.BytesIO((StoragePath(selection) / f"{filename}.parquet").read_bytes()))
        specs = [TaskRolloutSpec.model_validate_json(value) for value in table[TASK_ROLLOUT_COLUMN].to_pylist()]
        if len(specs) != len(cohort["rows"]):
            raise ValueError(f"{key}: published task census differs from the frozen panel")
        if key == "swesmith_ho":
            by_id = {spec.task.id: spec for spec in specs}
            if set(by_id) != {entry["task_id"] for entry in cohort["rows"]}:
                raise ValueError("Published SWESmith membership differs from frozen row hashes")
            specs = [by_id[entry["task_id"]] for entry in cohort["rows"]]
        for entry, spec in zip(cohort["rows"], specs, strict=True):
            entry["frozen_task_id"] = entry["task_id"]
            entry["task_id"] = spec.task.id
            entry["source"] = spec.task.source.model_dump(mode="json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="cohorts/manifest.json")
    parser.add_argument("--selection", default=REGIONAL_SELECTION)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    configure_coreweave_s3()
    source = StoragePath(args.manifest).read_bytes()
    manifest = resolved_manifest(json.loads(source), args.selection)
    manifest["frozen_manifest_file_sha256"] = hashlib.sha256(source).hexdigest()
    output = StoragePath(args.output)
    output.mkdirs()
    manifest_path = str(output / "resolved-manifest.json")
    (output / "resolved-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pattern = str(StoragePath(args.run_output) / "exports/dumped_evals/*/eval_only/*.jsonl")
    command = skyrl_cpu_command(
        "python",
        "vendor/skyrl/scripts/loop_audit.py",
        "--manifest",
        manifest_path,
        "--output",
        str(output / "audit.json"),
    )
    for cohort in manifest["cohorts"]:
        command.extend(("--input", f"{cohort}={pattern}"))
    if args.baseline:
        command.extend(("--baseline", args.baseline))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
