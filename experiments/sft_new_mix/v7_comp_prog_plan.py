# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the option-G V7 build plan for the two comp-prog sources.

Reads the pinned scripted-v3 plan, keeps only the comp-prog source jobs, and
writes the V7 config (scripted-v3 plus ``comp_prog_label_rescue`` and its
pinned label table), the tokenizer copy and the build plan to a new prefix.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_filters import Stage
from experiments.sft_new_mix.exp3_runtime import (
    BUILD_IMPLEMENTATION_FILES,
    configure_storage,
    implementation_identity,
    read_pinned,
)

logger = logging.getLogger(__name__)

COMP_PROG_SOURCES = (
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp",
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python",
)
EXPECTED_JOBS = 101
EXPECTED_ROWS = {"cpp": 331_927, "python": 333_399}
VERSION = "2026.11.18-v7-option-g"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted-plan", required=True)
    parser.add_argument("--scripted-plan-sha256", required=True)
    parser.add_argument("--scripted-config", required=True)
    parser.add_argument("--scripted-config-sha256", required=True)
    parser.add_argument("--label-table", required=True)
    parser.add_argument("--label-table-sha256", required=True)
    parser.add_argument("--labels-receipt", required=True, help="S1a receipt URI; binds the table pin")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True, help="Local directory for the plan and receipt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    plan_raw = read_pinned(args.scripted_plan, args.scripted_plan_sha256)
    scripted = json.loads(plan_raw)
    config_raw = read_pinned(args.scripted_config, args.scripted_config_sha256)
    config = json.loads(config_raw)
    labels_receipt = json.loads(StoragePath(f"{args.labels_receipt.rstrip('/')}/receipt.json").read_text())
    if labels_receipt.get("label_table_sha256") != args.label_table_sha256:
        raise ValueError("The label table pin does not match the S1a receipt")
    plan_tokenizer_sha = {job["tokenizer_sha256"] for job in scripted["jobs"]}
    if len(plan_tokenizer_sha) != 1:
        raise ValueError(f"The scripted plan pins {len(plan_tokenizer_sha)} tokenizers")
    tokenizer_sha = plan_tokenizer_sha.pop()
    tokenizer_bytes = read_pinned(str(StoragePath(args.scripted_plan).parent / "tokenizer.json"), tokenizer_sha)

    jobs = [job for job in scripted["jobs"] if job["source"] in COMP_PROG_SOURCES]
    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(f"Comp-prog shard jobs: {len(jobs)}, expected {EXPECTED_JOBS}")
    for source, lang in ((COMP_PROG_SOURCES[0], "cpp"), (COMP_PROG_SOURCES[1], "python")):
        source_jobs = [job for job in jobs if job["source"] == source]
        rows = sum(job["expected_rows"] for job in source_jobs)
        if rows != EXPECTED_ROWS[lang]:
            raise ValueError(f"{lang} rows: {rows}, expected {EXPECTED_ROWS[lang]}")
        if any(job["expected_rows"] <= 0 for job in source_jobs):
            raise ValueError(f"{lang} has an empty shard job")

    enabled = list(config["enabled"])
    if Stage.COMP_PROG_LABEL_RESCUE.value in enabled:
        raise ValueError("The scripted-v3 config already enables the label stage")
    # The scripted-v3 config predates the #11547 stage deletion. The deleted
    # opencode_user_quotes stage was gated on "opencode" in source, so it is a
    # strict no-op for the comp-prog sources this plan rebuilds.
    unknown = sorted(set(enabled) - {stage.value for stage in Stage})
    if unknown != ["opencode_user_quotes"]:
        raise ValueError(f"Scripted config has unknown stages: {unknown}")
    enabled = [stage for stage in enabled if stage != "opencode_user_quotes"]
    enabled.append(Stage.COMP_PROG_LABEL_RESCUE.value)
    v7_config = {
        **config,
        "version": VERSION,
        "enabled": enabled,
        "comp_prog_labels": {"table_uri": args.label_table, "table_sha256": args.label_table_sha256},
    }
    config_bytes = json.dumps(v7_config, sort_keys=True, indent=2).encode()
    tokenizer_bytes = (StoragePath(args.scripted_plan).parent / "tokenizer.json").read_bytes()

    destination = StoragePath(args.destination)
    config_uri = str(destination / "config.json")
    (destination / "config.json").write_bytes(config_bytes)
    tokenizer_uri = str(destination / "tokenizer.json")
    (destination / "tokenizer.json").write_bytes(tokenizer_bytes)

    v7_jobs = []
    for job in jobs:
        # The saved scripted plan omits the executed full-shard row ranges;
        # exp3_plan binds row_start=0 and row_stop=expected_rows per shard.
        v7_jobs.append(
            {
                **job,
                "row_start": 0,
                "row_stop": job["expected_rows"],
                "output_prefix": args.destination,
                "config_uri": config_uri,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "tokenizer_uri": tokenizer_uri,
                "tokenizer_sha256": tokenizer_sha,
                "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
            }
        )
    build_plan = {
        "output_prefix": args.destination,
        "jobs": v7_jobs,
        "sources": sorted({job["source"] for job in v7_jobs}),
        "scripted_plan_sha256": args.scripted_plan_sha256,
        "scripted_config_sha256": args.scripted_config_sha256,
        "label_table_sha256": args.label_table_sha256,
        "version": VERSION,
    }
    (output / "plan.json").write_text(json.dumps(build_plan, sort_keys=True, indent=2))
    plan_uri = str(destination / "plan.json")
    (destination / "plan.json").write_bytes(json.dumps(build_plan, sort_keys=True, indent=2).encode())
    receipt = {
        "jobs": len(v7_jobs),
        "rows": sum(job["expected_rows"] for job in v7_jobs),
        "tokens": sum(job["expected_tokens"] for job in v7_jobs),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "tokenizer_sha256": tokenizer_sha,
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
        "plan_uri": plan_uri,
        "label_table": args.label_table,
        "label_table_sha256": args.label_table_sha256,
        "omitted_stages": unknown,
        "version": VERSION,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    (destination / "plan-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
