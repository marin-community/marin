# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned tool selection with coordinator capacity for its manifest size."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_runtime import configure_storage, implementation_identity, read_pinned
from experiments.sft_new_mix.job_manifests import read_job_manifest, stage_job_manifest
from experiments.sft_new_mix.tool_overlay import VERIFIER_IMPLEMENTATION_FILES, materialize_archive

JOB_MANIFEST_WRITE_WORKERS = 16


def materialize_archive_reference(reference: dict) -> dict:
    configure_storage()
    return materialize_archive(read_job_manifest(reference))


def execute_references(
    references: list[dict], recipe: dict, output_prefix: str, workers: int, coordinator_memory: str
) -> None:
    """Verify archive jobs and bind their completed payload receipts."""
    with ZephyrContext(
        name="exp3-verified-tool-payloads",
        max_workers=workers,
        resources=ResourceConfig(cpu=2, ram="8g", disk="20g"),
        coordinator_resources=ResourceConfig(cpu=2, ram=coordinator_memory, preemptible=False),
        chunk_storage_prefix=prefix_join(output_prefix, "zephyr"),
        max_execution_retries=2,
    ) as context:
        result = context.execute(Dataset.from_list(references).map(materialize_archive_reference))
    records, receipts = [], []
    for value in result.results:
        content = StoragePath(value["receipt_uri"]).read_bytes()
        records.extend(json.loads(content)["records"])
        receipts.append({"uri": value["receipt_uri"], "sha256": hashlib.sha256(content).hexdigest()})
    (StoragePath(output_prefix) / "manifest.json").write_text(
        json.dumps({"recipe": recipe, "records": records, "archive_receipts": receipts}, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--verifier-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--coordinator-memory", required=True)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    if implementation_identity(VERIFIER_IMPLEMENTATION_FILES) != args.verifier_sha256:
        raise ValueError("Payload verifier changed from the explicit run pin")
    selection = json.loads(read_pinned(args.selection, args.selection_sha256))
    jobs = [
        {**job, "output_prefix": args.output_prefix, "implementation_sha256": args.verifier_sha256}
        for job in selection["jobs"]
    ]
    recipe = {key: value for key, value in vars(args).items()}
    recipe["orchestrator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (StoragePath(args.output_prefix) / "recipe.json").write_text(json.dumps(recipe, sort_keys=True))
    # Keep bulk records outside ActorCall protobufs, whose payloads are bounded.
    with ThreadPoolExecutor(max_workers=JOB_MANIFEST_WRITE_WORKERS) as pool:
        references = list(pool.map(stage_job_manifest, jobs))
    del jobs, selection
    (StoragePath(args.output_prefix) / "job-references.json").write_text(json.dumps(references, sort_keys=True))
    execute_references(references, recipe, args.output_prefix, args.workers, args.coordinator_memory)


if __name__ == "__main__":
    main()
