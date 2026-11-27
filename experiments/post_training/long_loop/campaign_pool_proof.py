# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU proof worker for the existing campaign pool graph; never submit training."""

import argparse
import asyncio
import hashlib
import json
import os

import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.execution.lazy import run
from marin.experiment.cli import graph_handles
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec

from experiments.post_training.long_loop.campaign import campaign_graph, campaign_round
from experiments.post_training.long_loop.campaign_data import load_mix, retained_task
from experiments.post_training.long_loop.campaign_stall_e2e import run_proof


def prove_pool(mix_id: str, version: str, source_commit: str, output: str) -> dict:
    """Build only the canonical CPU dependency and verify every prepared rollout."""
    if get_job_info() is None:
        raise ValueError("The pool proof must execute in an Iris CPU job")
    configure_coreweave_s3()
    mix = load_mix(mix_id)
    agentic = [label for label, pin in mix.sources.items() if pin["route"] == "agentic"]
    assert agentic, "The proof must contain an agentic source"
    spec = campaign_round(mix, version, max_steps=50)
    if not output.startswith(spec.storage.output_prefix.rstrip("/") + "/"):
        raise ValueError("The proof receipt must use the campaign's regional output prefix")
    os.environ.update(MARIN_PREFIX=spec.storage.output_prefix, MARIN_TEMP_PREFIX=spec.storage.temporary_prefix)
    graph = campaign_graph(mix, spec)
    pools = [handle for handle in graph_handles([graph.training]) if handle.name == spec.name + "/pools"]
    assert len(pools) == 1 and not pools[0].deps
    pool = pools[0]
    run(pool, max_concurrent=1, force_run_failed=False)
    root = StoragePath(pool.path(spec.storage.output_prefix))
    payload = (root / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    assert manifest["mix"]["id"] == mix_id and manifest["sources"] == mix.sources
    assert manifest["validation"]["size"] == 200
    checked = {}
    files = {**manifest["training_files"], "validation": "validation.parquet"}
    for label, name in files.items():
        count = 0
        with (root / name).open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=64, columns=["task_spec", "task_sha256", TASK_ROLLOUT_COLUMN]):
                for row in batch.to_pylist():
                    original = retained_task(row, label)
                    lowered = TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])
                    assert lowered.task == original, "Runtime lowering changed the task or its grader"
                    assert lowered.limits.repeat_stall_batches == 3
                    count += 1
        checked[label] = count
    receipt = {
        "status": "PASS",
        "source_commit": source_commit,
        "mix_id": mix_id,
        "version": version,
        "pool_uri": str(root),
        "pool_fingerprint": pool.fingerprint(),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "agentic_sources": agentic,
        "checked_rows": checked,
        "repeat_stall_batches": 3,
        "task_and_grader_identity_preserved": True,
        "grading_e2e": asyncio.run(run_proof()),
        "gpu_jobs_submitted": 0,
    }
    destination = StoragePath(output)
    destination.parent.mkdirs()
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prove_pool(args.mix_id, args.version, args.source_commit, args.output), indent=2))


if __name__ == "__main__":
    main()
