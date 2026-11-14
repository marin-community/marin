# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect frozen difficulty samples before allocating solver capacity."""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.adapter import declared_execution_limits, unsupported_harbor_contract
from taskcompendium.models import NoGrader, judge_spec

from experiments.post_training.task_curation.calibration_inventory import packages_lock_exists
from experiments.post_training.task_curation.difficulty_samples import DifficultySource
from experiments.post_training.task_curation.prebuilt import validate_native_package
from experiments.post_training.task_curation.sample_qc import panel_tasks

LOGGER = logging.getLogger(__name__)


def inspect_source(source: DifficultySource, output_root: str) -> dict:
    tasks, contracts = panel_tasks(source.panel)
    rows = []
    for task, contract in zip(tasks, contracts, strict=True):
        native = isinstance(task.grader, NoGrader) and "harbor" in task.tags
        if native:
            validate_native_package(task)
        judge = judge_spec(task.grader)
        environments = [task.environment_requirements]
        if not isinstance(task.grader, NoGrader):
            environments.append(task.grader.environment)
        locks = sorted(
            {
                environment.packages_lock
                for environment in environments
                if environment is not None and environment.packages_lock is not None
            }
        )
        missing_locks = [path for path in locks if not packages_lock_exists(path)]
        rows.append(
            {
                "task_id": task.id,
                "task_sha256": contract["task_sha256"],
                "workload": "native" if native else "task_spec",
                "unsupported_contract": None if native else unsupported_harbor_contract(task),
                "grader_type": type(task.grader).__name__,
                "original_packages_locks": locks,
                "missing_packages_locks": missing_locks,
                "judge": None if judge is None else asdict(judge),
                "declared_limits": declared_execution_limits(task),
                "environment": task.environment_requirements.model_dump(mode="json"),
            }
        )
    result = {
        "source_id": source.declaration.id,
        "backend": source.backend,
        "sample_receipt": source.sample_receipt.model_dump(mode="json"),
        "pre_oracle_filter": source.sample.pre_oracle_filter,
        "tasks": rows,
    }
    target = StoragePath(output_root) / source.sample_receipt.sha256 / "contracts.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Inspected %s: %d tasks", source.declaration.id, len(rows))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    if output.exists():
        raise FileExistsError(output)
    source_path = StoragePath(args.sources)
    if file_sha256(source_path) != args.sources_sha256:
        raise ValueError("Difficulty queue differs from its frozen digest")
    sources = [DifficultySource.model_validate(row) for row in json.loads(source_path.read_text())]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(partial(inspect_source, output_root=args.output_root), sources))
    reasons = Counter(
        task["unsupported_contract"] or "runnable_contract" for source in results for task in source["tasks"]
    )
    summary = {
        "sources_uri": args.sources,
        "sources_sha256": args.sources_sha256,
        "sources": results,
        "contract_counts": dict(reasons),
        "environment_qualified": False,
    }
    (output / "contracts.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Contract counts: %s", dict(reasons))


if __name__ == "__main__":
    main()
