# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate connected RNA-seq tasks on a CPU host with the pinned R environment."""

import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

from experiments.post_training.bio_tasks.build import identity, oracle_archive, validate_instance
from experiments.post_training.bio_tasks.generators.real_rnaseq import RECIPES as DEFAULT_RECIPES
from experiments.post_training.bio_tasks.recipes import RECIPES

logger = logging.getLogger(__name__)


def validate(output: Path, lock: Path, source_revision: str, seed: int, instances: int, recipe_ids: list[str]) -> dict:
    """Check native model fits, full artifacts and rejected controls for each recipe."""
    if instances < 1:
        raise ValueError("At least one validation instance is required")
    output.mkdir(parents=True, exist_ok=False)
    lock_bytes = lock.read_bytes()
    packages = json.loads(lock_bytes)
    report = {"schema_version": 1, "source_revision": source_revision, "checks": []}
    selected = [recipe for recipe in RECIPES if recipe.id in recipe_ids]
    if not recipe_ids or len(selected) != len(recipe_ids):
        raise ValueError("Select distinct registered recipe IDs")
    for recipe in selected:
        check = {
            "repository_index": 5,
            "repository": "DESeq2",
            "recipe": recipe.id,
            "status": "pending",
            "execution": "running",
            "environment": {
                "resolved_packages": packages,
                "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            },
            "oracle_archive_sha256": hashlib.sha256(oracle_archive()).hexdigest(),
            "cases": [],
        }
        report["checks"].append(check)
        for index in range(instances):
            task = identity(recipe, seed, index)
            instance = recipe.generate(task.seed)
            case_root = output / task.task_id
            started = time.monotonic()
            try:
                grades = validate_instance(recipe, instance, case_root)
            except subprocess.CalledProcessError as error:
                logger.error("Oracle failed: %s\n%s\n%s", task.task_id, error.stdout, error.stderr)
                raise
            elapsed = time.monotonic() - started
            case = {
                "task_id": task.task_id,
                "recipe": recipe.id,
                "generation_seed": str(task.seed),
                "execution": "completed",
                "input_sha256": {
                    name: hashlib.sha256(text.encode()).hexdigest() for name, text in instance.inputs.items()
                },
                "input_bytes": sum(len(text.encode()) for text in instance.inputs.values()),
                "contract_sha256": hashlib.sha256(instance.contract.model_dump_json().encode()).hexdigest(),
                "artifact_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(case_root.iterdir())
                },
                "commands": [
                    {
                        "argv": [
                            "python",
                            "-I",
                            "oracle.pyz",
                            recipe.id,
                            "--inputs",
                            "<inputs>",
                            "--answer",
                            "<output>/answer.json",
                        ],
                        "exit_code": 0,
                        "validation_seconds": elapsed,
                    }
                ],
                "verification": {"status": "scored", "reward": grades["oracle"], "controls": grades},
                "table_rows": {name: len(table.expected) for name, table in instance.contract.tables.items()},
                "query": json.loads(instance.inputs["query.json"]),
            }
            check["cases"].append(case)
            (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
            logger.info("Validated %s in %.2f seconds", task.task_id, elapsed)
        check["status"] = "passed"
        check["execution"] = "completed"
    (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--recipes", nargs="+", default=[recipe.id for recipe in DEFAULT_RECIPES])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    validate(args.output, args.lock, args.source_revision, args.seed, args.instances, args.recipes)


if __name__ == "__main__":
    main()
