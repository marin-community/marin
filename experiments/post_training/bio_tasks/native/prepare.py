# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare native-tool checks from a validated corpus; does not install or run tools."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.post_training.bio_tasks.native.check import OPERATIONS

SOURCE = Path(__file__).resolve().parents[1]


def prepare(corpus: Path, output: Path) -> dict:
    """Copy public inputs and separate private references, preserving corpus task IDs."""
    manifest = json.loads((corpus / "manifest.json").read_text())
    if manifest["status"] != "validated_locally":
        raise ValueError("native validation requires a locally validated corpus")
    ledger = [json.loads(line) for line in (corpus / "ledger.jsonl").read_text().splitlines()]
    inventory = json.loads((SOURCE / "repository_coverage.json").read_text())["repositories"]
    packages = json.loads((SOURCE / "tool_environments.json").read_text())["packages"]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for repository in inventory:
        index = repository["index"]
        row = {"repository_index": index, "repository": repository["name"], "execution": "pending", "cases": []}
        operation = OPERATIONS.get(index)
        if operation is None:
            row["adapter"] = "pending"
            rows.append(row)
            continue
        row["adapter"] = "implemented_unvalidated"
        row["recipe"] = operation.recipe
        row["package_specs"] = [
            f'{p["channel"]}::{p["package"]}={p["version"]}' for p in packages if p["repository_index"] == index
        ]
        row["package_specs"] += ["python=3.12", *operation.packages]
        candidates = [task for task in ledger if task["recipe"] == operation.recipe]
        if len(candidates) != 3:
            raise ValueError(f"Expected three corpus tasks for {operation.recipe}")
        for task in candidates:
            task_id = task["task_id"]
            source = corpus / "harbor" / "train" / task_id
            case = output / "cases" / f"{index:02d}" / task_id
            shutil.copytree(source / "setup_files" / "inputs", case / "inputs")
            private = output / "references" / f"{index:02d}" / task_id
            private.mkdir(parents=True)
            shutil.copyfile(source / "tests" / "reference.json", private / "reference.json")
            row["cases"].append(
                {
                    "task_id": task_id,
                    "inputs": str((case / "inputs").relative_to(output)),
                    "reference": str((private / "reference.json").relative_to(output)),
                }
            )
        rows.append(row)
    code = output / "code" / "experiments" / "post_training" / "bio_tasks"
    code.mkdir(parents=True)
    for directory in (code, code.parent, code.parent.parent):
        (directory / "__init__.py").write_text("")
    shutil.copytree(SOURCE / "native", code / "native", ignore=shutil.ignore_patterns("__pycache__"))
    (code / "solvers").mkdir()
    (code / "solvers" / "__init__.py").write_text("")
    shutil.copyfile(SOURCE / "solvers" / "formats.py", code / "solvers" / "formats.py")
    report = {
        "schema_version": 1,
        "corpus_manifest_sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        "expected_repositories": 50,
        "repositories": rows,
        "limits": {"concurrent_repositories": 1, "instances_per_repository": 3, "tool_command_seconds": 300},
        "status": "prepared; no native tools executed",
    }
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.corpus, args.output)
    print(
        json.dumps(
            {
                "repositories": len(report["repositories"]),
                "implemented_adapters": sum(row["adapter"] != "pending" for row in report["repositories"]),
                "cases": sum(len(row["cases"]) for row in report["repositories"]),
            }
        )
    )


if __name__ == "__main__":
    main()
