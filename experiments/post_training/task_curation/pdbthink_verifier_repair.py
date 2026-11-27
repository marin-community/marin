# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply PDBThink's versioned, user-approved Daytona verifier allocation."""

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.licenses import require_license
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.pdbthink.tasks import DATASET, REVISION
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.remediation_publish import TASK_SCHEMA, verify_retained
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

SOURCE = "OpenAthena:pdbthink-coordinate-tasks"
NATIVE_CONFIGURATION = "source_archive/task.toml"


class VerifierAllocationDeviation(BaseModel):
    """The approved change from the archived native grader allocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["2026.11.17.1"] = "2026.11.17.1"
    authority: Literal["user ruling 2026-11-17"] = "user ruling 2026-11-17"
    ruling: Literal["/Users/benfeuer/Documents/experiments/active/atlas-rulings/pdbthink-daytona-limits-20261117.md"] = (
        "/Users/benfeuer/Documents/experiments/active/atlas-rulings/pdbthink-daytona-limits-20261117.md"
    )
    original: ExecutionRequirements
    approved: ExecutionRequirements


def allocation_deviation(native: ExecutionRequirements) -> VerifierAllocationDeviation:
    if (native.cpus, native.memory_mb, native.storage_mb) != (1, 512, 1024):
        raise ValueError("PDBThink ruling requires the original 1 CPU / 512 MiB / 1024 MiB allocation")
    return VerifierAllocationDeviation(original=native, approved=native.model_copy(update={"memory_mb": 1024}))


def repair_task(task: TaskSpec) -> TaskSpec:
    """Apply the approved grader allocation while preserving the source task."""
    if (task.source.dataset, task.source.revision) != (DATASET, REVISION):
        raise ValueError("Verifier repair requires the original pinned PDBThink source")
    if not is_nonagentic_task(task) or task.completion_protocol != "tool_free_single":
        raise ValueError("Verifier repair requires PDBThink's original tool-free answer contract")
    if task.verifier_execution_requirements is not None:
        raise ValueError("PDBThink verifier allocation is already declared")
    configurations = [resource for resource in task.resources.verifier if resource.path == NATIVE_CONFIGURATION]
    if len(configurations) != 1:
        raise ValueError("PDBThink repair requires exactly one original native configuration")
    native = native_execution_limits(tomllib.loads(resource_bytes(configurations[0]).decode())).requirements
    if native is None or task.execution_requirements != native:
        raise ValueError("PDBThink task limits differ from its archived original configuration")
    return task.model_copy(update={"verifier_execution_requirements": allocation_deviation(native).approved})


def repair_cohort(source: Path, output: Path, *, expected_sha256: str, expected_tasks: int) -> dict:
    """Write the source-specific projection and a complete before/after hash ledger."""
    if file_sha256(StoragePath(str(source))) != expected_sha256:
        raise ValueError("PDBThink input Parquet differs from its pinned publication")
    output.mkdir(parents=True, exist_ok=False)
    tasks = output / "tasks.parquet"
    ledger = output / "verifier-repair.jsonl"
    expected: dict[str, tuple[str, str]] = {}
    allocations: Counter[str] = Counter()
    deviations: dict[str, VerifierAllocationDeviation] = {}
    with pq.ParquetWriter(tasks, TASK_SCHEMA, compression="zstd") as writer, ledger.open("w") as evidence:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=4):
            rows = []
            for row in batch.to_pylist():
                document = json.loads(row["task_spec"])
                original = TaskSpec.model_validate_json(json.dumps(document))
                before = hashlib.sha256(row["task_spec"].encode()).hexdigest()
                if row["id"] != original.id or row["task_sha256"] != before or original.id in expected:
                    raise ValueError("PDBThink publication repeats a task or has a stale task hash")
                repaired = repair_task(original)
                allocation = repaired.verifier_execution_requirements
                assert allocation is not None
                assert original.execution_requirements is not None
                deviation = allocation_deviation(original.execution_requirements)
                deviations[deviation.model_dump_json()] = deviation
                if repaired.model_dump(mode="json", exclude={"verifier_execution_requirements"}) != document:
                    raise ValueError("PDBThink repair would change other source task fields")
                encoded = repaired.model_dump_json()
                after = hashlib.sha256(encoded.encode()).hexdigest()
                runtime_sha256 = hashlib.sha256(row["runtime"].encode()).hexdigest()
                expected[original.id] = (after, runtime_sha256)
                allocations[allocation.model_dump_json()] += 1
                rows.append({**row, "task_spec": encoded, "task_sha256": after})
                evidence.write(
                    json.dumps(
                        {
                            "id": original.id,
                            "original_task_sha256": before,
                            "repaired_task_sha256": after,
                            "runtime_sha256": runtime_sha256,
                            "allocation_deviation": deviation.model_dump(mode="json"),
                        }
                    )
                    + "\n"
                )
            writer.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA))
    if len(expected) != expected_tasks or file_sha256(StoragePath(str(source))) != expected_sha256:
        raise ValueError("PDBThink repair does not cover its frozen input population")
    content_sha256 = verify_retained(tasks, expected)
    receipt = {
        "source_id": SOURCE,
        "original_dataset": DATASET,
        "original_revision": REVISION,
        "input_parquet": str(source),
        "input_sha256": expected_sha256,
        "tasks": len(expected),
        "tasks_parquet_sha256": file_sha256(StoragePath(str(tasks))),
        "retained_content_sha256": content_sha256,
        "ledger_sha256": file_sha256(StoragePath(str(ledger))),
        "verifier_allocation_counts": dict(allocations),
        "allocation_deviations": [deviation.model_dump(mode="json") for deviation in deviations.values()],
        "changed_fields": ["verifier_execution_requirements"],
        "scope": "Versioned grader allocation deviation; no oracle, quality or admission qualification",
    }
    (output / "verifier-repair.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = read_license_receipts(LICENSE_RECEIPT_INDEX)[SOURCE]
    require_license(receipt, SOURCE, "train", ban=None)
    require_unbanned(SOURCE, DATASET, receipt)
    repair_cohort(
        args.input_parquet, args.output_dir, expected_sha256=args.input_sha256, expected_tasks=args.expected_tasks
    )


if __name__ == "__main__":
    main()
