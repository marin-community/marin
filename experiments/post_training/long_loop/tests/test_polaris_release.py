# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.long_loop.polaris_release import rewrite_cohort
from experiments.post_training.long_loop.polaris_repair import GRADER_ENVIRONMENT
from experiments.post_training.task_curation.images.build import EnvironmentArtifact, identity_digest

FIXTURE = Path(__file__).with_name("fixtures") / "polaris_plain_task.json"
BUILT = EnvironmentArtifact(path="s3://fixture/polaris-grader", identity=identity_digest(GRADER_ENVIRONMENT))


def retained_rows(count: int) -> list[dict]:
    task = TaskSpec.model_validate_json(FIXTURE.read_bytes())
    rows = []
    for index in range(count):
        identity = f"{task.id}-{index}"
        encoded = task.model_copy(update={"id": identity}).model_dump_json()
        rows.append(
            {
                "id": identity,
                "path": f"task-{index}",
                "task_spec": encoded,
                "task_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "runtime": json.dumps({"task_machine": {"cpu": 2}, "grader_machine": {"cpu": 1}}),
                "retained_extra": {"index": index, "payload": b"unchanged"},
            }
        )
    return rows


def test_repair_keeps_every_identity_runtime_reference_and_unknown_column_across_batches(tmp_path):
    rows = retained_rows(129)
    original = pa.Table.from_pylist(rows)
    source, output, inventory = (tmp_path / name for name in ("source.parquet", "tasks.parquet", "inventory.jsonl"))
    pq.write_table(original, source)
    receipt = rewrite_cohort(source, output, inventory, BUILT, len(rows))
    result = pq.read_table(output)
    assert result.schema == original.schema
    assert result.drop(["task_spec", "task_sha256"]).equals(original.drop(["task_spec", "task_sha256"]))
    ledger = [json.loads(line) for line in inventory.read_text().splitlines()]
    assert receipt.rows == len(rows) == len(ledger)
    for before, after, record in zip(rows, result.to_pylist(), ledger, strict=True):
        old = TaskSpec.model_validate_json(before["task_spec"])
        new = TaskSpec.model_validate_json(after["task_spec"])
        assert new.context == old.context
        assert new.source == old.source
        assert new.environment_requirements == old.environment_requirements
        old_config = next(resource_bytes(item) for item in old.resources.verifier if item.path == "config.json")
        new_config = next(resource_bytes(item) for item in new.resources.verifier if item.path == "config.json")
        assert new_config == old_config
        assert record["parent_task_sha256"] == before["task_sha256"]
        assert record["task_sha256"] == after["task_sha256"] == hashlib.sha256(after["task_spec"].encode()).hexdigest()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == receipt.tasks_sha256
    assert hashlib.sha256(inventory.read_bytes()).hexdigest() == receipt.inventory_sha256


@pytest.mark.parametrize("defect", ["late_checksum", "duplicate_id", "duplicate_path", "missing_population"])
def test_invalid_source_never_leaves_a_completed_candidate(tmp_path, defect):
    rows = retained_rows(65)
    if defect == "late_checksum":
        rows[-1]["task_sha256"] = "0" * 64
    elif defect == "duplicate_id":
        rows[-1].update({key: rows[0][key] for key in ("id", "task_spec", "task_sha256")})
    elif defect == "duplicate_path":
        rows[-1]["path"] = rows[0]["path"]
    else:
        rows.pop()
    source, output, inventory = (tmp_path / name for name in ("source.parquet", "tasks.parquet", "inventory.jsonl"))
    pq.write_table(pa.Table.from_pylist(rows), source)
    with pytest.raises(ValueError):
        rewrite_cohort(source, output, inventory, BUILT, 65)
    assert not output.exists()
    assert not inventory.exists()
