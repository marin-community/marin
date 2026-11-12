# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full Harbor trial receipts preserve oracle exits and filter exact converted bytes."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_filter import filter_cohort

from .prebuilt_fixtures import prebuilt_task


def test_full_oracle_filter_excludes_failed_solutions_even_with_reward_one(tmp_path):
    rows = []
    mappings = {}
    for task_id in ("pass", "failed-solution", "missing-exit", "infra"):
        row = prebuilt_task(task_id)
        envelope = json.loads(row["lowered_task_spec"])
        task = TaskSpec.model_validate(envelope["task"])
        task = task.model_copy(
            update={
                "resources": task.resources.model_copy(
                    update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)}
                )
            }
        )
        envelope["task"] = task.model_dump(mode="json")
        row["lowered_task_spec"] = json.dumps(envelope)
        rows.append(row)
        checksum = canonical_sha256({"native": task_id})
        mappings[task_id] = {
            "task_name": task_id,
            "task_checksum": checksum,
            "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            "runtime_sha256": canonical_sha256(envelope["runtime"]),
        }
        for attempt in range(3 if task_id == "infra" else 1):
            root = tmp_path / "jobs" / f"{task_id}-{attempt}"
            (root / "verifier").mkdir(parents=True)
            (root / "agent").mkdir()
            task_path = str(tmp_path / "native" / task_id)
            result = {
                "id": str(uuid4()),
                "task_name": task_id,
                "trial_name": root.name,
                "trial_uri": root.as_uri(),
                "task_id": {"path": task_path},
                "task_checksum": checksum,
                "config": {
                    "task": {"path": task_path},
                    "agent": {"name": "oracle"},
                    "environment": {"type": "local"},
                    "trials_dir": str(tmp_path / "jobs"),
                    "trial_name": root.name,
                },
                "agent_info": {"name": "oracle", "version": "1"},
                "verifier_result": {"rewards": {"reward": 1}},
                "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                "completion": {"trajectory_present": False},
                "exception_info": (
                    {
                        "exception_type": "RuntimeError",
                        "exception_message": "backend unavailable",
                        "exception_traceback": "",
                        "occurred_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                    }
                    if task_id == "infra"
                    else None
                ),
            }
            (root / "result.json").write_text(json.dumps(result))
            (root / "verifier/verdict.json").write_text(json.dumps({"status": "scored", "reward": 1}))
            if task_id != "missing-exit":
                (root / "agent/exit-code.txt").write_text("7" if task_id == "failed-solution" else "0")
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    task_map = tmp_path / "task-map.json"
    task_map.write_text(json.dumps(mappings))
    output = tmp_path / "filtered/tasks.parquet"
    receipt = filter_cohort(cohort, task_map, tmp_path / "jobs", output)
    assert [row["path"] for row in pq.read_table(output).to_pylist()] == ["pass"]
    assert receipt.input_rows == 4
    assert receipt.tasks["failed-solution"].trials[0].oracle_exit_code == 7
    assert not receipt.tasks["failed-solution"].passed
    assert not receipt.tasks["infra"].passed
    assert receipt.tasks["missing-exit"].trials[0].oracle_exit_code is None
    assert not receipt.tasks["missing-exit"].passed

    envelope = json.loads(rows[0]["lowered_task_spec"])
    envelope["task"]["labels"] = {"changed": "after Harbor export"}
    rows[0]["lowered_task_spec"] = json.dumps(envelope)
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    with pytest.raises(ValueError, match="changed after Harbor export"):
        filter_cohort(cohort, task_map, tmp_path / "jobs", tmp_path / "stale/tasks.parquet")
