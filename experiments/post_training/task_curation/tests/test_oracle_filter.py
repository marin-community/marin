# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full Harbor trial receipts preserve oracle exits and filter exact converted bytes."""

import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialCompletion, TrialResult
from harbor.models.verifier.result import VerifierResult
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_filter import filter_cohort

from .test_prebuilt import _task


def test_full_oracle_filter_excludes_failed_solutions_even_with_reward_one(tmp_path):
    rows = []
    mappings = {}
    for task_id in ("pass", "failed-solution", "infra"):
        row = _task(task_id)
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
            config = TrialConfig(
                task=TaskConfig(path=tmp_path / "native" / task_id),
                agent=AgentConfig(name="oracle"),
                environment=EnvironmentConfig(type="local"),
                trials_dir=tmp_path / "jobs",
                trial_name=root.name,
            )
            result = TrialResult(
                task_name=task_id,
                trial_name=root.name,
                trial_uri=root.as_uri(),
                task_id=config.task.get_task_id(),
                task_checksum=checksum,
                config=config,
                agent_info=AgentInfo(name="oracle", version="1"),
                verifier_result=VerifierResult(rewards={"reward": 1}),
                finished_at=datetime(2026, 11, 12, tzinfo=UTC),
                completion=TrialCompletion(trajectory_present=False),
                exception_info=(
                    ExceptionInfo(
                        exception_type="RuntimeError",
                        exception_message="backend unavailable",
                        exception_traceback="",
                        occurred_at=datetime(2026, 11, 12, tzinfo=UTC),
                    )
                    if task_id == "infra"
                    else None
                ),
            )
            (root / "result.json").write_text(result.model_dump_json())
            (root / "verifier/verdict.json").write_text(json.dumps({"status": "scored", "reward": 1}))
            if task_id == "failed-solution":
                (root / "agent/exit-code.txt").write_text("7")
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    task_map = tmp_path / "task-map.json"
    task_map.write_text(json.dumps(mappings))
    output = tmp_path / "filtered/tasks.parquet"
    receipt = filter_cohort(cohort, task_map, tmp_path / "jobs", output)
    assert [row["path"] for row in pq.read_table(output).to_pylist()] == ["pass"]
    assert receipt.input_rows == 3
    assert receipt.tasks["failed-solution"].trials[0].oracle_exit_code == 7
    assert not receipt.tasks["failed-solution"].passed
    assert not receipt.tasks["infra"].passed

    envelope = json.loads(rows[0]["lowered_task_spec"])
    envelope["task"]["labels"] = {"changed": "after Harbor export"}
    rows[0]["lowered_task_spec"] = json.dumps(envelope)
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    with pytest.raises(ValueError, match="changed after Harbor export"):
        filter_cohort(cohort, task_map, tmp_path / "jobs", tmp_path / "stale/tasks.parquet")
