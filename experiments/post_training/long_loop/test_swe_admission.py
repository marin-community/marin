# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect CPU admission submission and complete, graded-only release selection."""

import hashlib
import json

import pytest
from fray.iris_backend import convert_container_profile
from fray.types import Entrypoint
from iris.rpc import job_pb2

from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.seal_swe_admission import admitted, load_admission
from experiments.post_training.long_loop.swe_admission_cluster import admission_job
from experiments.post_training.long_loop.swe_verified_data import RUNTIME


def test_actual_admission_request_compiles_native_privileged_profile():
    request = admission_job("swe-controls", Entrypoint.from_binary("python", ["-m", "fixture"]))
    assert convert_container_profile(request.resources) == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert request.resources.chip_count() == 0
    assert request.max_retries_failure == request.max_retries_preemption == 0


def test_complete_control_evidence_never_promotes_infra_zero_or_degenerate_task(tmp_path):
    task_ids = ["swe/good", "swe/degenerate", "swe/infra", "swe/swesmith-09581"]
    plan = {
        "submitted": True,
        "runtime_sha256": file_digest(RUNTIME),
        "shards": [{"index": 0, "sha256": "shard-pin", "task_ids": task_ids}],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "input-exclusions.json").write_text("[]")
    shard = tmp_path / "shards/0000"
    shard.mkdir(parents=True)
    complete = shard / "complete.json"
    complete.write_text(
        json.dumps({"shard_sha256": "shard-pin", "results": [{"task_id": task_id} for task_id in task_ids]})
    )
    for task_id in task_ids:
        task_root = shard / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()
        task_root.mkdir(parents=True)
        bug = {
            "status": "infra_error" if task_id.endswith("infra") else "scored",
            "reward": int(task_id.endswith("degenerate")),
        }
        receipt = {
            "task_id": task_id,
            "status": "admitted",
            "controls": {"bug": bug, "reference": {"status": "scored", "reward": 1}},
        }
        (task_root / "receipt.json").write_text(json.dumps(receipt))
        if task_id == "swe/good":
            (task_root / "bootstrap").mkdir()
            (task_root / "bootstrap/dependencies.json").write_text(json.dumps({"packages": [["pytest", "8.4.2"]]}))
    rows, _ = load_admission(str(tmp_path), workers=2)
    assert [task_id for task_id, row in rows.items() if admitted(row["receipt"])] == ["swe/good"]
    complete.unlink()
    with pytest.raises(FileNotFoundError):
        load_admission(str(tmp_path), workers=2)
