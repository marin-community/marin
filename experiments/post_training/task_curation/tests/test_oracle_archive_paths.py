# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archived native trials remain verifiable outside their producer's filesystem."""

import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_filter import filter_cohort, harbor_inspection, write_task_map
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy


@pytest.mark.parametrize("oracle_exit", [0, 7])
def test_filter_reads_relocated_archive_without_rewriting_results(tmp_path, oracle_exit):
    original = tmp_path / "producer"
    native = original / "native" / "relocated"
    for directory in ("environment", "tests", "solution"):
        (native / directory).mkdir(parents=True)
    (native / "instruction.md").write_text("Run the task.\n")
    (native / "tests/test.sh").write_text("true\n")
    (native / "solution/solve.sh").write_text("true\n")
    (native / "task.toml").write_text(
        "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
        "[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
    )
    row = prebuilt_task("relocated")
    envelope = json.loads(row["lowered_task_spec"])
    task = TaskSpec.model_validate(envelope["task"])
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)})
        }
    )
    envelope["task"] = task.model_dump(mode="json")
    row["lowered_task_spec"] = json.dumps(envelope)
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist([row]), cohort)
    mapping = tmp_path / "task-map.json"
    write_task_map(cohort, native.parent, mapping)
    checksum = harbor_inspection("tasks", [str(native)])[str(native)]["task_checksum"]
    trial = original / "jobs" / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    result = {
        "id": str(uuid4()),
        "task_name": "relocated",
        "trial_name": "trial",
        "trial_uri": trial.as_uri(),
        "task_id": {"path": str(native)},
        "task_checksum": checksum,
        "config": {
            "task": {"path": str(native)},
            "agent": {"name": "oracle"},
            "environment": {"type": "local"},
            "trials_dir": str(original / "jobs"),
            "trial_name": "trial",
            "resolved_timeouts": {"agent": 300, "verifier": 60, "agent_setup": 360, "environment_build": 600},
        },
        "agent_info": {"name": "oracle", "version": "1"},
        "verifier_result": {"rewards": {"reward": 1}},
        "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
        "completion": {"trajectory_present": False},
        "exception_info": None,
    }
    raw = json.dumps(result).encode()
    (trial / "result.json").write_bytes(raw)
    (trial / "config.json").write_text(json.dumps(result["config"]))
    (trial / "agent/exit-code.txt").write_text(str(oracle_exit))
    (trial / "verifier/verdict.json").write_text('{"status":"scored","reward":1}')
    archive = tmp_path / "pair.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(original, arcname="pair")
    shutil.rmtree(original)
    restored = tmp_path / "different-root"
    with tarfile.open(archive) as bundle:
        bundle.extractall(restored, filter="data")
    assert (restored / "pair/native/relocated/environment").is_dir()
    decision = tmp_path / "policy.json"
    decision.write_bytes(b'{"scope":"global_workspace"}\n')
    policy = negative_control_policy().model_copy(
        update={
            "decision": ContentEvidenceReference(
                uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest()
            )
        }
    )
    receipt = filter_cohort(
        cohort,
        mapping,
        original / "jobs",
        tmp_path / "filtered/tasks.parquet",
        archive_roots={str(original): str(restored / "pair")},
        negative_control_policy=policy,
    )
    entry = receipt.tasks[task.id]
    assert entry.calibration.passed is (oracle_exit == 0)
    assert entry.calibration.attempts[0].result == result
    assert entry.calibration.attempts[0].result_sha256 == canonical_sha256(result)
    assert entry.calibration.attempts[0].path == str(trial / "result.json")
    assert (restored / "pair/jobs/trial/result.json").read_bytes() == raw
    assert not original.exists()
