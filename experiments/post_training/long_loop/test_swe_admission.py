# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect CPU admission submission and complete, graded-only release selection."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.iris_backend import convert_container_profile
from fray.types import Entrypoint
from iris.rpc import job_pb2

from experiments.post_training.long_loop.admission_index import finalize_admission
from experiments.post_training.long_loop.repartition_swe_admission import recovery_inputs
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.long_loop.seal_swe_admission import admitted, load_admission
from experiments.post_training.long_loop.swe_admission_cluster import admission_job, publish_directory
from experiments.post_training.long_loop.swe_verified_data import RECIPE_VERSION, RUNTIME


@pytest.fixture
def grader_profiles():
    return {file_digest(RUNTIME): {"uri": str(RUNTIME), "version": RECIPE_VERSION}}


def test_actual_admission_request_compiles_native_privileged_profile():
    request = admission_job("swe-controls", Entrypoint.from_binary("python", ["-m", "fixture"]))
    assert convert_container_profile(request.resources) == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert request.resources.chip_count() == 0
    assert request.max_retries_failure == request.max_retries_preemption == 0


def test_complete_control_evidence_never_promotes_infra_zero_or_degenerate_task(tmp_path, grader_profiles):
    task_ids = ["swe/good", "swe/degenerate", "swe/infra", "swe/swesmith-09581"]
    plan = {
        "submitted": True,
        "manifest_sha256": "corpus-pin",
        "runtime_sha256": file_digest(RUNTIME),
        "shards": [{"index": 0, "sha256": "shard-pin", "task_ids": task_ids}],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "input-exclusions.json").write_text("[]")
    shard = tmp_path / "shards/0000"
    shard.mkdir(parents=True)
    for task_id in task_ids:
        task_root = shard / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()
        staged = tmp_path / "staged" / hashlib.sha256(task_id.encode()).hexdigest()
        staged.mkdir(parents=True)
        bug = {
            "status": "infra_error" if task_id.endswith("infra") else "scored",
            "reward": int(task_id.endswith("degenerate")),
        }
        receipt = {
            "task_id": task_id,
            "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
            "status": "admitted",
            "controls": {"bug": bug, "reference": {"status": "scored", "reward": 1}},
        }
        (staged / "receipt.json").write_text(json.dumps(receipt))
        if task_id == "swe/good":
            (staged / "bootstrap").mkdir()
            (staged / "bootstrap/dependencies.json").write_text(json.dumps({"packages": [["pytest", "8.4.2"]]}))
        publish_directory(staged, str(task_root))
    finalize_admission([str(tmp_path)], str(tmp_path), workers=2, profiles=grader_profiles, replacement_sources=())
    rows, _ = load_admission(str(tmp_path), workers=2)
    assert [task_id for task_id, row in rows.items() if admitted(row["receipt"])] == ["swe/good"]
    dependency = shard / "tasks" / hashlib.sha256(b"swe/good").hexdigest() / "bootstrap/dependencies.json"
    dependency_raw = dependency.read_bytes()
    dependency.write_text("{}")
    with pytest.raises(ValueError, match="dependency metadata"):
        load_admission(str(tmp_path), workers=2)
    dependency.write_bytes(dependency_raw)
    (task_root / "receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_admission(str(tmp_path), workers=2)


def test_recovery_index_preserves_completed_receipts_and_rejects_duplicate_commits(tmp_path, grader_profiles):
    original, recovery = tmp_path / "original", tmp_path / "recovery"
    for root, ids in ((original, ["swe/one", "swe/two"]), (recovery, ["swe/two"])):
        root.mkdir()
        (root / "plan.json").write_text(
            json.dumps(
                {
                    "submitted": True,
                    "manifest_sha256": "corpus-pin",
                    "runtime_sha256": file_digest(RUNTIME),
                    "shards": [{"index": 0, "task_ids": ids}],
                }
            )
        )
        (root / "input-exclusions.json").write_text("[]")
    for root, task_id in ((original, "swe/one"), (recovery, "swe/two")):
        staged = tmp_path / task_id.replace("/", "-")
        staged.mkdir()
        (staged / "receipt.json").write_text(
            json.dumps(
                {"task_id": task_id, "status": "excluded", "task_sha256": hashlib.sha256(task_id.encode()).hexdigest()}
            )
        )
        key = hashlib.sha256(task_id.encode()).hexdigest()
        publish_directory(staged, str(root / "shards/0000/tasks" / key))
    sources = [str(original), str(recovery)]
    index = finalize_admission(
        sources, str(tmp_path / "final"), workers=2, profiles=grader_profiles, replacement_sources=()
    )
    assert [row["task_id"] for row in index["tasks"]] == ["swe/one", "swe/two"]
    assert index["tasks"][0]["uri"].startswith(str(original))
    assert index["tasks"][1]["uri"].startswith(str(recovery))
    duplicate = original / "shards/0000/tasks" / hashlib.sha256(b"swe/two").hexdigest()
    publish_directory(tmp_path / "swe-two", str(duplicate))
    with pytest.raises(ValueError, match="Duplicate committed"):
        finalize_admission(
            sources, str(tmp_path / "duplicate"), workers=2, profiles=grader_profiles, replacement_sources=()
        )


def test_repartition_requires_terminal_workers_and_preserves_existing_evidence(tmp_path):
    source, admission = tmp_path / "source", tmp_path / "admission"
    source.mkdir()
    admission.mkdir()
    files = {}
    for name, ids in (("agentic-train.parquet", ["swe/one", "swe/two"]), ("validation.parquet", ["swe/three"])):
        pq.write_table(
            pa.Table.from_pylist([{"task_spec": json.dumps({"id": task_id})} for task_id in ids]), source / name
        )
        files[name] = {"sha256": file_digest(source / name)}
    (source / "manifest.json").write_text(json.dumps({"files": files}))
    (admission / "plan.json").write_text(
        json.dumps(
            {
                "submitted": True,
                "manifest_sha256": file_digest(source / "manifest.json"),
                "runtime_sha256": file_digest(RUNTIME),
                "shards": [{"index": 0, "job_id": "/owned/slow", "task_ids": ["swe/one", "swe/two", "swe/three"]}],
            }
        )
    )
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "receipt.json").write_text(
        json.dumps({"task_id": "swe/one", "status": "excluded", "task_sha256": hashlib.sha256(b"swe/one").hexdigest()})
    )
    committed = admission / "shards/0000/tasks" / hashlib.sha256(b"swe/one").hexdigest()
    publish_directory(staged, str(committed))
    partial = admission / "shards/0000/tasks" / hashlib.sha256(b"swe/two").hexdigest()
    partial.mkdir(parents=True)
    (partial / "receipt.json").write_text("uncommitted bytes")
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps({"/owned/slow": {"state": "running"}}))
    output = tmp_path / "recovery"
    with pytest.raises(ValueError, match="must be terminal"):
        recovery_inputs(source, str(admission), [0], terminal, output, workers=2)
    assert not output.exists()
    terminal.write_text(json.dumps({"/owned/slow": {"state": "killed"}}))
    result = recovery_inputs(source, str(admission), [0], terminal, output, workers=2)
    assert result["pending_task_ids"] == ["swe/three", "swe/two"]
    assert [row["task_id"] for row in result["preserved_commits"]] == ["swe/one"]
    assert (partial / "receipt.json").read_text() == "uncommitted bytes"
    assert json.loads((committed / "receipt.json").read_text())["task_id"] == "swe/one"
    assert pq.read_table(output / "agentic-train.parquet").to_pylist() == [{"task_spec": json.dumps({"id": "swe/two"})}]


def test_repaired_profile_supersedes_only_excluded_task_with_new_exact_package(tmp_path, grader_profiles):
    original, repaired = tmp_path / "original", tmp_path / "repaired"
    old_runtime = tmp_path / "old-grader.py"
    old_runtime.write_text("# preserved admitted guest bytes\n")
    old_sha = file_digest(old_runtime)
    profiles = {**grader_profiles, old_sha: {"uri": str(old_runtime), "version": "preserved"}}
    for root, runtime_sha, status in ((original, old_sha, "excluded"), (repaired, file_digest(RUNTIME), "admitted")):
        root.mkdir()
        (root / "plan.json").write_text(
            json.dumps(
                {
                    "submitted": True,
                    "manifest_sha256": "corpus",
                    "runtime_sha256": runtime_sha,
                    "shards": [{"index": 0, "task_ids": ["swe/task"]}],
                }
            )
        )
        (root / "input-exclusions.json").write_text("[]")
        staged = root / "staged"
        staged.mkdir()
        (staged / "receipt.json").write_text(
            json.dumps({"task_id": "swe/task", "status": status, "task_sha256": runtime_sha})
        )
        publish_directory(staged, str(root / "shards/0000/tasks" / hashlib.sha256(b"swe/task").hexdigest()))
    index = finalize_admission([str(original), str(repaired)], str(tmp_path / "final"), 2, profiles, (str(repaired),))
    assert index["tasks"][0]["runtime_sha256"] == file_digest(RUNTIME)
    assert index["superseded"][0]["runtime_sha256"] == old_sha
    receipt_path = original / "staged/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "admitted"
    receipt_path.write_text(json.dumps(receipt))
    publish_directory(original / "staged", str(original / "shards/0000/tasks" / hashlib.sha256(b"swe/task").hexdigest()))
    with pytest.raises(ValueError, match="Duplicate committed"):
        finalize_admission([str(original), str(repaired)], str(tmp_path / "invalid"), 2, profiles, (str(repaired),))
