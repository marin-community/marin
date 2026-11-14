# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the separate small-source authority on actual completed source trials."""

import gzip
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.legacy_roles import LegacySmallRoleScope
from taskcompendium.pipeline.oracle_filter import (
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleTaskReceipt,
    ResourceCalibrationReceipt,
    load_oracle_filter,
    negative_control_applicability,
    require_oracle_filter,
)

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.legacy_role_qualification import (
    LegacyRoleQualificationEvidence,
    legacy_role_qualifications,
)
from experiments.post_training.task_curation.oracle_filter import filter_cohort, write_task_map
from experiments.post_training.task_curation.tests.test_legacy_role_qualification import FIXTURES as OLD_FIXTURES
from experiments.post_training.task_curation.tests.test_legacy_role_qualification import reference
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/legacy_small_roles"


@pytest.fixture
def small_evidence(tmp_path):
    inputs = json.loads(gzip.decompress((FIXTURES / "inputs.json.gz").read_bytes()))
    audit = json.loads(gzip.decompress((FIXTURES / "audit.json.gz").read_bytes()))
    for entry, row in zip(inputs["entries"], audit["rows"], strict=True):
        name = entry.pop("fixture_name")
        row.pop("fixture_name")
        receipt = tmp_path / f"{name}.json"
        receipt.write_bytes(gzip.decompress((FIXTURES / f"{name}.receipt.json.gz").read_bytes()))
        entry.update(receipt_uri=str(receipt), trial_archive_uri=str(FIXTURES / f"{name}.tar.gz"))
        row.update(entry)
    for kind, name in (("producer_manifest", "producer-manifest.json"), ("closure", "closure.json")):
        inputs[f"{kind}_uri"] = str(OLD_FIXTURES / name)
    index = tmp_path / "index.json"
    index.write_text(json.dumps(inputs))
    audit["input_partition_sha256"] = reference(index).sha256
    partition = tmp_path / "partition-v2.json"
    partition.write_text(json.dumps(audit))
    ruling = tmp_path / "ruling.md"
    ruling.write_text("Fixture authority for two frozen passes and one retained scored failure.\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "producer_revision": inputs["backend_revision"],
                "compatible_completed_passes": 2,
                "scored_failure_held_unchanged": 1,
                "providers": 0,
                "source_commands": 0,
                "files": [{"path": p.name, "sha256": reference(p).sha256} for p in (index, partition)],
            }
        )
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "version": 1,
                "disposition": "adopted",
                "producer_revision": inputs["backend_revision"],
                "input_index_sha256": reference(index).sha256,
                "input_partition_sha256": reference(partition).sha256,
                "audit_sha256": reference(manifest).sha256,
                "adopted_count": 2,
                "excluded_scored_failures": 1,
                "ruling": {"path": "/original/coordinator/ruling.md", "sha256": reference(ruling).sha256},
            }
        )
    )
    return LegacyRoleQualificationEvidence(
        input_partition=reference(index),
        audit=reference(partition),
        decision=reference(decision),
        small_scope=LegacySmallRoleScope(audit_manifest=reference(manifest), ruling=reference(ruling)),
    )


def test_small_adoption_checks_complete_partition_and_keeps_original_failure(tmp_path, small_evidence):
    inputs = json.loads(Path(small_evidence.input_partition.uri).read_text())
    ids = frozenset(e["task_id"] for e in inputs["entries"])
    qualified = legacy_role_qualifications(small_evidence, ids)
    assert set(qualified) == {e["task_id"] for e in inputs["entries"] if e["passed"]}
    for proof in qualified.values():
        assert proof.model_validate_json(proof.model_dump_json()) == proof
        assert proof.small_scope == small_evidence.small_scope
        assert proof.adoption_decision.ruling.path == "/original/coordinator/ruling.md"
    for name, source in zip(("unitsyn", "pym"), inputs["entries"][:2], strict=True):
        mapping = json.loads(gzip.decompress((FIXTURES / f"{name}.receipt.json.gz").read_bytes()))["receipt"]
        with tarfile.open(FIXTURES / f"{name}.tar.gz") as archive:
            original = TaskSpec.model_validate_json(archive.extractfile("original-packet/task-spec.json").read())
            native = archive.extractfile("original-packet/task.toml").read()
        calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
        qualification = qualified[source["task_id"]]
        task = calibrated_task_spec(
            original, calibration.execution_requirements, qualification.verifier.execution_requirements, native
        )
        entry = OracleTaskReceipt(
            task_sha256=canonical_sha256(task.model_dump(mode="json")),
            source_task_sha256=mapping["task_sha256"],
            source_task_checksum=mapping["source_task_checksum"],
            runtime_sha256=mapping["runtime_sha256"],
            harbor_task_checksum=mapping["task_checksum"],
            has_oracle=True,
            trials=(calibration.attempts[-1],),
            calibration=calibration,
            legacy_role_qualification=qualification,
            negative_control_applicability=negative_control_applicability(task),
            exclusion_reason=OracleExclusionReason.NEGATIVE_CONTROL_PENDING,
        )
        receipt = OracleFilterReceipt(
            negative_control_policy=negative_control_policy(),
            recipe_identity=None,
            input_cohort_sha256=qualification.input_cohort_sha256,
            input_rows=1,
            conversion_rejections=0,
            tasks={task.id: entry},
        )
        path = tmp_path / f"{name}-filter.json"
        path.write_text(receipt.model_dump_json())
        loaded = load_oracle_filter(str(path))
        assert not require_oracle_filter(task, loaded, qualification.source_runtime.runtime)
        assert loaded.tasks[task.id].calibration.model_dump_json() == calibration.model_dump_json()
        for field in ("audit_manifest", "ruling"):
            foreign = receipt.model_dump(mode="json")
            foreign["tasks"][task.id]["legacy_role_qualification"]["small_scope"][field]["sha256"] = "0" * 64
            path.write_text(json.dumps(foreign))
            with pytest.raises(ValueError, match="adoption decision scope"):
                load_oracle_filter(str(path))
    # Selecting one pass must still verify the excluded member's physical bytes.
    failure = next(e for e in inputs["entries"] if not e["passed"])
    Path(failure["receipt_uri"]).write_bytes(b"foreign scored outcome")
    with pytest.raises(ValueError, match="evidence bytes changed"):
        legacy_role_qualifications(small_evidence, frozenset({inputs["entries"][0]["task_id"]}))


def test_public_load_retains_actual_scored_negative_without_allocation_inference(tmp_path):
    mapping = json.loads(gzip.decompress((FIXTURES / "pym-failure.receipt.json.gz").read_bytes()))["receipt"]
    with tarfile.open(FIXTURES / "pym-failure.tar.gz") as archive:
        task = TaskSpec.model_validate_json(archive.extractfile("original-packet/task-spec.json").read())
    calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
    entry = OracleTaskReceipt(
        task_sha256=mapping["task_sha256"],
        source_task_sha256=mapping["task_sha256"],
        source_task_checksum=mapping["source_task_checksum"],
        runtime_sha256=mapping["runtime_sha256"],
        harbor_task_checksum=mapping["task_checksum"],
        has_oracle=True,
        trials=(calibration.attempts[-1],),
        calibration=calibration,
        negative_control_applicability=negative_control_applicability(task),
        exclusion_reason=OracleExclusionReason.ORACLE_FAILED,
    )
    receipt = OracleFilterReceipt(
        negative_control_policy=negative_control_policy(),
        recipe_identity=None,
        input_cohort_sha256="c" * 64,
        input_rows=1,
        conversion_rejections=0,
        tasks={task.id: entry},
    )
    path = tmp_path / "negative.json"
    path.write_text(receipt.model_dump_json())
    loaded = load_oracle_filter(str(path))
    runtime = calibration.attempts[-1].result["config"]["task"]["source_runtime"]["runtime"]
    assert not require_oracle_filter(task, loaded, runtime)
    assert loaded.tasks[task.id].calibration.model_dump_json() == calibration.model_dump_json()
    assert loaded.tasks[task.id].verifier_execution_requirements is None
    assert loaded.tasks[task.id].legacy_role_qualification is None
    with pytest.raises(ValueError, match="Runtime changed"):
        require_oracle_filter(task, loaded, {})
    with pytest.raises(ValueError, match="Task changed"):
        require_oracle_filter(task.model_copy(update={"tags": (*task.tags, "foreign")}), loaded, runtime)


def test_production_collector_preserves_actual_scored_negative_and_raw_result(tmp_path):
    mapping = json.loads(gzip.decompress((FIXTURES / "pym-failure.receipt.json.gz").read_bytes()))["receipt"]
    template = mapping["calibration"]["attempts"][-1]
    with tarfile.open(FIXTURES / "pym-failure.tar.gz") as archive:
        task = TaskSpec.model_validate_json(archive.extractfile("original-packet/task-spec.json").read())
        packet = tmp_path / "native" / task.id
        for member in archive.getmembers():
            if member.isdir() and member.name.startswith("original-packet/"):
                (packet / Path(member.name).relative_to("original-packet")).mkdir(parents=True, exist_ok=True)
            if member.isfile() and member.name.startswith("original-packet/"):
                path = packet / Path(member.name).relative_to("original-packet")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    runtime = template["result"]["config"]["task"]["source_runtime"]["runtime"]
    encoded = task.model_dump_json()
    cohort = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": encoded, "runtime": json.dumps(runtime)}]), cohort)
    task_map = tmp_path / "map.json"
    write_task_map(cohort, packet.parent, task_map)
    jobs = tmp_path / "jobs"
    trial = jobs / "oracle"
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir()
    result = deepcopy(template["result"])
    result.update(task_name=task.id, trial_name="oracle", trial_uri=trial.as_uri(), task_id={"path": str(packet)})
    result["config"]["task"]["path"] = str(packet)
    result["config"].update(trial_name="oracle", trials_dir=str(jobs))
    payload = json.dumps(result).encode()
    (trial / "result.json").write_bytes(payload)
    (trial / "config.json").write_text(json.dumps(result["config"]))
    (trial / "agent/exit-code.txt").write_text(str(template["oracle_exit_code"]))
    (trial / "verifier/verdict.json").write_text(json.dumps(template["verdict"]))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    output = tmp_path / "filtered/tasks.parquet"
    filtered = filter_cohort(
        cohort,
        task_map,
        jobs,
        output,
        negative_control_policy=negative_control_policy().model_copy(update={"decision": reference(policy)}),
    )
    loaded = load_oracle_filter(str(output.parent / "oracle-filter.json"))
    entry = loaded.tasks[task.id]
    assert entry.exclusion_reason is OracleExclusionReason.ORACLE_FAILED
    assert not require_oracle_filter(task, loaded, runtime)
    assert entry.task_sha256 == canonical_sha256(task.model_dump(mode="json"))
    assert entry.calibration.attempts[-1].result == result
    assert entry.calibration.attempts[-1].verifier_execution_requirements is None
    assert (trial / "result.json").read_bytes() == payload
    assert pq.read_table(output).num_rows == 0
    assert pq.read_table(output.parent / "excluded-tasks.parquet").to_pylist()[0]["task_spec"] == encoded
    assert filtered == loaded
