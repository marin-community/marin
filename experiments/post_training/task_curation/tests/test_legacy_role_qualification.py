# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual frozen trials retain their raw history through explicit role qualification."""

import gzip
import hashlib
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
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
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/legacy_roles"


def reference(path):
    return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def scoped_decision(path, partition, audit, ruling):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "disposition": "adopted",
                "producer_revision": "2c5eea40f264158596f62e593dd21ecaeaa4dc52",
                "input_partition_sha256": partition.sha256,
                "audit_sha256": audit.sha256,
                "ruling": ruling.model_dump(mode="json"),
            }
        )
    )
    return reference(path)


@pytest.fixture
def evidence(tmp_path):
    inputs = json.loads((FIXTURES / "inputs.json").read_text())
    audit = json.loads((FIXTURES / "audit.json").read_text())
    for entry, row in zip(inputs["entries"], audit["rows"], strict=True):
        name = entry.pop("fixture_name")
        receipt = tmp_path / f"{name}.json"
        receipt.write_bytes(gzip.decompress((FIXTURES / f"{name}.receipt.json.gz").read_bytes()))
        entry["receipt_uri"] = str(receipt)
        entry["trial_archive_uri"] = str(FIXTURES / f"{name}.tar.gz")
        row.update(entry)
    for kind, name in (("producer_manifest", "producer-manifest.json"), ("closure", "closure.json")):
        inputs[f"{kind}_uri"] = str(FIXTURES / name)
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps(inputs))
    audit["input_partition_sha256"] = reference(partition).sha256
    observations = tmp_path / "audit.json"
    observations.write_text(json.dumps(audit))
    ruling = tmp_path / "ruling.md"
    ruling.write_text("Fixture decision scoped to these two immutable completed source passes.\n")
    return LegacyRoleQualificationEvidence(
        input_partition=reference(partition),
        audit=reference(observations),
        decision=scoped_decision(
            tmp_path / "decision.json", reference(partition), reference(observations), reference(ruling)
        ),
    )


@pytest.mark.parametrize(("name", "admitted", "actor_present"), [("calendar", True, False), ("taco", False, True)])
def test_actual_completed_pass_qualification_preserves_raw_history_and_control_gate(
    tmp_path, evidence, name, admitted, actor_present
):
    mapping = json.loads(gzip.decompress((FIXTURES / f"{name}.receipt.json.gz").read_bytes()))["receipt"]
    with tarfile.open(FIXTURES / f"{name}.tar.gz") as archive:
        original = TaskSpec.model_validate_json(archive.extractfile("original-packet/task-spec.json").read())
        native = archive.extractfile("original-packet/task.toml").read()
    qualification = legacy_role_qualifications(evidence, frozenset({original.id}))[original.id]
    calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
    before = calibration.model_dump_json()
    task = calibrated_task_spec(
        original, calibration.execution_requirements, qualification.verifier.execution_requirements, native
    )
    applicability = negative_control_applicability(original)
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
        negative_control_applicability=applicability,
        exclusion_reason=None if admitted else OracleExclusionReason.NEGATIVE_CONTROL_PENDING,
    )
    proof = OracleFilterReceipt(
        negative_control_policy=negative_control_policy(),
        recipe_identity=None,
        input_cohort_sha256=qualification.input_cohort_sha256,
        input_rows=1,
        conversion_rejections=0,
        tasks={task.id: entry},
    )
    path = tmp_path / "oracle-filter.json"
    path.write_text(proof.model_dump_json())
    reloaded = load_oracle_filter(str(path))
    assert require_oracle_filter(task, reloaded, qualification.source_runtime.runtime) is admitted
    qualified = reloaded.tasks[task.id]
    assert qualified.calibration.model_dump_json() == before
    assert qualified.calibration.attempts[-1].verifier_execution_requirements is None
    assert (qualification.actor is not None) is actor_present
    assert (qualification.oracle_helper is not None) is not actor_present
    assert task.verifier_execution_requirements == qualification.verifier.execution_requirements
    unqualified = proof.model_dump(mode="json")
    del unqualified["tasks"][task.id]["legacy_role_qualification"]
    path.write_text(json.dumps(unqualified))
    if actor_present:
        with pytest.raises(ValueError, match="requires measured or qualified verifier allocation"):
            require_oracle_filter(task, load_oracle_filter(str(path)), qualification.source_runtime.runtime)
    else:
        assert require_oracle_filter(task, load_oracle_filter(str(path)), qualification.source_runtime.runtime)
    for field in ("task_sha256", "source_task_sha256", "source_task_checksum", "harbor_task_checksum"):
        foreign = proof.model_dump(mode="json")
        foreign["tasks"][task.id][field] = "0" * 64
        path.write_text(json.dumps(foreign))
        with pytest.raises(ValueError, match="Legacy role qualification differs"):
            load_oracle_filter(str(path))
    foreign_scope = proof.model_dump(mode="json")
    foreign_scope["tasks"][task.id]["legacy_role_qualification"]["adoption_decision"]["audit_sha256"] = "0" * 64
    path.write_text(json.dumps(foreign_scope))
    with pytest.raises(ValueError, match="adoption decision scope"):
        load_oracle_filter(str(path))


@pytest.mark.parametrize(
    "legacy_reason",
    [None, OracleExclusionReason.NEGATIVE_CONTROL_PENDING, OracleExclusionReason.NETWORK_POLICY_UNSUPPORTED],
)
def test_actual_actorless_calendar_admits_original_unprojected_task_without_role_qualification(tmp_path, legacy_reason):
    payload = gzip.decompress((FIXTURES / "calendar.receipt.json.gz").read_bytes())
    mapping = json.loads(payload)["receipt"]
    with tarfile.open(FIXTURES / "calendar.tar.gz") as archive:
        task = TaskSpec.model_validate_json(archive.extractfile("original-packet/task-spec.json").read())
    assert is_nonagentic_task(task)
    assert task.execution_requirements is None and task.verifier_execution_requirements is None
    calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
    original_history = calibration.model_dump_json()
    entry = OracleTaskReceipt(
        task_sha256=canonical_sha256(task.model_dump(mode="json")),
        source_task_sha256=mapping["task_sha256"],
        source_task_checksum=mapping["source_task_checksum"],
        runtime_sha256=mapping["runtime_sha256"],
        harbor_task_checksum=mapping["task_checksum"],
        has_oracle=True,
        trials=(calibration.attempts[-1],),
        calibration=calibration,
        negative_control_applicability=(
            "workspace"
            if legacy_reason is OracleExclusionReason.NEGATIVE_CONTROL_PENDING
            else negative_control_applicability(task)
        ),
        exclusion_reason=legacy_reason,
    )
    proof = OracleFilterReceipt(
        negative_control_policy=negative_control_policy(),
        recipe_identity=None,
        input_cohort_sha256="c" * 64,
        input_rows=1,
        conversion_rejections=0,
        tasks={task.id: entry},
    )
    path = tmp_path / "actorless.json"
    path.write_text(proof.model_dump_json())
    loaded = load_oracle_filter(str(path))
    runtime = calibration.attempts[-1].result["config"]["task"]["source_runtime"]["runtime"]
    assert require_oracle_filter(task, loaded, runtime)
    assert loaded.tasks[task.id].legacy_role_qualification is None
    assert loaded.tasks[task.id].calibration.model_dump_json() == original_history
    assert loaded.tasks[task.id].calibration.attempts[-1].verifier_execution_requirements is None
    with pytest.raises(ValueError, match="Runtime changed"):
        require_oracle_filter(task, loaded, {"foreign": True})


def test_legacy_evidence_refuses_changed_physical_archive_or_unreviewed_partition(tmp_path, evidence):
    inputs = json.loads(Path(evidence.input_partition.uri).read_text())
    task_id = inputs["entries"][0]["task_id"]
    altered = tmp_path / "changed.tar.gz"
    altered.write_bytes((FIXTURES / "calendar.tar.gz").read_bytes() + b"changed")
    inputs["entries"][0]["trial_archive_uri"] = str(altered)
    partition = tmp_path / "changed-inputs.json"
    partition.write_text(json.dumps(inputs))
    changed = evidence.model_copy(update={"input_partition": reference(partition)})
    with pytest.raises(ValueError, match="adoption decision scope"):
        legacy_role_qualifications(changed, frozenset({task_id}))
    observations = Path(evidence.audit.uri)
    audit = json.loads(observations.read_text())
    audit["input_partition_sha256"] = reference(partition).sha256
    audit["rows"][0]["trial_archive_uri"] = str(altered)
    observations.write_text(json.dumps(audit))
    changed = changed.model_copy(update={"audit": reference(observations)})
    decision = json.loads(Path(evidence.decision.uri).read_text())
    changed = changed.model_copy(
        update={
            "decision": scoped_decision(
                tmp_path / "rebound-decision.json",
                changed.input_partition,
                changed.audit,
                ContentEvidenceReference.model_validate(decision["ruling"]),
            )
        }
    )
    with pytest.raises(ValueError, match="archive bytes changed"):
        legacy_role_qualifications(changed, frozenset({task_id}))


def test_approved_actual_calendar_subset_cannot_qualify_extra_taco_task(tmp_path, evidence):
    inputs = json.loads(Path(evidence.input_partition.uri).read_text())
    audit = json.loads(Path(evidence.audit.uri).read_text())
    calendar_id, taco_id = (entry["task_id"] for entry in inputs["entries"])
    subset = tmp_path / "calendar-only.json"
    subset.write_text(json.dumps({**inputs, "entries": inputs["entries"][:1], "tasks": 1}))
    subset_audit = tmp_path / "calendar-audit.json"
    subset_audit.write_text(
        json.dumps({**audit, "input_partition_sha256": reference(subset).sha256, "rows": audit["rows"][:1]})
    )
    decision = json.loads(Path(evidence.decision.uri).read_text())
    approved = evidence.model_copy(
        update={
            "input_partition": reference(subset),
            "audit": reference(subset_audit),
            "decision": scoped_decision(
                tmp_path / "calendar-adoption.json",
                reference(subset),
                reference(subset_audit),
                ContentEvidenceReference.model_validate(decision["ruling"]),
            ),
        }
    )
    assert set(legacy_role_qualifications(approved, frozenset({calendar_id}))) == {calendar_id}
    borrowed = evidence.model_copy(update={"decision": approved.decision})
    with pytest.raises(ValueError, match="adoption decision scope"):
        legacy_role_qualifications(borrowed, frozenset({calendar_id, taco_id}))


@pytest.mark.parametrize("reward", [0, 1])
def test_collector_keeps_original_actorless_contract_and_legacy_verifier_absence(tmp_path, reward):
    mapping = json.loads(gzip.decompress((FIXTURES / "calendar.receipt.json.gz").read_bytes()))["receipt"]
    template = mapping["calibration"]["attempts"][-1]
    with tarfile.open(FIXTURES / "calendar.tar.gz") as archive:
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
    result["verifier_result"]["rewards"] = {"reward": reward}
    payload = json.dumps(result).encode()
    (trial / "result.json").write_bytes(payload)
    (trial / "config.json").write_text(json.dumps(result["config"]))
    (trial / "agent/exit-code.txt").write_text("0")
    (trial / "verifier/verdict.json").write_text(json.dumps({"status": "scored", "reward": reward}))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    global_policy = negative_control_policy().model_copy(update={"decision": reference(policy)})
    output = tmp_path / "filtered/tasks.parquet"
    receipt = filter_cohort(cohort, task_map, jobs, output, negative_control_policy=global_policy)
    entry = receipt.tasks[task.id]
    assert entry.legacy_role_qualification is None
    assert entry.calibration.attempts[-1].verifier_execution_requirements is None
    assert entry.calibration.attempts[-1].result == result
    assert (trial / "result.json").read_bytes() == payload
    assert receipt.admits(task, runtime) is bool(reward)
    retained = pq.read_table(output).to_pylist()
    assert len(retained) == reward
    if reward:
        assert retained[0]["task_spec"] == encoded
        assert TaskSpec.model_validate_json(retained[0]["task_spec"]) == task
