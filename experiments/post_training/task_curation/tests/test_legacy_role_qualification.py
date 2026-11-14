# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual frozen trials retain their raw history through explicit role qualification."""

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
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
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/legacy_roles"


def reference(path):
    return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


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
    decision = tmp_path / "decision.md"
    decision.write_text("Fixture decision scoped to these two immutable completed source passes.\n")
    return LegacyRoleQualificationEvidence(
        input_partition=reference(partition), audit=reference(observations), decision=reference(decision)
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
    with pytest.raises(ValueError, match="requires measured or qualified verifier allocation"):
        require_oracle_filter(task, load_oracle_filter(str(path)), qualification.source_runtime.runtime)
    for field in ("task_sha256", "source_task_sha256", "source_task_checksum", "harbor_task_checksum"):
        foreign = proof.model_dump(mode="json")
        foreign["tasks"][task.id][field] = "0" * 64
        path.write_text(json.dumps(foreign))
        with pytest.raises(ValueError, match="Legacy role qualification differs"):
            load_oracle_filter(str(path))


def test_legacy_evidence_refuses_changed_physical_archive_or_unreviewed_partition(tmp_path, evidence):
    inputs = json.loads(Path(evidence.input_partition.uri).read_text())
    task_id = inputs["entries"][0]["task_id"]
    altered = tmp_path / "changed.tar.gz"
    altered.write_bytes((FIXTURES / "calendar.tar.gz").read_bytes() + b"changed")
    inputs["entries"][0]["trial_archive_uri"] = str(altered)
    partition = tmp_path / "changed-inputs.json"
    partition.write_text(json.dumps(inputs))
    changed = evidence.model_copy(update={"input_partition": reference(partition)})
    with pytest.raises(ValueError, match="exact frozen offline allocation audit"):
        legacy_role_qualifications(changed, frozenset({task_id}))
    observations = Path(evidence.audit.uri)
    audit = json.loads(observations.read_text())
    audit["input_partition_sha256"] = reference(partition).sha256
    audit["rows"][0]["trial_archive_uri"] = str(altered)
    observations.write_text(json.dumps(audit))
    changed = changed.model_copy(update={"audit": reference(observations)})
    with pytest.raises(ValueError, match="archive bytes changed"):
        legacy_role_qualifications(changed, frozenset({task_id}))
