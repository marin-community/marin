# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actual frozen trials retain their raw history through explicit role qualification."""

import gzip
import hashlib
import io
import json
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.legacy_producer_1485 import (
    LegacyProducer1485AdoptionDecision,
    LegacyProducer1485QualificationReceipt,
)
from taskcompendium.pipeline.oracle_filter import (
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    load_oracle_filter,
    negative_control_applicability,
    require_oracle_filter,
)

from experiments.post_training.task_curation import archive_evidence, legacy_role_qualification
from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.legacy_producer_1485_qualification import (
    LegacyProducer1485QualificationEvidence,
    legacy_producer_1485_qualifications,
)
from experiments.post_training.task_curation.legacy_role_qualification import (
    LegacyRoleQualificationEvidence,
    legacy_role_qualifications,
)
from experiments.post_training.task_curation.oracle_filter import collect_trials, filter_cohort, write_task_map
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/legacy_roles"
PRODUCER_1485_FIXTURES = Path(__file__).parent / "fixtures/legacy_producer_1485"


@dataclass(frozen=True)
class CollectedLegacyTrial:
    task: TaskSpec
    runtime: dict
    mapping: dict
    trial: OracleTrialReceipt
    archive: Path


def collected_legacy_trial(original_archive: Path, mapping: dict, root: Path) -> CollectedLegacyTrial:
    """Use real task bytes with fresh temporary provider metadata; execute no source commands."""
    template = mapping["calibration"]["attempts"][-1]
    parts = PurePosixPath(template["path"]).parts
    result_member = PurePosixPath("trials", *parts[parts.index("attempt-00") :])
    trial_root = root / "attempt-00/jobs/oracle/attempts/000"
    native_root = root / "native" / mapping["task_name"]
    with tarfile.open(original_archive) as archive:
        for entry in archive.getmembers():
            path = PurePosixPath(entry.name)
            if entry.isdir() and path.is_relative_to("original-packet"):
                (native_root / path.relative_to("original-packet")).mkdir(parents=True, exist_ok=True)
        members = {
            entry.name: (entry, archive.extractfile(entry).read()) for entry in archive.getmembers() if entry.isfile()
        }
    task = TaskSpec.model_validate_json(members["original-packet/task-spec.json"][1])
    binding = json.loads(members["original-packet/source-runtime.json"][1])
    for name, (info, data) in members.items():
        path = PurePosixPath(name)
        target = None
        if path.is_relative_to("original-packet"):
            target = native_root / path.relative_to("original-packet")
        elif path.is_relative_to(result_member.parent):
            target = trial_root / path.relative_to(result_member.parent)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(info.mode)
    result = deepcopy(template["result"])
    result.update(trial_uri=trial_root.as_uri(), task_id={"path": str(native_root)})
    result["config"]["task"]["path"] = str(native_root)
    result["config"]["trials_dir"] = str(root / "attempt-00/jobs")
    payload = json.dumps(result).encode()
    (trial_root / "result.json").write_bytes(payload)
    (trial_root / "config.json").write_text(json.dumps(result["config"]))
    trial = collect_trials(trial_root / "result.json")[(result["task_name"], result["task_checksum"])][0]
    current = deepcopy(mapping)
    current["task_path"] = str(native_root)
    current["calibration"]["attempts"] = [trial.model_dump(mode="json")]
    rebuilt = root / "trial.tar.gz"
    with tarfile.open(rebuilt, "w:gz") as archive:
        for name, (info, data) in members.items():
            if name in {str(result_member), "trials/attempt-00/jobs/oracle/result.json"}:
                data = payload
            elif name == str(result_member.parent / "config.json"):
                data = json.dumps(result["config"]).encode()
            info = deepcopy(info)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return CollectedLegacyTrial(task, binding["runtime"], current, trial, rebuilt)


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


@pytest.fixture
def producer_1485_evidence(tmp_path, monkeypatch):
    files = json.loads((PRODUCER_1485_FIXTURES / "files.json").read_bytes())
    local = {}
    for uri, record in files.items():
        path = PRODUCER_1485_FIXTURES / record["path"]
        if record["gzip"]:
            target = tmp_path / path.stem
            target.write_bytes(gzip.decompress(path.read_bytes()))
            path = target
        local[uri] = path

    def frozen_storage(uri):
        return local[uri] if uri in local else StoragePath(uri)

    monkeypatch.setattr(archive_evidence, "StoragePath", frozen_storage)
    monkeypatch.setattr(legacy_role_qualification, "StoragePath", frozen_storage)
    ruling = tmp_path / "1485-ruling.md"
    ruling.write_text("Fixture adoption of the 18 frozen producer-1485 passes, excluding all 14 unscored parents.\n")
    partition = reference(PRODUCER_1485_FIXTURES / "inputs.json")
    audit = reference(PRODUCER_1485_FIXTURES / "audit.json")
    decision = LegacyProducer1485AdoptionDecision(
        version=1,
        disposition="adopted",
        producer_revision="1485f1b6c64640b5c15a1bffc113d49eefdc60b6",
        input_index_sha256=partition.sha256,
        audit_sha256=audit.sha256,
        adopted_count=18,
        excluded_unscored=14,
        ruling=reference(ruling),
    )
    path = tmp_path / "1485-decision.json"
    path.write_text(decision.model_dump_json())
    return LegacyProducer1485QualificationEvidence(input_partition=partition, audit=audit, decision=reference(path))


def test_actual_producer_1485_partition_qualifies_only_passes_and_preserves_raw_histories(
    tmp_path, producer_1485_evidence
):
    evidence = producer_1485_evidence
    inputs = json.loads((PRODUCER_1485_FIXTURES / "inputs.json").read_bytes())
    qualified = legacy_producer_1485_qualifications(evidence, frozenset(entry["task_id"] for entry in inputs["entries"]))
    assert set(qualified) == {entry["task_id"] for entry in inputs["entries"] if entry["passed"]}
    assert len(qualified) == 18
    for number, entry in enumerate(inputs["entries"]):
        if not entry["passed"]:
            continue
        q = qualified[entry["task_id"]]
        raw = legacy_role_qualification.verified_legacy_evidence_bytes(
            ContentEvidenceReference(uri=entry["receipt_uri"], sha256=entry["receipt_sha256"])
        )
        mapping = json.loads(raw)["receipt"]
        calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
        original_history = calibration.model_dump_json()
        archive = archive_evidence.StoragePath(entry["trial_archive_uri"])
        with tarfile.open(archive) as source:
            task = TaskSpec.model_validate_json(source.extractfile("original-packet/task-spec.json").read())
            native = source.extractfile("original-packet/task.toml").read()
        published = calibrated_task_spec(
            task, calibration.execution_requirements, q.verifier.execution_requirements, native
        )
        receipt = OracleFilterReceipt(
            negative_control_policy=negative_control_policy(),
            recipe_identity=None,
            input_cohort_sha256=entry["cohort_sha256"],
            input_rows=1,
            conversion_rejections=0,
            tasks={
                task.id: OracleTaskReceipt(
                    task_sha256=canonical_sha256(published.model_dump(mode="json")),
                    source_task_sha256=mapping["task_sha256"],
                    source_task_checksum=mapping["source_task_checksum"],
                    runtime_sha256=mapping["runtime_sha256"],
                    harbor_task_checksum=mapping["task_checksum"],
                    has_oracle=True,
                    trials=(calibration.attempts[-1],),
                    calibration=calibration,
                    legacy_role_qualification=q,
                    negative_control_applicability=negative_control_applicability(task),
                    exclusion_reason=OracleExclusionReason.NEGATIVE_CONTROL_PENDING,
                )
            },
        )
        path = tmp_path / f"1485-{number}.json"
        path.write_text(receipt.model_dump_json())
        loaded = load_oracle_filter(str(path))
        actual = loaded.tasks[task.id]
        assert isinstance(actual.legacy_role_qualification, LegacyProducer1485QualificationReceipt)
        assert actual.original_oracle_passed and not actual.passed
        assert not require_oracle_filter(published, loaded, q.source_runtime.runtime)
        assert actual.calibration.model_dump_json() == original_history
        assert actual.calibration.attempts[-1].verifier_execution_requirements is None
        assert "source_runtime" not in actual.calibration.attempts[-1].result["config"]["task"]
        foreign = receipt.model_dump(mode="json")
        foreign["tasks"][task.id]["legacy_role_qualification"][
            "producer_revision"
        ] = "2c5eea40f264158596f62e593dd21ecaeaa4dc52"
        path.write_text(json.dumps(foreign))
        with pytest.raises(ValueError):
            load_oracle_filter(str(path))


@pytest.mark.parametrize("changed", ["unscored_receipt", "unselected_archive"])
def test_producer_1485_checks_unselected_partition_evidence_before_qualification(
    tmp_path, producer_1485_evidence, monkeypatch, changed
):
    entries = json.loads((PRODUCER_1485_FIXTURES / "inputs.json").read_bytes())["entries"]
    selected = next(entry for entry in entries if entry["passed"])
    if changed == "unscored_receipt":
        unscored = next(entry for entry in entries if not entry["scored"])
        path = legacy_role_qualification.StoragePath(unscored["receipt_uri"])
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        other = next(entry for entry in reversed(entries) if entry["passed"])
        source = archive_evidence.StoragePath(other["trial_archive_uri"])
        altered = tmp_path / "altered.tar.gz"
        altered.write_bytes(source.read_bytes() + b"changed")
        original_storage = archive_evidence.StoragePath
        monkeypatch.setattr(
            archive_evidence,
            "StoragePath",
            lambda uri: altered if uri == other["trial_archive_uri"] else original_storage(uri),
        )
    with pytest.raises(ValueError, match="bytes changed"):
        legacy_producer_1485_qualifications(producer_1485_evidence, frozenset({selected["task_id"]}))


@pytest.mark.parametrize("old_name", ["taco", "calendar"])
def test_collector_keeps_both_legacy_producers_and_pending_controls_in_one_cohort(
    tmp_path, evidence, producer_1485_evidence, old_name
):
    old_inputs = json.loads(Path(evidence.input_partition.uri).read_bytes())
    old_audit = json.loads(Path(evidence.audit.uri).read_bytes())
    old_entry = next(
        entry for entry in old_inputs["entries"] if Path(entry["trial_archive_uri"]).name == f"{old_name}.tar.gz"
    )
    old_mapping = json.loads(Path(old_entry["receipt_uri"]).read_bytes())["receipt"]
    older_inputs = json.loads(Path(producer_1485_evidence.input_partition.uri).read_bytes())
    older_audit = json.loads(Path(producer_1485_evidence.audit.uri).read_bytes())
    older_entry = next(entry for entry in older_inputs["entries"] if entry["passed"])
    older_mapping = json.loads(legacy_role_qualification.StoragePath(older_entry["receipt_uri"]).read_bytes())["receipt"]
    old = collected_legacy_trial(Path(old_entry["trial_archive_uri"]), old_mapping, tmp_path / "producer-2c5")
    older = collected_legacy_trial(
        archive_evidence.StoragePath(older_entry["trial_archive_uri"]), older_mapping, tmp_path / "producer-1485"
    )
    cohort = tmp_path / "mixed.parquet"
    rows = [{"task_spec": item.task.model_dump_json(), "runtime": json.dumps(item.runtime)} for item in (old, older)]
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    cohort_digest = reference(cohort).sha256

    def scoped_inputs(inputs, audit, entry, item, name):
        receipt_path = tmp_path / f"{name}-receipt.json"
        receipt_path.write_text(json.dumps({"receipt": item.mapping}))
        entry.update(
            receipt_uri=str(receipt_path),
            receipt_sha256=reference(receipt_path).sha256,
            trial_archive_uri=str(item.archive),
            trial_archive_sha256=reference(item.archive).sha256,
            cohort_sha256=cohort_digest,
        )
        if "parent_result_sha256" in entry:
            entry["parent_result_sha256"] = item.trial.result_sha256
        row = next(row for row in audit["rows"] if row["task_id"] == entry["task_id"])
        row.update(entry)
        row["observation"].update(
            archive_sha256=entry["trial_archive_sha256"], raw_result_sha256=item.trial.result_sha256
        )
        partition = tmp_path / f"{name}-inputs.json"
        partition.write_text(json.dumps(inputs))
        audit["input_partition_sha256"] = reference(partition).sha256
        audit_path = tmp_path / f"{name}-audit.json"
        audit_path.write_text(json.dumps(audit))
        return reference(partition), reference(audit_path)

    old_partition, old_observations = scoped_inputs(old_inputs, old_audit, old_entry, old, "2c5")
    old_decision = json.loads(Path(evidence.decision.uri).read_bytes())
    old_evidence = evidence.model_copy(
        update={
            "input_partition": old_partition,
            "audit": old_observations,
            "decision": scoped_decision(
                tmp_path / "2c5-decision.json",
                old_partition,
                old_observations,
                ContentEvidenceReference.model_validate(old_decision["ruling"]),
            ),
        }
    )
    older_partition, older_observations = scoped_inputs(older_inputs, older_audit, older_entry, older, "1485")
    older_decision = LegacyProducer1485AdoptionDecision.model_validate_json(
        Path(producer_1485_evidence.decision.uri).read_bytes()
    )
    updated = older_decision.model_copy(
        update={"input_index_sha256": older_partition.sha256, "audit_sha256": older_observations.sha256}
    )
    decision_path = tmp_path / "1485-mixed-decision.json"
    decision_path.write_text(updated.model_dump_json())
    older_evidence = producer_1485_evidence.model_copy(
        update={
            "input_partition": older_partition,
            "audit": older_observations,
            "decision": reference(decision_path),
        }
    )
    mappings = tmp_path / "mixed-map.json"
    mappings.write_text(json.dumps({item.task.id: item.mapping for item in (old, older)}))
    policy_path = tmp_path / "mixed-policy.json"
    policy_path.write_text("{}")
    policy = negative_control_policy().model_copy(update={"decision": reference(policy_path)})
    originals = {Path(item.trial.path): Path(item.trial.path).read_bytes() for item in (old, older)}
    output = tmp_path / "mixed/tasks.parquet"
    receipt = filter_cohort(
        cohort,
        mappings,
        Path(old.trial.path),
        output,
        negative_control_policy=policy,
        legacy_role_qualification_evidence=old_evidence,
        legacy_producer_1485_qualification_evidence=older_evidence,
        additional_trial_inputs=(Path(older.trial.path),),
    )
    assert {entry.legacy_role_qualification.producer_revision for entry in receipt.tasks.values()} == {
        "2c5eea40f264158596f62e593dd21ecaeaa4dc52",
        "1485f1b6c64640b5c15a1bffc113d49eefdc60b6",
    }
    assert all(entry.original_oracle_passed for entry in receipt.tasks.values())
    assert receipt.tasks[old.task.id].passed is (old_name == "calendar")
    assert receipt.tasks[older.task.id].exclusion_reason is OracleExclusionReason.NEGATIVE_CONTROL_PENDING
    if old_name == "taco":
        assert receipt.tasks[old.task.id].exclusion_reason is OracleExclusionReason.NEGATIVE_CONTROL_PENDING
    else:
        published = TaskSpec.model_validate_json(pq.read_table(output).to_pylist()[0]["task_spec"])
        assert canonical_sha256(published.model_dump(mode="json")) == (
            receipt.tasks[old.task.id].legacy_role_qualification.published_task_sha256
        )
        assert receipt.admits(published, old.runtime)
    assert all(
        entry.calibration.attempts[-1].verifier_execution_requirements is None for entry in receipt.tasks.values()
    )
    assert {path: path.read_bytes() for path in originals} == originals
    assert pq.read_table(output).num_rows == int(old_name == "calendar")
    assert load_oracle_filter(str(output.parent / "oracle-filter.json")) == receipt


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
