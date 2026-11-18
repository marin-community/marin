# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained production nop archives determine control outcomes without source replay."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.harbor.negative_control import NegativeControlSetupKind
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    NegativeControlApplicability,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
    PostGradeTeardownEvidence,
)

from experiments.post_training.task_curation.negative_controls import (
    negative_control_from_archive,
    require_negative_control,
)
from experiments.post_training.task_curation.post_grade_teardowns import NOP_MACHINE_CLEANUP_AUTHORITY_MESSAGE
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/negative_controls"


def test_actual_nop_packet_runtime_preserves_unset_configuration_and_admits_scored_negative():
    index = json.loads((FIXTURES / "packet-runtime-fallback-index.json").read_text())
    archive = FIXTURES / "packet-runtime-fallback.tar.gz"
    original = archive.read_bytes()
    source = index["source"]
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=index["archive_sha256"]),
        "nop-control/negative-control.json",
        source_task_sha256=source["source_task_sha256"],
        source_task_checksum=source["source_task_checksum"],
        execution_requirements=ExecutionRequirements.model_validate(source["execution_requirements"]),
    )
    loaded = OracleNegativeControlReceipt.model_validate_json(receipt.model_dump_json())
    require_negative_control(loaded)
    assert loaded.conclusive and not loaded.passed(None)
    assert loaded.score == 0
    assert loaded.prepared_source_runtime.runtime_sha256 == source["runtime_sha256"]
    assert loaded.trial.result["config"]["task"]["source_runtime"] is None
    assert archive.read_bytes() == original


@pytest.mark.parametrize("changed_binding", ["packet", "configuration"])
def test_nop_packet_runtime_rejects_another_binding_despite_consistent_archive_hash(tmp_path, changed_binding):
    index = json.loads((FIXTURES / "packet-runtime-fallback-index.json").read_text())
    source = index["source"]
    archive = tmp_path / "changed.tar.gz"
    with tarfile.open(FIXTURES / "packet-runtime-fallback.tar.gz") as original:
        control = json.load(original.extractfile("nop-control/negative-control.json"))
        binding = json.load(original.extractfile("consumed-packet/source-runtime.json"))
        binding["runtime"] = {"foreign": True}
        binding["runtime_sha256"] = canonical_sha256(binding["runtime"])
        updates = {}
        if changed_binding == "packet":
            updates["consumed-packet/source-runtime.json"] = json.dumps(binding).encode()
        else:
            result = control["trial_result"]
            result["config"]["task"]["source_runtime"] = binding
            payload = json.dumps(result).encode()
            name = "nop-control/trials/" + result["trial_name"] + "/result.json"
            updates[name] = payload
            control["trial_result_file_sha256"] = hashlib.sha256(payload).hexdigest()
            updates["nop-control/negative-control.json"] = json.dumps(control).encode()
        with tarfile.open(archive, "w:gz") as output:
            for info in original.getmembers():
                data = updates.get(info.name)
                if data is None and info.isfile():
                    data = original.extractfile(info).read()
                if data is not None:
                    info.size = len(data)
                output.addfile(info, io.BytesIO(data) if data is not None else None)
    with pytest.raises(ValueError, match=r"runtime|source contract"):
        negative_control_from_archive(
            ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
            "nop-control/negative-control.json",
            source_task_sha256=source["source_task_sha256"],
            source_task_checksum=source["source_task_checksum"],
            execution_requirements=ExecutionRequirements.model_validate(source["execution_requirements"]),
        )


def retained_archive(tmp_path, name):
    path = tmp_path / f"{name}.tar.zst"
    path.write_bytes(gzip.decompress((FIXTURES / f"{name}.tar.zst.gz").read_bytes()))
    return path


@pytest.mark.parametrize(
    "verifier_override",
    [
        {"import_path": "harbor.verifier.verifier:Verifier"},
        {"kwargs": {"skip_tests_upload": True}},
        {"env": {"SOURCE_GRADER_MODE": "always-pass"}},
    ],
)
def test_public_nop_receipt_rejects_grading_overrides_with_a_consistent_result_hash(tmp_path, verifier_override):
    index = json.loads((FIXTURES / "index.json").read_text())["negative"]
    archive = retained_archive(tmp_path, "negative")
    original_bytes = archive.read_bytes()
    task = TaskSpec.model_validate_json(gzip.decompress((FIXTURES / "negative.task-spec.json.gz").read_bytes()))
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=task.execution_requirements,
    )
    document = receipt.model_dump(mode="json")
    result = document["trial"]["result"]
    result["config"]["verifier"] = {**result["config"].get("verifier", {}), **verifier_override}
    document["trial"]["result_sha256"] = canonical_sha256(result)
    with pytest.raises(ValueError, match="cannot override its source grading"):
        OracleNegativeControlReceipt.model_validate_json(json.dumps(document))
    assert archive.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "name", ["tau-prepared", "language-mixing-prepared", "tau-missing-lock", "language-mixing-missing-lock"]
)
def test_production_taskspec_nop_preparation_archives_preserve_failed_setup_and_scored_negatives(tmp_path, name):
    index = json.loads((FIXTURES / "converted-preparation-index.json").read_text())[name]
    archive = retained_archive(tmp_path, name)
    original_bytes = archive.read_bytes()
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256=index["source_task_sha256"],
        source_task_checksum=index["source_task_checksum"],
        execution_requirements=ExecutionRequirements.model_validate(index["execution_requirements"]),
    )
    loaded = OracleNegativeControlReceipt.model_validate_json(receipt.model_dump_json())
    require_negative_control(loaded)
    assert loaded.setup_kind is NegativeControlSetupKind.TASK_SPEC_PREPARATION
    assert loaded.preparation_artifact is not None
    assert loaded.setup_exit_code is None
    assert loaded.conclusive is index["conclusive"]
    assert not loaded.passed(None)
    assert loaded.trial.result["agent_info"]["name"] == "nop"
    if index["conclusive"]:
        assert loaded.score == 0
        assert loaded.trial.result["environment_setup"]["finished_at"] is not None
    else:
        assert loaded.score is None
        assert loaded.trial.result["exception_info"]["exception_type"] == "FileNotFoundError"
    assert archive.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("name", "conclusive", "passed", "annotation"),
    [
        ("negative", True, False, None),
        ("sqlfluff", False, False, "negative_control_verifier_error"),
        ("patsy", True, True, None),
    ],
)
def test_actual_nop_archives_retain_native_identity_scores_and_source_errors(
    tmp_path, name, conclusive, passed, annotation
):
    index = json.loads((FIXTURES / "index.json").read_text())[name]
    archive = retained_archive(tmp_path, name)
    before = archive.read_bytes()
    assert hashlib.sha256(before).hexdigest() == index["archive_sha256"]
    task = TaskSpec.model_validate_json(gzip.decompress((FIXTURES / f"{name}.task-spec.json.gz").read_bytes()))
    assert canonical_sha256(task.model_dump(mode="json")) == index["actual_task_sha256"]
    assert task.execution_requirements is not None
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(archive), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=task.execution_requirements,
    )
    reloaded = OracleNegativeControlReceipt.model_validate_json(receipt.model_dump_json())
    require_negative_control(reloaded)
    assert reloaded.consumed_task_sha256 == index["actual_task_sha256"]
    assert reloaded.consumed_task_checksum == index["actual_native_checksum"]
    assert (reloaded.conclusive, reloaded.passed(None), reloaded.annotation) == (conclusive, passed, annotation)
    assert archive.read_bytes() == before
    changed = reloaded.model_copy(update={"verifier_exit_code": 42})
    with pytest.raises(ValueError, match="retained physical evidence"):
        require_negative_control(changed)


@pytest.mark.parametrize(("name", "admitted"), [("negative", True), ("sqlfluff", True), ("patsy", False)])
def test_actual_original_pass_and_physical_nop_round_trip_through_wire4_admission(tmp_path, name, admitted):
    index = json.loads((FIXTURES / "index.json").read_text())[name]
    anchors = json.loads((FIXTURES / "original-index.json").read_text())[name]
    original_bytes = gzip.decompress((FIXTURES / f"{name}.original-pass.json.gz").read_bytes())
    assert hashlib.sha256(original_bytes).hexdigest() == anchors["sha256"]
    original = json.loads(original_bytes)
    task = TaskSpec.model_validate_json(gzip.decompress((FIXTURES / f"{name}.task-spec.json.gz").read_bytes()))
    runtime = json.loads((FIXTURES / f"{name}.runtime.json").read_text())
    assert task.execution_requirements is not None
    control = negative_control_from_archive(
        ContentEvidenceReference(uri=str(retained_archive(tmp_path, name)), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256=anchors["source_task_sha256"],
        source_task_checksum=anchors["source_task_checksum"],
        execution_requirements=task.execution_requirements,
    )
    entry = {
        **original["original_pass_receipt"],
        "source_task_sha256": anchors["source_task_sha256"],
        "source_task_checksum": anchors["source_task_checksum"],
        "negative_control_applicability": NegativeControlApplicability.WORKSPACE,
        "negative_control": control.model_dump(mode="json"),
        "exclusion_reason": None if admitted else OracleExclusionReason.NEGATIVE_CONTROL_PASSED,
    }
    trial_bytes = json.dumps(entry["trials"], sort_keys=True)
    proof = OracleFilterReceipt(
        negative_control_policy=negative_control_policy(),
        recipe_identity=None,
        input_cohort_sha256="f" * 64,
        input_rows=1,
        conversion_rejections=0,
        tasks={task.id: OracleTaskReceipt.model_validate(entry)},
    )
    path = tmp_path / "filter.json"
    path.write_text(proof.model_dump_json())
    reloaded = OracleFilterReceipt.model_validate_json(path.read_text())
    assert reloaded.admits(task, runtime) is admitted
    assert (
        json.dumps(reloaded.tasks[task.id].trials, default=lambda value: value.model_dump(mode="json"), sort_keys=True)
        == trial_bytes
    )
    foreign = reloaded.model_dump(mode="json")
    foreign["tasks"][task.id]["negative_control"]["consumed_task_checksum"] = "foreign"
    with pytest.raises(ValueError, match=r"consumed native package|consumed task contract"):
        OracleFilterReceipt.model_validate(foreign)
    foreign = reloaded.model_dump(mode="json")
    foreign["tasks"][task.id]["negative_control_applicability"] = "answer_graded_not_applicable"
    with pytest.raises(ValueError, match="source or consumed task contract"):
        OracleFilterReceipt.model_validate(foreign)


@pytest.mark.parametrize(
    ("name", "annotation"),
    [
        ("deadline", "negative_control_verifier_timeout"),
        ("sqlfluff_cli_list", "negative_control_verifier_error"),
        ("sqlfluff_plugin_manager", "negative_control_verifier_error"),
        ("sqlfluff_plugin_manager_line80", "negative_control_verifier_error"),
        ("patsy_keyerror", "negative_control_verifier_error"),
        ("native_abort", None),
        ("native_segfault", None),
        ("killed", None),
    ],
)
def test_actual_deadlines_source_errors_and_native_signals_preserve_physical_outcomes(tmp_path, name, annotation):
    index = json.loads((FIXTURES / "exceptions-index.json").read_text())[name]
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(retained_archive(tmp_path, name)), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
        ),
    )
    require_negative_control(receipt)
    assert receipt.annotation == annotation
    assert not receipt.conclusive and not receipt.passed(None)
    if name == "deadline":
        assert receipt.score is None and receipt.trial.verdict is None
    else:
        assert receipt.score == 0
    assert receipt.trial.result["exception_info"] is not None


@pytest.mark.parametrize("index", range(4))
def test_actual_signal_redrive_preserves_both_archives_and_permits_only_one_repeat(tmp_path, index):
    item = json.loads((FIXTURES / "redrives-index.json").read_text())[index]
    requirements = ExecutionRequirements(
        cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
    )
    parent = negative_control_from_archive(
        ContentEvidenceReference(
            uri=str(retained_archive(tmp_path, f"redrive-{index}-parent")), sha256=item["parent_archive"]["sha256"]
        ),
        item["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=requirements,
    )
    assert parent.annotation is None and parent.signal_exit_code == item["signal"]
    repeat = negative_control_from_archive(
        ContentEvidenceReference(
            uri=str(retained_archive(tmp_path, f"redrive-{index}-repeat")), sha256=item["repeat_archive"]["sha256"]
        ),
        item["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=requirements,
        redrive_parent=parent,
        redrive_decision=ContentEvidenceReference(
            uri=str(FIXTURES / "redrive-decision.json"), sha256=item["redrive_decision"]["sha256"]
        ),
    )
    require_negative_control(repeat)
    assert repeat.annotation == "negative_control_verifier_error" and not repeat.passed(None)
    assert repeat.redrive_parent == parent
    document = repeat.model_dump(mode="json")
    document["redrive_parent"] = repeat.model_dump(mode="json")
    with pytest.raises(ValueError, match="Only one approved re-drive"):
        OracleNegativeControlReceipt.model_validate(document)


def test_actual_zero_test_setup_failure_remains_pending_despite_zero_shell_exit(tmp_path):
    index = json.loads((FIXTURES / "setup-failed-index.json").read_text())
    receipt = negative_control_from_archive(
        ContentEvidenceReference(uri=str(retained_archive(tmp_path, "setup-failed")), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256="a" * 64,
        source_task_checksum="b" * 64,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
        ),
    )
    require_negative_control(receipt)
    assert receipt.setup_exit_code == 0
    assert receipt.trial.verdict["detail"]["reason"] == "setup_failed"
    assert receipt.annotation is None and not receipt.conclusive and not receipt.passed(None)


def test_actual_source_interruption_needs_one_identical_repeat_and_unsupported_outcomes_remain_nonpassing(tmp_path):
    index = {item["label"]: item for item in json.loads((FIXTURES / "source-errors-index.json").read_text())}
    shape = ExecutionRequirements(
        cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=900, verifier_timeout_sec=600
    )

    def collect(label, **kwargs):
        item = index[label]
        return negative_control_from_archive(
            ContentEvidenceReference(uri=str(retained_archive(tmp_path, label)), sha256=item["archive"]["sha256"]),
            item["control_member"],
            source_task_sha256=item["source_task_sha256"],
            source_task_checksum=item["source_task_checksum"],
            execution_requirements=shape,
            **kwargs,
        )

    parent = collect("interrupted-parent")
    assert parent.source_producer_interrupted and parent.annotation is None and not parent.conclusive
    repeat = collect(
        "interrupted-repeat",
        redrive_parent=parent,
        redrive_decision=ContentEvidenceReference(
            uri=str(FIXTURES / "interruption-decision.json"),
            sha256="8835dfa4a282ba4cb05bce07722c6e8e880d95d66ce9ffd4ce42b2c8187801b5",
        ),
    )
    require_negative_control(repeat)
    assert repeat.redrive_parent == parent
    assert repeat.annotation == "negative_control_verifier_error" and not repeat.passed(None)
    unsupported = collect("unsupported-outcomes")
    require_negative_control(unsupported)
    assert unsupported.annotation == "negative_control_verifier_error" and not unsupported.passed(None)


@pytest.mark.parametrize("outcome", ["verifier_timeout", "native_scored"])
def test_conclusive_nop_compares_channels_only_to_a_graded_source(tmp_path, outcome):
    index = json.loads((FIXTURES / "index.json").read_text())["negative"]
    anchors = json.loads((FIXTURES / "original-index.json").read_text())["negative"]
    original = json.loads(gzip.decompress((FIXTURES / "negative.original-pass.json.gz").read_bytes()))
    control = negative_control_from_archive(
        ContentEvidenceReference(uri=str(retained_archive(tmp_path, "negative")), sha256=index["archive_sha256"]),
        index["control_member"],
        source_task_sha256=anchors["source_task_sha256"],
        source_task_checksum=anchors["source_task_checksum"],
        execution_requirements=ExecutionRequirements.model_validate(
            original["original_pass_receipt"]["calibration"]["attempts"][-1]["execution_requirements"]
        ),
    )
    entry = original["original_pass_receipt"]
    trial = entry["calibration"]["attempts"][-1]
    trial["verdict"] = None
    result = trial["result"]
    if outcome == "verifier_timeout":
        result["exception_info"] = {
            "exception_type": "VerifierTimeoutError",
            "exception_message": "Source verifier timed out",
        }
        result["verifier_result"] = {"rewards": {}}
        trial["native_grade"] = None
        entry["exclusion_reason"] = OracleExclusionReason.VERIFIER_TIMEOUT
    else:
        result["verifier_result"] = {"rewards": {"reward": 1}}
    trial["result_sha256"] = canonical_sha256(result)
    if outcome == "native_scored":
        trial["native_grade"] = {
            "rewards": {"reward": 1},
            "artifact_path": "verifier/reward.txt",
            "artifact_sha256": "a" * 64,
            "result_sha256": trial["result_sha256"],
            "task_checksum": result["task_checksum"],
        }
    entry.update(
        source_task_sha256=anchors["source_task_sha256"],
        source_task_checksum=anchors["source_task_checksum"],
        negative_control_applicability=NegativeControlApplicability.WORKSPACE,
        negative_control=control.model_dump(mode="json"),
        trials=[trial],
    )
    if outcome == "native_scored":
        with pytest.raises(ValueError, match="different production grading contract"):
            OracleTaskReceipt.model_validate(entry)
        return
    receipt = OracleTaskReceipt.model_validate(entry)
    assert not receipt.passed
    assert not receipt.calibration.attempts[-1].graded
    assert receipt.exclusion_reason is OracleExclusionReason.VERIFIER_TIMEOUT
    assert receipt.negative_control == control
    assert receipt.negative_control.conclusive and receipt.negative_control.score == 0


@pytest.mark.parametrize("approved", [True, False])
def test_recorded_zero_nop_before_teardown_requires_archive_and_decision_proof(tmp_path, approved):
    source = json.loads((FIXTURES / "packet-runtime-fallback-index.json").read_text())["source"]
    root = "nop-control/trials/nop"
    cleanup = "in _stop_environment\nharbor.trial.errors.EnvironmentStopError: Environment teardown failed\n"
    with tarfile.open(FIXTURES / "packet-runtime-fallback.tar.gz") as original:
        members = {item.name: original.extractfile(item).read() for item in original.getmembers() if item.isfile()}
    control = json.loads(members["nop-control/negative-control.json"])
    result = control["trial_result"]
    result.update(
        task_name="scale-swe-verified-nop-fixture",
        trial_uri=f"file:///tmp/{root}",
        verifier_result=None,
        completion=None,
        finished_at="2026-11-14T21:41:19Z",
        exception_info={"exception_type": "EnvironmentStopError", "exception_message": "Environment teardown failed"},
    )
    verdict = control["physical_verdict"]
    verdict["detail"]["exit_code"] = 0
    result_bytes = json.dumps(result).encode()
    control["trial_result_path"] = f"/tmp/{root}/result.json"
    control["trial_result_file_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    members.update(
        {
            f"{root}/result.json": result_bytes,
            f"{root}/config.json": json.dumps(result["config"]).encode(),
            f"{root}/verifier/verdict.json": json.dumps(verdict).encode(),
            f"{root}/verifier/reward.txt": b"0",
            f"{root}/exception.txt": cleanup.encode(),
            "nop-control/negative-control.json": json.dumps(control).encode(),
        }
    )
    archive = tmp_path / "recorded-nop.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    original_bytes = archive.read_bytes()
    reference = ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(original_bytes).hexdigest())
    digest = canonical_sha256(result)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "authority_message": "d01cf03119d70430139c23c1c1227f4f" if approved else "unapproved",
                "cases": {
                    digest: {"task_id": result["task_name"], "reward": 0},
                    "a" * 64: {"task_id": "scale-swe-verified-other", "reward": 0},
                },
            }
        )
    )

    def artifact(name):
        return ContentEvidenceReference(
            uri=f"{reference.uri}#{root}/{name}", sha256=hashlib.sha256(members[f"{root}/{name}"]).hexdigest()
        )

    proof = PostGradeTeardownEvidence(
        kind="nop_verdict_before_environment_stop",
        result_sha256=digest,
        archive=reference,
        result_artifact=artifact("result.json"),
        verdict_artifact=artifact("verifier/verdict.json"),
        reward_artifact=artifact("verifier/reward.txt"),
        teardown_exception_artifact=artifact("exception.txt"),
        decision=ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest()),
    )
    arguments = {
        "source_task_sha256": source["source_task_sha256"],
        "source_task_checksum": source["source_task_checksum"],
        "execution_requirements": ExecutionRequirements.model_validate(source["execution_requirements"]),
    }
    original = negative_control_from_archive(reference, "nop-control/negative-control.json", **arguments)
    assert not original.conclusive
    if not approved:
        with pytest.raises(ValueError, match="approved physical Trial partition"):
            negative_control_from_archive(
                reference, "nop-control/negative-control.json", post_grade_teardown=proof, **arguments
            )
        return
    observed = negative_control_from_archive(
        reference, "nop-control/negative-control.json", post_grade_teardown=proof, **arguments
    )
    require_negative_control(observed)
    assert observed.conclusive and not observed.passed(None)
    assert observed.trial.result == original.trial.result and observed.trial.verdict == original.trial.verdict
    assert observed.trial.result["completion"] is None and observed.trial.result["verifier_result"] is None
    assert observed.trial.post_grade_teardown == proof and observed.score == 0
    assert archive.read_bytes() == original_bytes


@pytest.fixture()
def machine_cleanup_control(tmp_path):
    """A retained nop control whose verdict scored 0.0 before the task machine's cleanup failed."""
    source = json.loads((FIXTURES / "packet-runtime-fallback-index.json").read_text())["source"]
    root = "nop-control/trials/nop"
    cleanup = (
        "Traceback (most recent call last):\n"
        '  File "rolloutengine/machines.py", line 255, in _machine_ownership\n'
        "RuntimeError: Machine cleanup failed: ['machine_close']\n"
    )
    with tarfile.open(FIXTURES / "packet-runtime-fallback.tar.gz") as original:
        members = {item.name: original.extractfile(item).read() for item in original.getmembers() if item.isfile()}
    control = json.loads(members["nop-control/negative-control.json"])
    result = control["trial_result"]
    result.update(
        task_name="tasktrove-nl2bash-machine-cleanup-fixture",
        trial_uri=f"file:///tmp/{root}",
        verifier_result=None,
        completion=None,
        finished_at="2026-11-18T20:19:07Z",
        exception_info={
            "exception_type": "RuntimeServiceError",
            "exception_message": "Required trial service failed",
            "exception_traceback": f"ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)\n{cleanup}",
        },
    )
    return source, root, members, control, result, tmp_path


def _archive_machine_cleanup_control(tmp_path, root, members, control, result, verdict):
    """Write the control archive with the given verdict and return its evidence reference."""
    result_bytes = json.dumps(result).encode()
    control["trial_result_path"] = f"/tmp/{root}/result.json"
    control["trial_result_file_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    control["physical_verdict"] = verdict
    members.update(
        {
            f"{root}/result.json": result_bytes,
            f"{root}/config.json": json.dumps(result["config"]).encode(),
            f"{root}/exception.txt": (
                (
                    result["exception_info"]["exception_traceback"] + "Machine cleanup failed: ['machine_close']\n"
                ).encode()
            ),
            "nop-control/negative-control.json": json.dumps(control).encode(),
        }
    )
    if verdict is not None:
        members[f"{root}/verifier/verdict.json"] = json.dumps(verdict).encode()
    else:
        members.pop(f"{root}/verifier/verdict.json", None)
    members.pop(f"{root}/verifier/reward.txt", None)
    archive = tmp_path / "machine-cleanup-nop.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return ContentEvidenceReference(uri=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


def test_scored_zero_nop_before_machine_cleanup_binds(tmp_path, machine_cleanup_control):
    """A verdict scored before the post-grade machine cleanup failure binds and is conclusive."""
    source, root, members, control, result, _ = machine_cleanup_control
    verdict = {
        "reward": 0.0,
        "status": "scored",
        "detail": {"exit_code": 0, "stderr": "", "channel": "reward.json", "metrics": {"reward": 0.0}},
    }
    reference = _archive_machine_cleanup_control(tmp_path, root, members, control, result, verdict)
    digest = canonical_sha256(result)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "authority_message": NOP_MACHINE_CLEANUP_AUTHORITY_MESSAGE,
                "cases": {digest: {"task_id": result["task_name"], "reward": 0.0}},
            }
        )
    )

    def artifact(name):
        return ContentEvidenceReference(
            uri=f"{reference.uri}#{root}/{name}", sha256=hashlib.sha256(members[f"{root}/{name}"]).hexdigest()
        )

    proof = PostGradeTeardownEvidence(
        kind="nop_verdict_before_environment_stop",
        result_sha256=digest,
        archive=reference,
        result_artifact=artifact("result.json"),
        verdict_artifact=artifact("verifier/verdict.json"),
        teardown_exception_artifact=artifact("exception.txt"),
        decision=ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest()),
    )
    arguments = {
        "source_task_sha256": source["source_task_sha256"],
        "source_task_checksum": source["source_task_checksum"],
        "execution_requirements": ExecutionRequirements.model_validate(source["execution_requirements"]),
    }
    original = negative_control_from_archive(reference, "nop-control/negative-control.json", **arguments)
    assert not original.conclusive
    observed = negative_control_from_archive(
        reference, "nop-control/negative-control.json", post_grade_teardown=proof, **arguments
    )
    require_negative_control(observed)
    assert observed.conclusive and not observed.passed(None)
    assert observed.score == 0
    assert observed.trial.post_grade_teardown == proof
    assert observed.trial.result == original.trial.result and observed.trial.verdict == original.trial.verdict


def test_machine_cleanup_before_scoring_stays_unbindable(tmp_path, machine_cleanup_control):
    """A control whose teardown failed before any scored verdict cannot bind and stays pending."""
    source, root, members, control, result, _ = machine_cleanup_control
    reference = _archive_machine_cleanup_control(tmp_path, root, members, control, result, verdict=None)
    digest = canonical_sha256(result)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "authority_message": NOP_MACHINE_CLEANUP_AUTHORITY_MESSAGE,
                "cases": {digest: {"task_id": result["task_name"], "reward": 0.0}},
            }
        )
    )

    def artifact(name):
        return ContentEvidenceReference(
            uri=f"{reference.uri}#{root}/{name}", sha256=hashlib.sha256(members[f"{root}/{name}"]).hexdigest()
        )

    proof = PostGradeTeardownEvidence(
        kind="nop_verdict_before_environment_stop",
        result_sha256=digest,
        archive=reference,
        result_artifact=artifact("result.json"),
        verdict_artifact=ContentEvidenceReference(uri=f"{reference.uri}#{root}/verifier/verdict.json", sha256="0" * 64),
        teardown_exception_artifact=artifact("exception.txt"),
        decision=ContentEvidenceReference(uri=str(decision), sha256=hashlib.sha256(decision.read_bytes()).hexdigest()),
    )
    arguments = {
        "source_task_sha256": source["source_task_sha256"],
        "source_task_checksum": source["source_task_checksum"],
        "execution_requirements": ExecutionRequirements.model_validate(source["execution_requirements"]),
    }
    original = negative_control_from_archive(reference, "nop-control/negative-control.json", **arguments)
    assert not original.conclusive
    with pytest.raises(ValueError, match="Recorded grade does not precede the reported teardown failure"):
        negative_control_from_archive(
            reference, "nop-control/negative-control.json", post_grade_teardown=proof, **arguments
        )
