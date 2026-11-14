# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained production nop archives determine control outcomes without source replay."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    NegativeControlApplicability,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
)

from experiments.post_training.task_curation.negative_controls import (
    negative_control_from_archive,
    require_negative_control,
)
from lib.taskcompendium.tests.oracle_filter_fixtures import negative_control_policy

FIXTURES = Path(__file__).parent / "fixtures/negative_controls"


def retained_archive(tmp_path, name):
    path = tmp_path / f"{name}.tar.zst"
    path.write_bytes(gzip.decompress((FIXTURES / f"{name}.tar.zst.gz").read_bytes()))
    return path


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
