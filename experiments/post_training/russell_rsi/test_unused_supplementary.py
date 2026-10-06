# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from marin.external_dependencies import MARIN_SKYRL
from taskcompendium.models import TaskSpec
from taskcompendium.parquet import read_task_records, write_task_records

from experiments.post_training.russell_rsi.bootstrap_loop import checkpoint_score
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    SelectionConfig,
    StudySelectionConfig,
    seal_study_selection,
)
from experiments.post_training.russell_rsi.test_rsi_continuation import continuation_inputs as continuation_inputs
from experiments.post_training.russell_rsi.test_teacher_four_pass import four_update_qualification
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, preflight_task
from experiments.post_training.russell_rsi.unused_supplementary import (
    PROTOCOL,
    STUDY_PROTOCOL,
    TASK_ORDER,
    PanelAssemblyConfig,
    assemble_unused_panel,
    promoted_study_checkpoint,
)


def pin_json(path, value):
    path.write_text(json.dumps(value))
    return PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def raw_panel(tmp_path):
    protocol = pin_json(tmp_path / "protocol.json", {"protocol": PROTOCOL})
    raw = []
    task_pins = []
    rows: list[dict[str, Any]] = []
    for index, contract in enumerate(TASK_ORDER):
        task = preflight_task(index, PREFLIGHT_INSTRUCTION, 23).model_dump(mode="json")
        task.pop("interaction_tools")
        task.pop("output_paths")
        record = json.dumps(task, indent=index + 1)
        raw.append(record)
        task_pin = pin_json(tmp_path / f"task-{index}.json", task)
        task_pins.append(task_pin)
        rows.append(
            {
                "contract_id": contract,
                "source_commit": f"{index:040x}",
                "qualified": True,
                "sealed_task_json_file_sha256": task_pin.sha256,
                "task_sha256": digest(task),
                "admission_result_sha256": f"{index + 100:064x}",
            }
        )
    parquets = []
    admissions = []
    for cohort, indices in enumerate(((0, 2), (1, 3))):
        path = tmp_path / f"cohort-{cohort}.parquet"
        write_task_records(str(path), [raw[i] for i in indices])
        pin = PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        parquets.append(pin)
        files = {"supplementary-evaluation.parquet": pin.sha256}
        for i in indices:
            rows[i].update(admitted_cohort_parquet_sha256=pin.sha256, admission_artifact_uri=f"cohort-{cohort}")
            files[f"contracts/{TASK_ORDER[i]}/task.json"] = task_pins[i].sha256
            files[f"contracts/{TASK_ORDER[i]}/result.json"] = rows[i]["admission_result_sha256"]
        evidence = pin_json(
            tmp_path / f"evidence-{cohort}.json",
            {
                "files": files,
                "provider_requests": 0,
                "source_split": "dev",
                "repository_holdout": False,
            },
        )
        completion = pin_json(
            tmp_path / f"completion-{cohort}.json",
            {
                "status": "passed",
                "task_count": 2,
                "evidence_manifest_sha256": evidence.sha256,
                "provider_requests": 0,
            },
        )
        admissions.append(
            {
                "artifact_uri": f"cohort-{cohort}",
                "completion": {"path": completion.uri, "sha256": completion.sha256},
                "evidence_manifest": {"path": evidence.uri, "sha256": evidence.sha256},
            }
        )
    handoff = pin_json(
        tmp_path / "handoff.json",
        {
            "protocol": PROTOCOL,
            "comparison_protocol_sha256": protocol.sha256,
            "task_order": TASK_ORDER,
            "task_count": 4,
            "source_split": "dev",
            "repository_holdout": False,
            "tasks": rows,
            "admission_artifacts": admissions,
            "runtime_bundle": {},
        },
    )
    return PanelAssemblyConfig(handoff, protocol, tuple(parquets), tuple(task_pins), tmp_path / "panel"), raw


def test_panel_preserves_raw_rows_despite_schema_default_expansion(raw_panel):
    config, raw = raw_panel
    manifest = assemble_unused_panel(config)
    saved = list(read_task_records(str(config.output_path / "supplementary.parquet")))
    assert saved == raw
    assert (config.output_path / "comparison-protocol.json").read_bytes() == Path(config.protocol.uri).read_bytes()
    assert manifest["task_order"] == list(TASK_ORDER)
    assert manifest["tasks"][0]["task_sha256"] == digest(json.loads(saved[0]))
    assert digest(TaskSpec.model_validate_json(saved[0]).model_dump(mode="json")) != digest(json.loads(saved[0]))


@pytest.mark.parametrize("change", ["record", "order"])
def test_changed_or_reordered_admission_cannot_be_sealed(raw_panel, change):
    config, raw = raw_panel
    if change == "record":
        path = config.parquets[0].uri
        altered = json.loads(raw[0])
        altered["metadata"] = {"changed": "admitted content"}
        write_task_records(path, [json.dumps(altered), raw[2]])
    else:
        handoff = config.handoff.read_json()
        handoff["tasks"].reverse()
        config = replace(config, handoff=pin_json(Path(config.handoff.uri), handoff))
    with pytest.raises(ValueError):
        assemble_unused_panel(config)
    assert not Path(config.output_path).exists()


@pytest.fixture
def promoted_inputs(tmp_path, request):
    old, _, _ = request.getfixturevalue("continuation_inputs")
    source = json.loads(Path(old["incumbent_artifact_uri"]).read_text())
    incumbent_identity = f"{source['name']}@{source['version']}:{source['fingerprint']}"
    root = tmp_path / "candidate"
    root.mkdir()
    identity = "checkpoints/candidate@future:abcd"
    producer = pin_json(
        root / ".artifact.json",
        {
            "name": "checkpoints/candidate",
            "version": "future",
            "fingerprint": "abcd",
            "output_path": str(root),
            "result_type": "marin.training.training.LevanterCheckpoint",
        },
    )
    status = root / ".executor_status"
    status.write_text("SUCCESS")
    candidate_status = PinnedFile(str(status), hashlib.sha256(status.read_bytes()).hexdigest())
    reload_root = tmp_path / "reload"
    reload_root.mkdir()
    reload_dir = reload_root / "records/run"
    reload_dir.mkdir(parents=True)
    training_root = tmp_path / "training"
    training_root.mkdir()
    training_identity = "checkpoints/training@future:1234"
    training_pin = pin_json(
        training_root / ".artifact.json",
        {
            "name": "checkpoints/training",
            "version": "future",
            "fingerprint": "1234",
            "output_path": str(training_root),
            "result_type": "marin.training.training.LevanterCheckpoint",
        },
    )
    training_status = training_root / ".executor_status"
    training_status.write_text("SUCCESS")
    export = str(training_root / "hf/step-3")
    evidence = pin_json(
        reload_dir / "record.json",
        {
            "run_id": "run",
            "status": "succeeded",
            "error": None,
            "metrics": {"mmlu": {"accuracy": 0.0}},
            "model": {"config": {"identity": training_identity}, "location": export},
            "eval": {"name": "mmlu-smoke", "evalchemy": {"max_eval_instances": 1}},
        },
    )
    reload_producer = pin_json(
        reload_root / ".artifact.json",
        {
            "output_path": str(reload_root),
            "config": {"model": {"identity": training_identity, "location": export}, "evals": "mmlu-smoke", "limit": 1},
            "result": {"run_ids": ["run"], "records_prefix": str(reload_root / "records")},
            "deps": ["checkpoints/training@future"],
        },
    )
    reload_status = reload_root / ".executor_status"
    reload_status.write_text("SUCCESS")
    qualification = four_update_qualification(training_identity, str(training_root))
    qualification["serving_reload"].update(evidence_uri=evidence.uri, evidence_sha256=evidence.sha256)
    qualification_pin = pin_json(tmp_path / "candidate-qualification.json", qualification)
    adopted_record = producer.read_json()
    adopted_record.update(
        config={"sft": training_identity, "qualification_sha256": qualification_pin.sha256}, source=export, result=None
    )
    producer = pin_json(root / ".artifact.json", adopted_record)
    incumbent = {"checkpoint_identity": incumbent_identity, "development": [0.7, 0.8], "retention": 1 / 3}
    parent = {"artifact_identity": "historical-parent", "uri": "/historical-parent"}
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    coding_root = tmp_path / "coding"
    retention_root = tmp_path / "retention"
    coding_root.mkdir()
    retention_root.mkdir()
    pin_json(
        coding_root / "coding-evidence.json",
        {
            "model_identity": identity,
            "panel_sha256": "panel",
            "scores": {"humanevalplus": 0.8, "mbppplus": 0.8},
        },
    )
    pin_json(
        retention_root / "failure_summary.json",
        {
            "model_identity": identity,
            "tasks_identity": "retention",
            "count": 3,
            "task_rewards": {"a": [1], "b": [0], "c": [0]},
        },
    )
    seal_study_selection(
        StudySelectionConfig(
            SelectionConfig(
                (str(coding_root),),
                (str(retention_root),),
                (identity,),
                "panel",
                "retention",
                ("a", "b", "c"),
                checkpoint_score(incumbent),
                str(selection_root),
            ),
            STUDY_PROTOCOL,
            checkpoint_score({**incumbent, "checkpoint_identity": parent["artifact_identity"]}),
        )
    )
    selection_file = selection_root / "post-sft-selection.json"
    selection_pin = PinnedFile(str(selection_file), hashlib.sha256(selection_file.read_bytes()).hexdigest())
    selection_producer = pin_json(
        selection_root / ".artifact.json",
        {"output_path": str(selection_root), "name": f"documents/russell-rsi-{STUDY_PROTOCOL}-selection"},
    )
    selection_status = selection_root / ".executor_status"
    selection_status.write_text("SUCCESS")
    historical = pin_json(
        tmp_path / "historical.json",
        {
            "parent": parent,
            "checkpoint_record_uri": old["incumbent_artifact_uri"],
            "checkpoint_record_sha256": old["incumbent_artifact_sha256"],
            "selected_candidate": incumbent_identity,
        },
    )
    historical_launch = pin_json(
        tmp_path / "historical-launch.json",
        {"config_sha256": historical.sha256, "config_readback_verified": True, "selected_checkpoint": incumbent},
    )
    panel_root = tmp_path / "assembled-panel"
    panel_root.mkdir()
    panel_manifest = pin_json(
        panel_root / "panel-manifest.json",
        {
            "protocol": PROTOCOL,
            "task_order": TASK_ORDER,
            "tasks": [{"task_id": key} for key in TASK_ORDER],
            "parquet_sha256": "a" * 64,
            "comparison_protocol_sha256": "c" * 64,
            "runtime_bundle": {},
        },
    )
    return {
        "panel": {
            "uri": str(panel_root),
            "identity_config": {
                "manifest_sha256": panel_manifest.sha256,
                "panel_sha256": "a" * 64,
                "comparison_protocol_sha256": "c" * 64,
            },
        },
        "task_ids": list(TASK_ORDER),
        "runtime_bundle": {},
        "protocol": PROTOCOL,
        "runtime_commit": MARIN_SKYRL.commit,
        "parent": parent,
        "incumbent_producer": {"uri": old["incumbent_artifact_uri"], "sha256": old["incumbent_artifact_sha256"]},
        "incumbent_qualification": {"uri": old["qualification_uri"], "sha256": old["qualification_sha256"]},
        "selection": asdict(selection_pin),
        "selection_producer": asdict(selection_producer),
        "selection_status": {
            "uri": str(selection_status),
            "sha256": hashlib.sha256(selection_status.read_bytes()).hexdigest(),
        },
        "historical_comparison": asdict(historical),
        "historical_launch": asdict(historical_launch),
        "candidate_producer": asdict(producer),
        "candidate_status": asdict(candidate_status),
        "candidate_training_producer": asdict(training_pin),
        "candidate_training_status": {
            "uri": str(training_status),
            "sha256": hashlib.sha256(training_status.read_bytes()).hexdigest(),
        },
        "candidate_qualification": asdict(qualification_pin),
        "candidate_reload_producer": asdict(reload_producer),
        "candidate_reload_status": {
            "uri": str(reload_status),
            "sha256": hashlib.sha256(reload_status.read_bytes()).hexdigest(),
        },
    }


def test_promoted_sft_resolves_its_own_export_and_zero_accuracy_reload(promoted_inputs):
    selected = promoted_study_checkpoint(promoted_inputs)
    producer = PinnedFile(**promoted_inputs["candidate_producer"]).read_json()
    assert selected.export_uri == producer["source"]
    assert selected.source_identity == "checkpoints/candidate@future:abcd"


@pytest.mark.parametrize("change", ["comparator", "nonpromoted", "candidate"])
def test_unpromoted_or_rebound_candidate_cannot_start_unused_comparison(promoted_inputs, change):
    if change == "comparator":
        promoted_inputs["parent"] = {"artifact_identity": "wrong-parent", "uri": "/wrong"}
    elif change == "nonpromoted":
        pin = promoted_inputs["selection"]
        value = PinnedFile(**pin).read_json()
        value["promoted"] = value["incumbent"]
        promoted_inputs["selection"] = asdict(pin_json(Path(pin["uri"]), value))
    else:
        pin = promoted_inputs["candidate_producer"]
        value = PinnedFile(**pin).read_json()
        value["fingerprint"] = "other"
        promoted_inputs["candidate_producer"] = asdict(pin_json(Path(pin["uri"]), value))
    with pytest.raises(ValueError):
        promoted_study_checkpoint(promoted_inputs)


def test_lowered_incumbent_scores_cannot_change_the_fixed_promotion_threshold(promoted_inputs):
    pin = promoted_inputs["selection"]
    decision = PinnedFile(**pin).read_json()
    decision["incumbent"]["development"] = [0.0, 0.0]
    promoted_inputs["selection"] = asdict(pin_json(Path(pin["uri"]), decision))
    with pytest.raises(ValueError):
        promoted_study_checkpoint(promoted_inputs)


def test_another_self_consistent_eight_update_checkpoint_cannot_replace_fixed_incumbent(promoted_inputs, tmp_path):
    source_pin = promoted_inputs["incumbent_producer"]
    source = PinnedFile(**source_pin).read_json()
    old_identity = f"{source['name']}@{source['version']}:{source['fingerprint']}"
    source["fingerprint"] = "different"
    new_identity = f"{source['name']}@{source['version']}:{source['fingerprint']}"
    source_pin = pin_json(tmp_path / "other-incumbent.json", source)
    qualification = PinnedFile(**promoted_inputs["incumbent_qualification"]).read_json()
    qualification["model_identity"] = new_identity
    qualification["serving_reload"]["model_identity"] = new_identity
    promoted_inputs["incumbent_producer"] = asdict(source_pin)
    promoted_inputs["incumbent_qualification"] = asdict(
        pin_json(tmp_path / "other-incumbent-qualification.json", qualification)
    )
    decision_pin = promoted_inputs["selection"]
    decision = PinnedFile(**decision_pin).read_json()
    assert decision["incumbent"]["checkpoint_identity"] == old_identity
    decision["incumbent"]["checkpoint_identity"] = new_identity
    promoted_inputs["selection"] = asdict(pin_json(Path(decision_pin["uri"]), decision))
    with pytest.raises(ValueError, match="historical"):
        promoted_study_checkpoint(promoted_inputs)


@pytest.mark.parametrize("change", ["optimizer", "reload"])
def test_rl_candidate_requires_its_actual_optimizer_gate_and_reload(promoted_inputs, tmp_path, change):
    config = promoted_inputs
    pin = config["candidate_producer"]
    producer = PinnedFile(**pin).read_json()
    identity = f"{producer['name']}@{producer['version']}:{producer['fingerprint']}"
    export = producer["output_path"] + "/exports/global_step_4/policy"
    producer.update(
        result_type="marin.rl.skyrl.SkyRLRun",
        result={
            "global_step": 4,
            "hf_model_uri": export,
            "tokenizer_uri": MODEL,
            "tokenizer_revision": MODEL_REVISION,
        },
    )
    config["candidate_producer"] = asdict(pin_json(Path(pin["uri"]), producer))
    optimizer_root = tmp_path / "optimizer"
    optimizer_root.mkdir()
    optimizer = pin_json(
        optimizer_root / ".artifact.json",
        {
            "name": f"documents/russell-rsi-{STUDY_PROTOCOL}-optimizer-gate",
            "version": "future",
            "output_path": str(optimizer_root),
            "deps": [f"{producer['name']}@{producer['version']}"],
            "config": {"expected_updates": 4, "actual_updates": 4, "export_uri": export},
        },
    )
    status = optimizer_root / ".executor_status"
    status.write_text("SUCCESS")
    config["candidate_optimizer_producer"] = asdict(optimizer)
    config["candidate_optimizer_status"] = {
        "uri": str(status),
        "sha256": hashlib.sha256(status.read_bytes()).hexdigest(),
    }
    qualification = PinnedFile(**config["candidate_qualification"]).read_json()
    reload_evidence = qualification["serving_reload"]
    evidence_pin = PinnedFile(reload_evidence["evidence_uri"], reload_evidence["evidence_sha256"])
    evidence = evidence_pin.read_json()
    evidence["model"]["config"]["identity"] = identity
    evidence["model"]["location"] = export
    evidence_pin = pin_json(Path(evidence_pin.uri), evidence)
    config["candidate_reload_evidence"] = {"evidence_uri": evidence_pin.uri, "evidence_sha256": evidence_pin.sha256}
    reload_pin = config["candidate_reload_producer"]
    reload = PinnedFile(**reload_pin).read_json()
    reload["config"]["model"]["identity"] = identity
    reload["config"]["model"]["location"] = export
    reload["deps"] = [
        f"{producer['name']}@{producer['version']}",
        f"documents/russell-rsi-{STUDY_PROTOCOL}-optimizer-gate@future",
    ]
    config["candidate_reload_producer"] = asdict(pin_json(Path(reload_pin["uri"]), reload))
    assert promoted_study_checkpoint(config).export_uri == export
    if change == "optimizer":
        value = optimizer.read_json()
        value["config"]["actual_updates"] = 3
        config["candidate_optimizer_producer"] = asdict(pin_json(Path(optimizer.uri), value))
    else:
        evidence["status"] = "failed"
        changed = pin_json(Path(evidence_pin.uri), evidence)
        config["candidate_reload_evidence"] = {"evidence_uri": changed.uri, "evidence_sha256": changed.sha256}
    with pytest.raises(ValueError):
        promoted_study_checkpoint(config)
