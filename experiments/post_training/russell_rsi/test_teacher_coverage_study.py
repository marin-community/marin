# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi import (
    test_incumbent_bank_trial,
    test_teacher_collection,
    test_teacher_diversity_study,
)
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.launch import adopted
from experiments.post_training.russell_rsi.launch_incumbent_bank_trial import incumbent_bank_workflow
from experiments.post_training.russell_rsi.launch_teacher_sft import TeacherCollectionConfig
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_study import collect_chat_rows, qualified_row
from experiments.post_training.russell_rsi.teacher_collection import TeacherTask
from experiments.post_training.russell_rsi.teacher_coverage_study import (
    CANDIDATE_INDICES,
    PERMITTED_SLOTS,
    PROTOCOL,
    CoverageCollectionConfig,
    coverage_candidates,
    coverage_collection,
    coverage_sft_workflow,
    require_coverage_condition,
    run_coverage_collection,
)

continuation_inputs = test_incumbent_bank_trial.continuation_inputs
incumbent_inputs = test_incumbent_bank_trial.incumbent_inputs
student_tokenizer = test_teacher_collection.student_tokenizer


def pin(path, value):
    data = json.dumps(value, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"uri": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def producer_pins(handle, root):
    identity = artifact_identity(handle)
    record = {
        "name": handle.name,
        "version": handle.version,
        "fingerprint": identity.rsplit(":", 1)[1],
        "output_path": str(root),
    }
    producer = pin(root / ".artifact.json", record)
    status = root / ".executor_status"
    status.write_text(STATUS_SUCCESS)
    return {
        "producer": producer,
        "status": {"uri": str(status), "sha256": hashlib.sha256(status.read_bytes()).hexdigest()},
    }


@pytest.fixture
def coverage_bank():
    tasks = [
        {
            "task_id": str(index),
            "capability": "coding",
            "source_id": f"source-{index}",
            "admission_sha256": f"admission-{index}",
            "contract_id": f"family-{index}",
            "relation": "new_contract",
        }
        for index in range(32)
    ]
    records = {
        str(index): json.dumps({"id": str(index), "description": f"Synthetic task {index}"}) for index in range(32)
    }
    for task in tasks:
        task["task_sha256"] = digest(json.loads(records[task["task_id"]]))
    families = {task["task_id"]: task["contract_id"] for task in tasks}
    families["24"] = families["8"]
    candidates = []
    for index in CANDIDATE_INDICES:
        task = tasks[index]
        family = families[task["task_id"]]
        candidates.append(
            {
                **task,
                "stored_index": index,
                "family": family,
                "family_previous_attempt_count": 0,
                "remaining_new_trajectory_allowance": 2,
                "lifetime_trajectory_ceiling": 2,
                "new_trajectory_cap": 2,
                "exclusion_proof_witness": {
                    "clear": True,
                    "excluded_family_hits": [],
                    "excluded_repository_hits": [],
                    "excluded_source_id": False,
                    "family": family,
                    "task_id": task["task_id"],
                    "source_id": task["source_id"],
                    "stored_index": index,
                    "source_repository_witness_available": True,
                },
            }
        )
    return {"tasks": tasks, "family_by_task": families}, records, candidates


def test_general_coverage_preserves_original_labels_and_family_order(coverage_bank):
    bank, records, candidates = coverage_bank
    selected = coverage_candidates(
        bank=bank,
        records=records,
        candidates=candidates,
        history={"records": [], "consumed_trajectories": 0},
        retained=[],
    )
    assert [entry["task_id"] for entry in selected] == [str(index) for index in CANDIDATE_INDICES]
    assert [entry["capability"] for entry in selected] == ["coding"] * 12
    assert len({entry["family"] for entry in selected}) == 12


def test_consumed_alias_cannot_reset_family_budget(coverage_bank):
    bank, records, candidates = coverage_bank
    history = {
        "records": [{"producer": "prior", "slot": "02-0", "family": bank["family_by_task"]["24"], "task_id": "24"}],
        "consumed_trajectories": 1,
    }
    with pytest.raises(ValueError, match="lifetime budget"):
        coverage_candidates(bank=bank, records=records, candidates=candidates, history=history, retained=[])


def test_alias_cannot_replace_fixed_candidate_and_raw_changes_fail(coverage_bank):
    bank, records, candidates = coverage_bank
    changed = deepcopy(candidates)
    changed[5]["stored_index"] = 24
    with pytest.raises(ValueError, match="fixed bank order"):
        coverage_candidates(
            bank=bank,
            records=records,
            candidates=changed,
            history={"records": [], "consumed_trajectories": 0},
            retained=[],
        )
    records["1"] = json.dumps({"id": "1", "description": "Changed admitted content"})
    with pytest.raises(ValueError, match="admission hash"):
        coverage_candidates(
            bank=bank,
            records=records,
            candidates=candidates,
            history={"records": [], "consumed_trajectories": 0},
            retained=[],
        )


@pytest.fixture
def coverage_condition_inputs(incumbent_inputs, tmp_path):
    v21, tasks, _ = incumbent_inputs
    outputs, bound, _ = test_incumbent_bank_trial.sealed_calibration(v21, tasks, tmp_path, [0] * 8)
    qualification = json.loads(Path(v21["qualification_uri"]).read_bytes())
    parent = {
        "name": "coverage-parent",
        "version": "2026.10.06.22",
        "uri": qualification["hf_export_uri"],
        "identity_config": {
            "source_identity": qualification["model_identity"],
            "source_artifact_sha256": v21["incumbent_artifact_sha256"],
            "qualification_sha256": v21["qualification_sha256"],
        },
    }
    from_parent = adopted(parent)
    condition = {
        "kind": "calibration_failure",
        "v21_config": pin(tmp_path / "v21.json", v21),
        "summary": {
            "uri": v21["calibration_summary_uri"],
            "sha256": hashlib.sha256(Path(v21["calibration_summary_uri"]).read_bytes()).hexdigest(),
        },
        "decision": {"uri": v21["calibration_decision_uri"], "sha256": v21["calibration_decision_sha256"]},
        "producers": {
            "calibration": producer_pins(outputs["calibration"], Path(bound.summary_path)),
            "decision": producer_pins(outputs["decision"], tmp_path / "decision"),
        },
    }
    config = {
        "condition": condition,
        "bank": v21["bank"],
        "bank_record_uri": v21["bank_record_uri"],
        "bank_record_sha256": v21["bank_record_sha256"],
        "selection": {
            "bank_sha256": v21["bank"]["identity_config"]["bank_sha256"],
            "train_sha256": v21["bank"]["identity_config"]["train_sha256"],
        },
        "runtime_bundle": v21["runtime_bundle"],
        "parent": parent,
        "initializer": {
            "parent_identity": artifact_identity(from_parent),
            "source_artifact": {"uri": v21["incumbent_artifact_uri"], "sha256": v21["incumbent_artifact_sha256"]},
            "qualification": {"uri": v21["qualification_uri"], "sha256": v21["qualification_sha256"]},
        },
    }
    return config


def test_complete_negative_incumbent_calibration_is_required(coverage_condition_inputs):
    config = coverage_condition_inputs
    condition = config["condition"]
    assert require_coverage_condition(config)["calibration_decision_sha256"] == condition["decision"]["sha256"]
    status = Path(condition["producers"]["decision"]["status"]["uri"])
    status.write_text("failed")
    condition["producers"]["decision"]["status"]["sha256"] = hashlib.sha256(status.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="completed producer"):
        require_coverage_condition(config)


def test_full_chat_collection_keeps_eight_rows_and_consumes_fixed_slots(tmp_path, student_tokenizer):
    tasks = [
        test_teacher_diversity_study.preflight_task(
            index, test_teacher_diversity_study.PREFLIGHT_INSTRUCTION, 71000 + index
        )
        for index in range(20)
    ]
    records = {task.id: task.model_dump_json() for task in tasks}
    teacher_tasks = [
        asdict(TeacherTask(f"family-{index}", "coding", task.id, digest(json.loads(records[task.id]))))
        for index, task in enumerate(tasks)
    ]
    retained = []
    for index, task in enumerate(tasks[:8]):
        row = qualified_row(test_teacher_diversity_study.successful(f"retained-{index}"), task, student_tokenizer)
        retained.append(
            {
                "task": teacher_tasks[index],
                "attempt": 0,
                "slot": "02-0",
                "row": row["example"],
                "row_sha256": compact_json_sha256(row["example"]),
                "witness": row,
            }
        )
    plan = {
        "protocol": PROTOCOL,
        "selection": {
            "selected": teacher_tasks[8:],
            "teacher_model": {"max_tokens": 16384, "temperature": 0, "reasoning_effort": "medium"},
        },
        "retained": retained,
        "permitted_slots": list(PERMITTED_SLOTS),
        "consumed_trajectories": 23,
    }
    directory = StoragePath(str(tmp_path / "coverage"))
    interrupted = directory / "trajectories/02-0/trajectory.json"
    write_once(interrupted, {"task": teacher_tasks[10], "attempt": 0, "slot": "02-0"})
    calls = []

    async def scripted(task, slot, settings):
        calls.append(slot.name)
        if slot.name == "00-0":
            record = test_teacher_diversity_study.successful("failed")
            record["grade"]["reward"] = 0
            return record
        if slot.name == "01-0":
            return test_teacher_diversity_study.successful("retained-0")
        return test_teacher_diversity_study.successful(f"new-{slot.name}")

    result = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, scripted, required_rows=16))
    assert result["status"] == "passed"
    assert result["accepted"][:8] == retained
    assert [entry["task"]["family"] for entry in result["accepted"][8:]] == [f"family-{index}" for index in range(8, 16)]
    assert calls == ["00-0", "00-1", "01-0", "01-1", "02-1", "03-0", "04-0", "05-0", "06-0", "07-0"]
    assert [(entry["slot"], entry["status"]) for entry in result["attempts"][:6]] == [
        ("00-0", "failed"),
        ("00-1", "accepted"),
        ("01-0", "duplicate_student_row"),
        ("01-1", "accepted"),
        ("02-0", "interrupted_consumed"),
        ("02-1", "accepted"),
    ]
    assert result["new_trajectories"] == 11 and result["cumulative_trajectories"] == 34
    assert not (directory / "trajectories/08-0").exists()
    resumed = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, scripted, required_rows=16))
    assert resumed == result and len(calls) == 10


def test_exhausted_twenty_four_slots_preserve_insufficient_rows(tmp_path, student_tokenizer):
    tasks = [
        test_teacher_diversity_study.preflight_task(
            index, test_teacher_diversity_study.PREFLIGHT_INSTRUCTION, 72000 + index
        )
        for index in range(12)
    ]
    records = {task.id: task.model_dump_json() for task in tasks}
    plan = {
        "protocol": PROTOCOL,
        "selection": {
            "selected": [
                asdict(TeacherTask(f"family-{index}", "coding", task.id, digest(json.loads(records[task.id]))))
                for index, task in enumerate(tasks)
            ],
            "teacher_model": {"max_tokens": 16384, "temperature": 0, "reasoning_effort": "medium"},
        },
        "retained": [],
        "permitted_slots": list(PERMITTED_SLOTS),
        "consumed_trajectories": 23,
    }
    calls = []

    async def failed(task, slot, settings):
        calls.append(slot.name)
        record = test_teacher_diversity_study.successful("unqualified")
        record["grade"]["reward"] = 0
        return record

    directory = StoragePath(str(tmp_path / "insufficient"))
    result = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, failed, required_rows=16))
    assert result["status"] == "insufficient_rows" and result["new_trajectories"] == 24
    assert calls == list(PERMITTED_SLOTS)
    assert json.loads((directory / "collection.json").read_text()) == result
    assert (
        asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, failed, required_rows=16)) == result
    )
    assert len(calls) == 24


def test_incomplete_calibration_cannot_trigger_teacher(coverage_condition_inputs, tmp_path):
    config = coverage_condition_inputs
    condition = config["condition"]
    path = Path(condition["summary"]["uri"])
    summary = json.loads(path.read_bytes())
    first = next(iter(summary["task_rewards"]))
    summary["task_rewards"][first].pop()
    condition["summary"] = pin(path, summary)
    decision_path = Path(condition["decision"]["uri"])
    decision = json.loads(decision_path.read_bytes())
    decision["summary_sha256"] = condition["summary"]["sha256"]
    condition["decision"] = pin(decision_path, decision)
    with pytest.raises(test_incumbent_bank_trial.IncompleteCalibrationError):
        require_coverage_condition(config)
    output = tmp_path / "unissued-coverage"
    worker = CoverageCollectionConfig(
        TeacherCollectionConfig(
            config["selection"],
            condition["decision"]["uri"],
            condition["decision"]["sha256"],
            config["bank"]["uri"],
            config["parent"]["uri"],
            config["initializer"]["parent_identity"],
            {},
            RuntimeBundle(**config["runtime_bundle"]),
            "/fixture/relay",
            str(output),
        ),
        config,
    )
    with pytest.raises(test_incumbent_bank_trial.IncompleteCalibrationError):
        run_coverage_collection(worker)
    assert not output.exists()


@pytest.fixture
def coverage_training_inputs(coverage_condition_inputs, tmp_path):
    study = coverage_condition_inputs
    study.update(
        protocol=PROTOCOL,
        version="2026.10.06.22",
        collection_version="2026.10.06.22",
        runtime_commit=MARIN_SKYRL.commit,
        tokenizer_files={"tokenizer.json": "a" * 64},
        relay_job="/fixture/relay",
        student_training_template_uri="fixture-template",
        student_training_template_sha256="b" * 64,
    )
    expected = coverage_collection(study)
    root = tmp_path / "artifacts" / expected.name / expected.version
    producer = producer_pins(expected, root)
    accepted = [
        {"task": {"family": f"family-{index}"}, "row": {"messages": [{"role": "assistant", "content": f"row-{index}"}]}}
        for index in range(16)
    ]
    result = {"protocol": PROTOCOL, "status": "passed", "accepted": accepted}
    train_path = root / "train.jsonl"
    train_path.write_text("".join(json.dumps(row["row"], sort_keys=True) + "\n" for row in accepted))
    train = {"uri": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
    collection = {
        **producer,
        "result": pin(root / "collection.json", result),
        "train": train,
        "dataset": pin(
            root / "dataset.json",
            {
                "sha256": train["sha256"],
                "collection_sha256": compact_json_sha256(result),
                "rows": 16,
                "passes": 2,
                "batch_size": 8,
                "optimizer_updates": 4,
                "example_exposures": 32,
            },
        ),
    }
    config = {
        "version": "2026.10.06.23",
        "runtime_commit": MARIN_SKYRL.commit,
        "protocol": PROTOCOL,
        "study_config": pin(tmp_path / "study.json", study),
        "collection": collection,
        "sft_source_review": pin(tmp_path / "source-review.json", {"source_head": "fixture"}),
    }
    return config, expected


def test_new_training_version_preserves_collection_and_requires_loader(coverage_training_inputs):
    config, expected = coverage_training_inputs
    collection = config["collection"]
    stages = coverage_sft_workflow(config)
    assert artifact_identity(stages["collect"]) == artifact_identity(expected)
    assert stages["train"].version == "2026.10.06.23"
    assert stages["loader"] in graph_handles([stages["train"]])
    assert stages["train"] in graph_handles([stages["reload"]])
    assert not any(
        "skyrl" in handle.name or "calibration" in handle.name for handle in graph_handles([stages["reload"]])
    )
    ctx = StepContext.for_fingerprint(stages["train"].runtime_args, stages["train"].deps)
    train = stages["train"].build_config(ctx)
    assert train.env_vars["WANDB_MODE"] == "disabled"
    assert train.train_config.trainer.load_checkpoint is False
    assert train.train_config.trainer.metrics_start_step == 0
    proof = stages["loader"].build_config(
        StepContext.for_fingerprint(stages["loader"].runtime_args, stages["loader"].deps)
    )
    assert proof.input_pins["collection_identity"] == artifact_identity(expected)
    assert proof.input_pins["jsonl_sha256"] == collection["train"]["sha256"]


def test_completed_trial_nonpromotion_recomputes_original_selection(
    coverage_condition_inputs, incumbent_inputs, tmp_path
):
    config = coverage_condition_inputs
    v21, tasks, _ = incumbent_inputs
    trial_root = tmp_path / "trial"
    outputs, bound, decision = test_incumbent_bank_trial.sealed_calibration(v21, tasks, trial_root, [0, 1] * 4)
    condition = config["condition"]
    condition["kind"] = "trial_nonpromotion"
    condition["v21_config"] = pin(tmp_path / "v21.json", v21)
    condition["decision"] = {"uri": v21["calibration_decision_uri"], "sha256": v21["calibration_decision_sha256"]}
    condition["summary"] = {
        "uri": v21["calibration_summary_uri"],
        "sha256": hashlib.sha256(Path(v21["calibration_summary_uri"]).read_bytes()).hexdigest(),
    }
    condition["producers"]["calibration"] = producer_pins(outputs["calibration"], Path(bound.summary_path))
    condition["producers"]["decision"] = producer_pins(outputs["decision"], trial_root / "decision")
    assert decision["signal_gate_passed"] is True
    training = incumbent_bank_workflow(v21, "train")
    evaluated = incumbent_bank_workflow(v21, "evaluate")
    for role, handle in {**training, **evaluated}.items():
        if role == "terminal":
            continue
        root = tmp_path / "artifacts" / handle.name / handle.version
        condition["producers"][role] = producer_pins(handle, root)
        if role == "rl":
            path = root / ".artifact.json"
            record = json.loads(path.read_bytes())
            record["result"] = {"global_step": 4}
            condition["producers"][role]["producer"] = pin(path, record)
    identity = artifact_identity(training["rl"])
    coding = json.loads(Path(v21["incumbent_coding_uri"]).read_bytes())
    coding.update(model_identity=identity, scores={"humanevalplus": 27 / 32, "mbppplus": 26 / 32})
    retention = json.loads(Path(v21["incumbent_retention_uri"]).read_bytes())
    retention["model_identity"] = identity
    root = tmp_path / "artifacts"
    condition["coding"] = pin(Path(evaluated["coding"].path(str(root))) / "coding-evidence.json", coding)
    condition["retention"] = pin(Path(evaluated["retention"].path(str(root))) / "failure_summary.json", retention)
    selection_handle = evaluated["selection"]
    ctx = StepContext.for_run(selection_handle.path(str(root)), str(root), deps=selection_handle.deps)
    test_incumbent_bank_trial.seal_incumbent_selection(selection_handle.build_config(ctx))
    selection_path = Path(ctx.output_path) / "continuation-selection.json"
    condition["selection"] = {
        "uri": str(selection_path),
        "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    }
    assert require_coverage_condition(config)["calibration_decision_sha256"] == condition["decision"]["sha256"]
    forged = json.loads(selection_path.read_bytes())
    forged["candidate"]["development"] = [1.0, 1.0]
    condition["selection"] = pin(selection_path, forged)
    with pytest.raises(ValueError, match="promoted trial"):
        require_coverage_condition(config)
