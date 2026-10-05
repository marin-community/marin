# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.training.training import LevanterCheckpoint

from experiments.post_training.russell_rsi import test_rsi_continuation
from experiments.post_training.russell_rsi.bootstrap_loop import CheckpointScore, QualifiedTask
from experiments.post_training.russell_rsi.collection_recovery import (
    REASONING_MAPPING_VERSION,
    CollectionRecovery,
    StudentContextAmendment,
)
from experiments.post_training.russell_rsi.feedback import SKILL_DESCRIPTIONS, CodingSkill
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    SelectionConfig,
    StudySelectionConfig,
    adopted,
    qualified_four_update_sft,
    qualified_sft,
    seal_study_calibration,
    seal_study_selection,
)
from experiments.post_training.russell_rsi.launch_rsi_continuation import (
    continuation_workflow,
    seal_continuation_calibration,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_four_pass import (
    PROTOCOL,
    four_pass_post_workflow,
    four_pass_teacher_workflow,
    run_four_pass_collection,
)

continuation_inputs = test_rsi_continuation.continuation_inputs


@pytest.fixture
def study_inputs(continuation_inputs, tmp_path):
    source, tasks, _ = continuation_inputs

    def pin(config, name, value):
        path = tmp_path / f"{name}-{compact_json_sha256(value)}.json"
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        config[f"{name}_uri"], config[f"{name}_sha256"] = str(path), hashlib.sha256(raw).hexdigest()

    calibration = continuation_workflow(source, "calibrate")
    bound = calibration["decision"].build_config(
        StepContext.for_run(
            str(tmp_path / "source-decision"),
            str(tmp_path / "artifacts"),
            deps=calibration["decision"].deps,
        )
    )
    Path(bound.summary_path).mkdir(parents=True)
    (Path(bound.summary_path) / "failure_summary.json").write_text(
        json.dumps(
            {
                "model_identity": bound.plan.current_checkpoint,
                "tasks_identity": bound.plan.bank_identity,
                "count": len(tasks),
                "samples_per_task": 8,
                "task_rewards": {task.task_id: [0, 1] * 4 for task in tasks},
            }
        )
    )
    seal_continuation_calibration(bound)
    source["calibration_decision_uri"] = str(tmp_path / "source-decision/calibration-decision.json")
    source["calibration_decision_sha256"] = hashlib.sha256(
        Path(source["calibration_decision_uri"]).read_bytes()
    ).hexdigest()
    study = {
        **source,
        "protocol": PROTOCOL,
        "coding_panel_sha256": compact_json_sha256(json.loads(Path(source["panel_uri"]).read_text())),
        "candidate_coding_identity": "coding-current",
        "tokenizer_files": {"tokenizer.json": "a" * 64},
        "student_training_template_uri": "/source/template.jinja",
        "student_training_template_sha256": "b" * 64,
        "relay_job": "never-called",
    }
    pin(study, "continuation_config", source)
    pin(
        study,
        "candidate_artifact",
        {
            "name": "candidate",
            "version": "v1",
            "fingerprint": "abc",
            "result_type": "marin.rl.skyrl.SkyRLRun",
            "result": {"global_step": 4},
        },
    )
    qualification = json.loads(Path(source["qualification_uri"]).read_text())
    study["incumbent_qualification_uri"], study["incumbent_qualification_sha256"] = (
        source["qualification_uri"],
        source["qualification_sha256"],
    )
    identity = qualification["model_identity"]
    parent = json.loads(Path(source["parent_coding_uri"]).read_text())["model_identity"]
    baseline = {"checkpoint_identity": identity, "development": [25 / 32, 27 / 32], "retention": 1 / 3}
    candidate = {"checkpoint_identity": "candidate@v1:abc", "development": [25 / 32, 27 / 32], "retention": 0.0}
    decision = {
        "protocol": "champion-rsi-r1",
        "incumbent": baseline,
        "candidate": candidate,
        "selected": baseline,
        "original_parent": {**baseline, "checkpoint_identity": parent, "development": [25 / 32, 25 / 32]},
    }
    pin(study, "continuation_selection", decision)
    pin(
        study,
        "candidate_coding",
        {
            "model_identity": candidate["checkpoint_identity"],
            "panel_sha256": study["coding_panel_sha256"],
            "scores": {"humanevalplus": 25 / 32, "mbppplus": 27 / 32},
        },
    )
    pin(
        study,
        "candidate_retention",
        {
            "model_identity": candidate["checkpoint_identity"],
            "tasks_identity": bound.plan.retention_identity,
            "count": 3,
            "task_rewards": {"a": [0], "b": [0], "c": [0]},
        },
    )
    release = {"skills": [{"label": "types", "description": SKILL_DESCRIPTIONS[CodingSkill.TYPES]}]}
    pin(study, "capability_release", release)
    pin(
        study,
        "capability_review",
        {
            "decision": "approve",
            "binding": {
                "candidate_identity": candidate["checkpoint_identity"],
                "coding_evidence_identity": "coding-current",
                "coding_evidence_sha256": study["candidate_coding_sha256"],
                "coding_panel_sha256": study["coding_panel_sha256"],
                "capability_release_sha256": study["capability_release_sha256"],
                "source": "coding-development",
            },
        },
    )
    study["parent"] = {
        "name": "champion-hf",
        "version": source["version"],
        "uri": qualification["hf_export_uri"],
        "identity_config": {
            "source_identity": identity,
            "source_artifact_sha256": source["incumbent_artifact_sha256"],
            "qualification_sha256": source["qualification_sha256"],
        },
    }
    study["selection"] = {
        "bank_sha256": source["bank_record_sha256"],
        "train_sha256": "f" * 64,
        "capabilities": release,
        "selected": [{"task_id": str(i)} for i in range(12)],
        "teacher_model": {"max_tokens": 512, "temperature": 1, "reasoning_effort": "low"},
    }
    retained = json.loads(Path(source["bank_record_uri"]).read_text())
    additions = [
        QualifiedTask(
            str(index),
            f"hash-{index}",
            f"admission-{index}",
            f"new-source-{index}",
            "api_contracts,types",
            f"new-family-{index}",
            relation="new_contract",
        )
        for index in range(28, 32)
    ]
    expanded = {
        "tasks": [*retained["tasks"], *(asdict(task) for task in additions)],
        "family_by_task": {**retained["family_by_task"], **{task.task_id: task.contract_id for task in additions}},
    }
    pin(study, "bank_record", expanded)
    study["bank"] = {
        **source["bank"],
        "name": "expanded-bank",
        "uri": str(tmp_path / "expanded-bank"),
        "identity_config": {"bank_sha256": study["bank_record_sha256"]},
    }
    study["selection"]["bank_sha256"] = study["bank_record_sha256"]
    pin(
        study,
        "bank_expansion",
        {
            "decision": "approve",
            "binding": {
                "source_bank_sha256": source["bank_record_sha256"],
                "bank_sha256": study["bank_record_sha256"],
                "train_sha256": study["selection"]["train_sha256"],
                "family_map_sha256": compact_json_sha256(expanded["family_by_task"]),
                "capability_release_sha256": study["capability_release_sha256"],
            },
            "additions": [
                {
                    "task": {
                        key: getattr(task, key)
                        for key in ("task_id", "task_sha256", "source_id", "contract_id", "admission_sha256")
                    },
                    "admission_evidence_uri": f"/admission/{task.task_id}",
                    "admission_evidence_sha256": "a" * 64,
                    "exclusion_review_uri": f"/exclusion/{task.task_id}",
                    "exclusion_review_sha256": "b" * 64,
                }
                for task in additions
            ],
        },
    )
    return study, pin


def test_four_pass_gate_binds_new_release_before_collection(study_inputs, tmp_path):
    study, pin = study_inputs
    outputs = four_pass_teacher_workflow(study)
    collected = outputs["collect"]
    bound = collected.build_config(
        StepContext.for_run(
            str(tmp_path / "collection"),
            str(tmp_path / "artifacts"),
            deps=collected.deps,
        )
    )
    review = json.loads(Path(study["capability_review_uri"]).read_text())
    review["binding"]["coding_evidence_identity"] = "old-pilot2"
    pin(study, "capability_review", review)
    with pytest.raises(ValueError, match="new reviewed candidate"):
        run_four_pass_collection(bound)
    assert not (tmp_path / "collection").exists()


@pytest.mark.parametrize(
    "defect, message", [("promoted", "nonpromoted"), ("selected", "nonpromoted"), ("alias", "HF alias")]
)
def test_four_pass_refuses_changed_condition_before_graph_build(study_inputs, defect, message):
    study, pin = study_inputs
    if defect == "alias":
        study["parent"]["uri"] += "/wrong"
    else:
        decision = json.loads(Path(study["continuation_selection_uri"]).read_text())
        if defect == "promoted":
            decision["candidate"]["development"] = [26 / 32, 27 / 32]
            decision["candidate"]["retention"] = 1 / 3
        else:
            decision["selected"]["retention"] = 0.0
        pin(study, "continuation_selection", decision)
    with pytest.raises(ValueError, match=message):
        four_pass_teacher_workflow(study)


def test_four_pass_training_has_four_complete_batches_and_final_reload(study_inputs, tmp_path):
    study, _ = study_inputs
    outputs = four_pass_teacher_workflow(study)
    trained = outputs["train"]
    pod = trained.build_config(
        StepContext.for_run(
            str(tmp_path / "trained"), str(tmp_path / "artifacts"), deps=trained.deps, runtime_args=trained.runtime_args
        )
    )
    config = pod.train_config
    assert config.trainer.num_train_steps == 4
    assert config.trainer.train_batch_size == 8
    assert config.data.mixture_block_size == 8
    assert config.train_seq_len == 4096
    assert outputs["collect"] in graph_handles([trained])
    reload = outputs["reload"].build_config(
        StepContext.for_run(
            str(tmp_path / "reload"),
            str(tmp_path / "artifacts"),
            deps=outputs["reload"].deps,
            runtime_args=outputs["reload"].runtime_args,
        )
    )
    assert reload.model.location.endswith("/hf/step-3")
    assert reload.model.identity == artifact_identity(trained)


def four_update_qualification(identity, root):
    return {
        "protocol": "teacher-sft-four-update-qualification-v1",
        "sft_identity": identity,
        "sft_root": root,
        "hf_export_uri": root + "/hf/step-3",
        "optimizer_updates": 4,
        "learning_rate": 1e-6,
        "loss": 2.0,
        "gradient_norm": 0.5,
        "update_norm": 0.01,
        "optimizer_steps": [
            {"step": i, "learning_rate": 1e-6, "loss": 2.0, "gradient_norm": 0.5, "update_norm": 0.01, "skipped": False}
            for i in range(4)
        ],
        "hf_files": [{"path": "config.json", "sha256": "a" * 64}],
        "hf_shards": [{"path": "model.safetensors", "size": 100}],
        "hf_weight_map": {"weight": "model.safetensors"},
        "hf_verified": dict.fromkeys(("shards", "config", "tokenizer", "eos"), True),
        "serving_reload": {
            "verified": True,
            "model_uri": root + "/hf/step-3",
            "model_identity": identity,
            "suite": "mmlu-smoke",
            "limit": 1,
            "evidence_uri": "reload",
            "evidence_sha256": "b" * 64,
        },
    }


def test_four_update_qualification_cannot_reuse_one_update_or_skipped_step():
    record = four_update_qualification("sft", "/sft")
    assert qualified_four_update_sft(record, identity="sft", root="/sft") == "/sft/hf/step-3"
    with pytest.raises(ValueError, match="1-update"):
        qualified_sft(record, identity="sft", root="/sft")
    one_update = {
        **record,
        "protocol": "teacher-sft-one-update-qualification-v1",
        "optimizer_updates": 1,
        "hf_export_uri": "/sft/hf/step-0",
    }
    with pytest.raises(ValueError, match="4-update"):
        qualified_four_update_sft(one_update, identity="sft", root="/sft")
    record["optimizer_steps"][2]["skipped"] = True
    with pytest.raises(ValueError, match="four complete"):
        qualified_four_update_sft(record, identity="sft", root="/sft")


@pytest.mark.parametrize("signal, additions", [(False, 2), (True, 4)])
def test_four_pass_calibration_is_fresh_and_all_evaluations_wait_for_rl(study_inputs, tmp_path, signal, additions):
    study, pin = study_inputs
    bank = json.loads(Path(study["bank_record_uri"]).read_text())
    bank["tasks"] = bank["tasks"][: 28 + additions]
    bank["family_by_task"] = {task["task_id"]: bank["family_by_task"][task["task_id"]] for task in bank["tasks"]}
    pin(study, "bank_record", bank)
    study["bank"]["identity_config"]["bank_sha256"] = study["bank_record_sha256"]
    study["selection"]["bank_sha256"] = study["bank_record_sha256"]
    proof = json.loads(Path(study["bank_expansion_uri"]).read_text())
    proof["binding"]["bank_sha256"] = study["bank_record_sha256"]
    proof["binding"]["family_map_sha256"] = compact_json_sha256(bank["family_by_task"])
    proof["additions"] = proof["additions"][:additions]
    pin(study, "bank_expansion", proof)
    config = {
        **study,
        "sft": {
            "name": "four-pass-sft",
            "version": study["version"],
            "uri": str(tmp_path / "sft"),
            "identity_config": {},
        },
    }
    pin(config, "sft_config", study)
    sft_identity = artifact_identity(adopted(config["sft"], LevanterCheckpoint))
    record = four_update_qualification(sft_identity, config["sft"]["uri"])
    record["source_config_sha256"] = config["sft_config_sha256"]
    pin(config, "qualification", record)
    outputs = four_pass_post_workflow(config, "calibrate")
    bound = outputs["decision"].build_config(
        StepContext.for_run(str(tmp_path / "decision"), str(tmp_path / "artifacts"), deps=outputs["decision"].deps)
    )
    assert len(bound.record.plan.task_bank) == 28 + additions
    assert bound.record.plan.current_checkpoint != study["parent"]["identity_config"]["source_identity"]
    Path(bound.record.summary_path).mkdir(parents=True)
    (Path(bound.record.summary_path) / "failure_summary.json").write_text(
        json.dumps(
            {
                "model_identity": bound.record.plan.current_checkpoint,
                "tasks_identity": bound.record.plan.bank_identity,
                "count": 28 + additions,
                "samples_per_task": 8,
                "task_rewards": {
                    task.task_id: [0, 1] * 4 if signal else [0] * 8 for task in bound.record.plan.task_bank
                },
            }
        )
    )
    seal_study_calibration(bound)
    raw = (tmp_path / "decision/calibration-decision.json").read_bytes()
    config["calibration_decision_uri"] = str(tmp_path / "decision/calibration-decision.json")
    config["calibration_decision_sha256"] = hashlib.sha256(raw).hexdigest()
    config["calibration_summary_uri"] = str(Path(bound.record.summary_path) / "failure_summary.json")
    evaluated = four_pass_post_workflow(config, "evaluate")
    if signal:
        schedule = json.loads(raw)["schedule"]
        assert schedule["experiment_limits"]["rollouts"] == 256
        assert len(schedule["schedule"]) == 64
        for key in ("coding-sft", "coding-sft-rl", "retention-sft", "retention-sft-rl"):
            assert evaluated["rl"] in graph_handles([evaluated[key]])
            assert evaluated["reload"] in graph_handles([evaluated[key]])
    else:
        assert "rl" not in evaluated
        assert "coding-sft-rl" not in evaluated


@pytest.mark.parametrize(
    "sft_scores,rl_scores,rl_reward,selected,promoted,original_comparison",
    [
        ((26 / 32, 27 / 32), (27 / 32, 28 / 32), 0, "sft", "sft", "sft"),
        ((25 / 32, 26 / 32), (25 / 32, 26 / 32), 1, "sft", "champion", "sft"),
        ((25 / 32, 27 / 32), (26 / 32, 27 / 32), 1, "rl", "rl", "rl"),
    ],
)
def test_study_selection_keeps_champion_and_original_parent_gates_separate(
    tmp_path,
    sft_scores,
    rl_scores,
    rl_reward,
    selected,
    promoted,
    original_comparison,
):
    coding_paths = []
    retention_paths = []
    for identity, scores, reward in [("sft", sft_scores, 1), ("rl", rl_scores, rl_reward)]:
        directory = tmp_path / identity
        directory.mkdir()
        (directory / "coding-evidence.json").write_text(
            json.dumps(
                {
                    "model_identity": identity,
                    "panel_sha256": "panel",
                    "scores": dict(zip(("humanevalplus", "mbppplus"), scores, strict=True)),
                }
            )
        )
        (directory / "failure_summary.json").write_text(
            json.dumps(
                {
                    "model_identity": identity,
                    "tasks_identity": "retention",
                    "count": 3,
                    "task_rewards": {"a": [reward], "b": [0], "c": [0]},
                }
            )
        )
        coding_paths.append(str(directory))
        retention_paths.append(str(directory))
    record = SelectionConfig(
        tuple(coding_paths),
        tuple(retention_paths),
        ("sft", "rl"),
        "panel",
        "retention",
        ("a", "b", "c"),
        CheckpointScore("champion", (25 / 32, 27 / 32), 1 / 3),
        str(tmp_path / "selection"),
    )
    seal_study_selection(StudySelectionConfig(record, PROTOCOL, CheckpointScore("original", (25 / 32, 25 / 32), 1 / 3)))
    result = json.loads((tmp_path / "selection/post-sft-selection.json").read_text())
    assert result["protocol"] == PROTOCOL
    assert result["selected"]["checkpoint_identity"] == selected
    assert result["promoted"]["checkpoint_identity"] == promoted
    assert result["original_parent_comparison"]["checkpoint_identity"] == original_comparison


def test_six_eligible_families_cannot_start_teacher_collection(study_inputs):
    study, _ = study_inputs
    study["selection"]["selected"] = study["selection"]["selected"][:6]
    with pytest.raises(ValueError, match="eight to twelve eligible"):
        four_pass_teacher_workflow(study)


@pytest.mark.parametrize("defect", ["retained", "variant", "source_alias"])
def test_teacher_expansion_preserves_retained_tasks_and_independent_additions(study_inputs, defect):
    study, pin = study_inputs
    bank = json.loads(Path(study["bank_record_uri"]).read_text())
    if defect == "retained":
        bank["tasks"][0]["admission_sha256"] = "changed"
    elif defect == "variant":
        bank["tasks"][-1]["relation"] = "variant"
    else:
        bank["tasks"][-1]["source_id"] = bank["tasks"][0]["source_id"]
    pin(study, "bank_record", bank)
    study["bank"]["identity_config"]["bank_sha256"] = study["bank_record_sha256"]
    study["selection"]["bank_sha256"] = study["bank_record_sha256"]
    with pytest.raises(ValueError, match=r"retain all 28|new independent API"):
        four_pass_teacher_workflow(study)


def test_recovery_wrapper_changes_only_explicit_collection_fingerprint(study_inputs):
    study, _ = study_inputs
    original = four_pass_teacher_workflow(study)["collect"]
    original_payload = original.fingerprint_payload()
    recovery = CollectionRecovery(
        predecessor_uri="/predecessor",
        predecessor_identity="collection@v1:abcd",
        executor_info_sha256="a" * 64,
        executor_status_sha256="a" * 64,
        plan_sha256="a" * 64,
        fatal_marker_sha256="a" * 64,
        slot_reservation_sha256="a" * 64,
        preflight_identity="preflight",
        preflight_reservation_sha256="b" * 64,
        preflight_result_sha256="c" * 64,
        token_proof_uri="/proof.json",
        token_proof_sha256="d" * 64,
        mapping_version=REASONING_MAPPING_VERSION,
        consumed_slot="00-0",
    )
    amendment = StudentContextAmendment(
        "/amendment.json",
        "e" * 64,
        "champion-rsi-teacher-sixteen-k-context-amendment-v1",
        4096,
        16384,
    )
    outputs = four_pass_teacher_workflow(
        {
            **study,
            "collection_recovery": asdict(recovery),
            "student_context_amendment": asdict(amendment),
        }
    )
    amended = outputs["collect"]
    trained = outputs["train"]
    train_config = trained.build_config(
        StepContext.for_fingerprint(trained.runtime_args.keys(), trained.deps)
    ).train_config
    assert train_config.train_seq_len == amendment.context_tokens
    assert train_config.trainer.num_train_steps == 4
    assert train_config.trainer.train_batch_size == 8
    config = amended.build_config(StepContext.for_fingerprint(amended.runtime_args.keys(), amended.deps))
    assert asdict(config)["recovery"] == asdict(recovery)
    assert config.context_amendment.context_tokens == train_config.train_seq_len
    assert amended.fingerprint() != original.fingerprint()
    assert four_pass_teacher_workflow(study)["collect"].fingerprint_payload() == original_payload
    assert set(json.loads(original_payload)) == {"collection", "study"}
    assert canonical_json(config.original.collection) == canonical_json(
        original.build_config(StepContext.for_fingerprint(original.runtime_args.keys(), original.deps)).collection
    )
