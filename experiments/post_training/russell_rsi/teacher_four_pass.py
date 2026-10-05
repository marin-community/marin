# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a separate four-pass teacher study to a completed nonpromoted continuation."""

import json
import os
from dataclasses import dataclass, replace

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import (
    QualifiedTask,
    checkpoint_score,
    promotes,
    restored_round,
    write_once,
)
from experiments.post_training.russell_rsi.collection_recovery import CollectionRecovery, StudentContextAmendment
from experiments.post_training.russell_rsi.feedback import SKILL_DESCRIPTIONS, CodingSkill
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    adopted,
    evaluated_score,
    post_sft_stages,
    qualified_four_update_sft,
    validate_source_replay,
)
from experiments.post_training.russell_rsi.launch_rsi_continuation import PROTOCOL as CONTINUATION_PROTOCOL
from experiments.post_training.russell_rsi.launch_rsi_continuation import qualified_champion
from experiments.post_training.russell_rsi.launch_teacher_sft import (
    TEACHER_PIP_PACKAGES,
    CollectionBinding,
    StudentTrainingTemplate,
    TeacherCollectionConfig,
    collect_teacher_dataset,
    teacher_sft_steps,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_collection import (
    STUDENT_ROWS,
    TEACHER_FAMILY_LIMIT,
    selected_teacher_tasks,
)

PROTOCOL = "champion-rsi-teacher-four-pass-prospective-v1"
SOURCE_BANK_TASKS = 28
SOURCE_BANK_FAMILIES = 22
MINIMUM_NEW_FAMILIES = 2
MAXIMUM_NEW_FAMILIES = 4


def pinned_record(config: dict, name: str) -> dict:
    return json.loads(pinned_bytes(config[f"{name}_uri"], config[f"{name}_sha256"]))


def original_retention_task_ids(source: dict, retention_identity: str) -> tuple[str, ...]:
    """Read the original parent task IDs from pinned retention evidence."""
    retained = pinned_record(source, "parent_retention")
    if retained["tasks_identity"] != retention_identity:
        raise ValueError("Original parent retention evidence identifies a different task artifact")
    return tuple(sorted(retained["task_rewards"]))


def require_four_pass_condition(config: dict) -> dict:
    """Reject stale labels and promoted continuations before any teacher request."""
    decision = pinned_record(config, "continuation_selection")
    source = pinned_record(config, "continuation_config")
    if config["protocol"] != PROTOCOL or decision["protocol"] != CONTINUATION_PROTOCOL:
        raise ValueError("Four-pass study requires its separate continuation protocol")
    incumbent = checkpoint_score(decision["incumbent"])
    candidate = checkpoint_score(decision["candidate"])
    if promotes(candidate, incumbent) or checkpoint_score(decision["selected"]) != incumbent:
        raise ValueError("Four-pass study requires a completed nonpromoted continuation")
    artifact = pinned_record(config, "candidate_artifact")
    candidate_identity = f"{artifact['name']}@{artifact['version']}:{artifact['fingerprint']}"
    if artifact["result_type"] != "marin.rl.skyrl.SkyRLRun" or candidate_identity != candidate.checkpoint_identity:
        raise ValueError("Teacher condition candidate differs from its completed artifact")
    if config["coding_panel_sha256"] != compact_json_sha256(pinned_record(source, "panel")):
        raise ValueError("Teacher condition uses a different coding development panel")
    coding = pinned_record(config, "candidate_coding")
    retention = pinned_record(config, "candidate_retention")
    retention_identity = artifact_identity(adopted(source["retention"]))
    if (
        evaluated_score(
            coding,
            retention,
            candidate_identity,
            config["coding_panel_sha256"],
            retention_identity,
            original_retention_task_ids(source, retention_identity),
        )
        != candidate
    ):
        raise ValueError("Teacher condition differs from its candidate development evidence")
    if config["runtime_bundle"] != source["runtime_bundle"]:
        raise ValueError("Teacher runtime differs from the completed continuation")
    release = pinned_record(config, "capability_release")
    review = pinned_record(config, "capability_review")
    binding = {
        "candidate_identity": candidate_identity,
        "coding_evidence_identity": config["candidate_coding_identity"],
        "coding_evidence_sha256": config["candidate_coding_sha256"],
        "coding_panel_sha256": config["coding_panel_sha256"],
        "capability_release_sha256": config["capability_release_sha256"],
        "source": "coding-development",
    }
    if review.get("decision") != "approve" or review.get("binding") != binding:
        raise ValueError("Teacher labels require new reviewed candidate coding provenance")
    if config["selection"]["capabilities"] != release or not release["skills"]:
        raise ValueError("Teacher task selection differs from its canonical release")
    labels = []
    for item in release["skills"]:
        skill = CodingSkill(item["label"])
        if set(item) != {"label", "description"} or item["description"] != SKILL_DESCRIPTIONS[skill]:
            raise ValueError("Teacher release must contain only canonical labels and descriptions")
        labels.append(skill)
    if len(labels) > 4 or len(set(labels)) != len(labels):
        raise ValueError("Teacher release has duplicate or excess canonical labels")
    bank = expanded_study_bank(config, source)
    selected = selected_teacher_tasks(bank, release, tuple(row["task_id"] for row in config["selection"]["selected"]))
    if not STUDENT_ROWS <= len(selected) <= TEACHER_FAMILY_LIMIT or len({task.family for task in selected}) != len(
        selected
    ):
        raise ValueError("Four-pass teacher selection requires eight to twelve eligible independent families")
    original = pinned_record(config, "incumbent_artifact")
    qualification = pinned_record(config, "incumbent_qualification")
    export = qualified_champion(qualification, original)
    expected_alias = {
        "source_identity": incumbent.checkpoint_identity,
        "source_artifact_sha256": config["incumbent_artifact_sha256"],
        "qualification_sha256": config["incumbent_qualification_sha256"],
    }
    if (
        qualification["model_identity"] != incumbent.checkpoint_identity
        or qualification["source_artifact_uri"] != config["incumbent_artifact_uri"]
        or qualification["source_artifact_sha256"] != config["incumbent_artifact_sha256"]
        or config["parent"]["uri"] != export
        or config["parent"]["identity_config"] != expected_alias
    ):
        raise ValueError("Teacher HF alias differs from its original champion qualification")
    return decision


def expanded_study_bank(config: dict, source: dict) -> dict:
    """Keep every source task and require reviewed independent API-family additions."""
    retained = pinned_record(source, "bank_record")
    bank = pinned_record(config, "bank_record")
    proof = pinned_record(config, "bank_expansion")
    original_tasks = tuple(QualifiedTask(**item) for item in retained["tasks"])
    tasks = tuple(QualifiedTask(**item) for item in bank["tasks"])
    additions = tasks[len(original_tasks) :]
    families = bank["family_by_task"]
    if (
        len(original_tasks) != SOURCE_BANK_TASKS
        or len(set(retained["family_by_task"].values())) != SOURCE_BANK_FAMILIES
        or source["bank_record_sha256"] != source["bank"]["identity_config"]["bank_sha256"]
        or tasks[: len(original_tasks)] != original_tasks
        or any(families.get(key) != value for key, value in retained["family_by_task"].items())
        or not MINIMUM_NEW_FAMILIES <= len(additions) <= MAXIMUM_NEW_FAMILIES
        or len(set(families.values())) != SOURCE_BANK_FAMILIES + len(additions)
        or set(families) != {task.task_id for task in tasks}
        or len({task.task_id for task in tasks}) != len(tasks)
        or len({task.source_id for task in additions}) != len(additions)
        or config["bank_record_sha256"] != config["bank"]["identity_config"]["bank_sha256"]
        or config["selection"]["bank_sha256"] != config["bank_record_sha256"]
    ):
        raise ValueError("Four-pass expansion must retain all 28 source tasks and add two to four families")
    for task in additions:
        if (
            task.relation not in ("independent", "new_contract")
            or task.contract_id in set(retained["family_by_task"].values())
            or families[task.task_id] != task.contract_id
            or task.source_id in {item.source_id for item in original_tasks}
            or "api_contracts" not in task.capability.split(",")
        ):
            raise ValueError("Four-pass additions must be new independent API-contract families")
    binding = {
        "source_bank_sha256": source["bank_record_sha256"],
        "bank_sha256": config["bank_record_sha256"],
        "train_sha256": config["selection"]["train_sha256"],
        "family_map_sha256": compact_json_sha256(families),
        "capability_release_sha256": config["capability_release_sha256"],
    }
    if proof.get("decision") != "approve" or proof.get("binding") != binding:
        raise ValueError("Four-pass expanded bank needs its exact reviewed provenance")
    evidence = proof["additions"]
    if len(evidence) != len(additions):
        raise ValueError("Four-pass additions have incomplete admission or exclusion evidence")
    for task, entry in zip(additions, evidence, strict=True):
        expected = {
            key: getattr(task, key) for key in ("task_id", "task_sha256", "source_id", "contract_id", "admission_sha256")
        }
        if (
            entry["task"] != expected
            or not entry["admission_evidence_uri"]
            or len(entry["admission_evidence_sha256"]) != 64
            or not entry["exclusion_review_uri"]
            or len(entry["exclusion_review_sha256"]) != 64
        ):
            raise ValueError("Four-pass additions have incomplete admission or exclusion evidence")
    return bank


@dataclass(frozen=True)
class FourPassCollectionConfig:
    collection: TeacherCollectionConfig
    study: dict


def run_four_pass_collection(config: FourPassCollectionConfig) -> None:
    decision = require_four_pass_condition(config.study)
    write_once(StoragePath(config.collection.output_path) / "four-pass-study.json", config.study)
    collect_teacher_dataset(
        config.collection,
        decision,
        StudentTrainingTemplate(
            config.study["student_training_template_uri"],
            config.study["student_training_template_sha256"],
        ),
        "continuation_selection_sha256",
    )


def run_four_pass_collection_remote(config: FourPassCollectionConfig) -> None:
    remote(
        run_four_pass_collection,
        resources=ResourceConfig.with_cpu(cpu=8, ram="64GB", disk="64GB"),
        pip_packages=list(TEACHER_PIP_PACKAGES),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


@dataclass(frozen=True)
class RecoveryFourPassCollectionConfig:
    original: FourPassCollectionConfig
    recovery: CollectionRecovery
    context_amendment: StudentContextAmendment


def run_recovery_four_pass_collection(config: RecoveryFourPassCollectionConfig) -> None:
    original = config.original
    decision = require_four_pass_condition(original.study)
    write_once(StoragePath(original.collection.output_path) / "four-pass-study.json", original.study)
    collect_teacher_dataset(
        original.collection,
        decision,
        StudentTrainingTemplate(
            original.study["student_training_template_uri"],
            original.study["student_training_template_sha256"],
        ),
        "continuation_selection_sha256",
        config.recovery,
        config.context_amendment,
    )


def run_recovery_four_pass_collection_remote(config: RecoveryFourPassCollectionConfig) -> None:
    remote(
        run_recovery_four_pass_collection,
        resources=ResourceConfig.with_cpu(cpu=8, ram="64GB", disk="64GB"),
        pip_packages=list(TEACHER_PIP_PACKAGES),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def four_pass_teacher_workflow(config: dict) -> dict[str, ArtifactStep]:
    require_four_pass_condition(config)
    collection = {
        **config,
        "dose_decision_uri": config["continuation_selection_uri"],
        "dose_decision_sha256": config["continuation_selection_sha256"],
    }
    if ("collection_recovery" in config) != ("student_context_amendment" in config):
        raise ValueError("Teacher recovery requires its explicit prospective context amendment")
    if "collection_recovery" in config:
        recovery = CollectionRecovery(**config["collection_recovery"])
        amendment = StudentContextAmendment(**config["student_context_amendment"])
        binding = CollectionBinding(
            lambda base: RecoveryFourPassCollectionConfig(FourPassCollectionConfig(base, config), recovery, amendment),
            run_recovery_four_pass_collection_remote,
        )
        return teacher_sft_steps(collection, binding, 4, "teacher-four-pass", amendment.context_tokens)
    return teacher_sft_steps(
        collection,
        CollectionBinding(lambda base: FourPassCollectionConfig(base, config), run_four_pass_collection_remote),
        4,
        "teacher-four-pass",
    )


def four_pass_post_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    """Use the fixed continuation bank after a qualified four-update SFT export."""
    study = pinned_record(config, "sft_config")
    decision = require_four_pass_condition(study)
    source_config = pinned_record(study, "continuation_config")
    for key in ("parent", "bank", "runtime_bundle"):
        if config[key] != study[key]:
            raise ValueError(f"Four-pass post-SFT {key} differs from its SFT inputs")
    for key in ("retention", "machine_config", "panel_uri", "panel_sha256"):
        if config[key] != source_config[key]:
            raise ValueError(f"Four-pass post-SFT {key} differs from the continuation")
    completed = restored_round(pinned_record(source_config, "source_round"))
    source = pinned_record(source_config, "source_replay")
    if source != completed.replay_plan:
        raise ValueError("Four-pass replay differs from its sealed source round")
    validate_source_replay(source, completed.plan)
    bank_record = expanded_study_bank(study, source_config)
    retained_record = pinned_record(source_config, "bank_record")
    tasks = tuple(QualifiedTask(**item) for item in bank_record["tasks"])
    if tasks[: len(completed.plan.task_bank)] != completed.plan.task_bank or any(
        bank_record["family_by_task"][key] != value for key, value in source["family_by_task"].items()
    ):
        raise ValueError("Four-pass bank changed a retained source task or family")
    bank, retention = adopted(config["bank"]), adopted(config["retention"])
    calibrated = pinned_record(source_config, "calibration_decision")
    schedule = calibrated["schedule"]
    if (
        calibrated["protocol"] != CONTINUATION_PROTOCOL
        or calibrated["signal_gate_passed"] is not True
        or calibrated["qualification_sha256"] != source_config["qualification_sha256"]
        or calibrated["plan"]["task_bank"] != retained_record["tasks"]
        or calibrated["plan"]["bank_identity"] != artifact_identity(adopted(source_config["bank"]))
        or schedule["family_by_task"] != retained_record["family_by_task"]
        or len(schedule["sampling_spec"]["targeted_task_ids"]) != 2
    ):
        raise ValueError("Four-pass source calibration differs from the sealed continuation bank")
    source_plan = replace(completed.plan, task_bank=tasks, bank_identity=artifact_identity(bank))
    sft = adopted(config["sft"], LevanterCheckpoint)
    qualification = pinned_record(config, "qualification")
    if qualification["source_config_sha256"] != config["sft_config_sha256"]:
        raise ValueError("Four-pass qualification identifies different SFT configuration")
    export = qualified_four_update_sft(qualification, identity=artifact_identity(sft), root=config["sft"]["uri"])
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        config["version"],
        export,
        kind=LevanterCheckpoint,
        config={"sft": artifact_identity(sft), "qualification_sha256": config["qualification_sha256"]},
    )
    baseline = checkpoint_score(decision["incumbent"])
    original_parent = checkpoint_score(decision["original_parent"])
    retention_task_ids = original_retention_task_ids(source_config, artifact_identity(retention))
    for label, score in (("incumbent", baseline), ("parent", original_parent)):
        coding = pinned_record(source_config, f"{label}_coding")
        retained = pinned_record(source_config, f"{label}_retention")
        if (
            evaluated_score(
                coding,
                retained,
                score.checkpoint_identity,
                study["coding_panel_sha256"],
                artifact_identity(retention),
                retention_task_ids,
            )
            != score
        ):
            raise ValueError("Four-pass baseline differs from its original evidence identity")
    return post_sft_stages(
        config,
        stage,
        model=model,
        bank=bank,
        retention=retention,
        source_plan=source_plan,
        source={**schedule, "family_by_task": bank_record["family_by_task"]},
        export_uri=export,
        study=StudyBaseline(PROTOCOL, baseline, original_parent, retention_task_ids),
    )
