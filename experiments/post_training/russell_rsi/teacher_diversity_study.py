# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind eight teacher families to new coding evidence and retained qualified rows."""

import json
import os
from dataclasses import asdict, dataclass, replace

from fray.types import ResourceConfig
from levanter.tokenizers import MarinTokenizer
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.russell_rsi.bootstrap_loop import (
    QualifiedTask,
    checkpoint_score,
    promotes,
    restored_round,
    write_once,
)
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.feedback import SKILL_DESCRIPTIONS, CodingSkill
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    adopted,
    evaluated_score,
    post_sft_stages,
    qualified_four_update_sft,
    validate_source_replay,
)
from experiments.post_training.russell_rsi.launch_teacher_sft import (
    SFT_LEARNING_RATE,
    TEACHER_PIP_PACKAGES,
    CollectionBinding,
    StudentTrainingTemplate,
    TeacherCollectionConfig,
    teacher_sft_steps,
)
from experiments.post_training.russell_rsi.rollout_eval import (
    DevelopmentEvaluationConfig,
    calibration_evaluation_journal,
    run_development_evaluation,
)
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_study import PROTOCOL as RETAINED_PROTOCOL
from experiments.post_training.russell_rsi.teacher_chat_study import collect_chat_dataset, qualified_row
from experiments.post_training.russell_rsi.teacher_collection import selected_teacher_tasks
from experiments.post_training.russell_rsi.teacher_four_pass import (
    original_retention_task_ids,
    pinned_record,
    require_four_pass_condition,
)

PROTOCOL = "champion-rsi-teacher-eight-family-diversity-v1"
NAMESPACE = "teacher-eight-family-diversity"
RETAINED_ROWS = 4
RETAINED_SLOTS = ("00-1", "02-0", "05-1", "08-0")
NEW_FAMILIES = 6
ROWS = 8
PASSES = 4
UPDATES = 4
CONTEXT_TOKENS = 16384
MINIMUM_BANK_TASKS = 32
MAXIMUM_BANK_TASKS = 36
PERMITTED_SLOTS = tuple(f"{family:02}-{attempt}" for family in range(NEW_FAMILIES) for attempt in range(2))


@dataclass(frozen=True)
class DiversityCollectionConfig:
    collection: TeacherCollectionConfig
    study: dict


def diversity_bank(config: dict, original: dict) -> dict:
    """Keep the admitted source bank and bind each prospective addition to review."""
    retained = pinned_record(original, "bank_record")
    bank = pinned_record(config, "bank_record")
    tasks = tuple(QualifiedTask(**item) for item in bank["tasks"])
    previous = tuple(QualifiedTask(**item) for item in retained["tasks"])
    families = bank["family_by_task"]
    if (
        len(previous) != MINIMUM_BANK_TASKS
        or not MINIMUM_BANK_TASKS <= len(tasks) <= MAXIMUM_BANK_TASKS
        or tasks[: len(previous)] != previous
        or len({task.task_id for task in tasks}) != len(tasks)
        or set(families) != {task.task_id for task in tasks}
        or any(families.get(key) != value for key, value in retained["family_by_task"].items())
        or config["bank"]["identity_config"]["bank_sha256"] != config["bank_record_sha256"]
        or config["selection"]["bank_sha256"] != config["bank_record_sha256"]
    ):
        raise ValueError("Diversity bank must preserve all 32 admitted tasks and families")
    additions = tasks[len(previous) :]
    for task in additions:
        if (
            task.relation not in ("independent", "new_contract")
            or task.source_id in {item.source_id for item in previous}
            or task.contract_id in set(retained["family_by_task"].values())
            or families[task.task_id] != task.contract_id
            or "api_contracts" not in task.capability.split(",")
        ):
            raise ValueError("New diversity tasks require independent admitted API families")
    if len({task.source_id for task in additions}) != len(additions) or len(
        {families[task.task_id] for task in additions}
    ) != len(additions):
        raise ValueError("New diversity tasks share a source or family")
    review = pinned_record(config, "bank_expansion")
    binding = {
        "source_bank_sha256": original["bank_record_sha256"],
        "bank_sha256": config["bank_record_sha256"],
        "train_sha256": config["selection"]["train_sha256"],
        "family_map_sha256": compact_json_sha256(families),
        "capability_release_sha256": config["capability_release_sha256"],
    }
    if review["decision"] != "approve" or review["binding"] != binding or len(review["additions"]) != len(additions):
        raise ValueError("Diversity bank needs its exact admission and exclusion review")
    for task, evidence in zip(additions, review["additions"], strict=True):
        expected = {
            key: getattr(task, key) for key in ("task_id", "task_sha256", "source_id", "contract_id", "admission_sha256")
        }
        if evidence["task"] != expected:
            raise ValueError("Diversity addition differs from reviewed admission")
        for name in ("admission_evidence", "exclusion_review"):
            PinnedFile(evidence[f"{name}_uri"], evidence[f"{name}_sha256"]).read_json()
    return bank


def require_diversity_condition(config: dict) -> tuple[dict, dict, dict]:
    """Audit old lineage separately from the new coding and family-selection decision."""
    if config["protocol"] != PROTOCOL:
        raise ValueError("Wrong teacher diversity protocol")
    original = PinnedFile(**config["original_study"]).read_json()
    lineage = require_four_pass_condition(original)
    current = pinned_record(config, "current_selection")
    coding = pinned_record(config, "current_coding")
    incumbent = checkpoint_score(lineage["incumbent"])
    candidate = checkpoint_score(current["sft"])
    if (
        promotes(candidate, incumbent)
        or checkpoint_score(current["selected"]) != candidate
        or checkpoint_score(current["incumbent"]) != incumbent
        or checkpoint_score(current["promoted"]) != incumbent
        or current["sft_rl"] is not None
        or current["original_parent"] != lineage["original_parent"]
        or coding["model_identity"] != candidate.checkpoint_identity
        or coding["panel_sha256"] != original["coding_panel_sha256"]
        or tuple(coding["scores"][key] for key in ("humanevalplus", "mbppplus")) != candidate.development
    ):
        raise ValueError("Diversity study requires completed current coding and retained update8")
    source_config = pinned_record(original, "continuation_config")
    panel = pinned_record(source_config, "panel")
    expected = {(row["suite"], row["benchmark_id"]): row["prompt_sha256"] for row in panel["items"]}
    seen = set()
    for row in coding["rows"]:
        key = (row["suite"], row["benchmark_id"])
        if key in seen or key not in expected or row["prompt_sha256"] != expected[key] or row["pass_rate"] not in (0, 1):
            raise ValueError("Current coding evidence has incomplete or invalid development outcomes")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Current coding evidence changed the frozen development panel")
    for suite in ("humanevalplus", "mbppplus"):
        rows = [row for row in coding["rows"] if row["suite"] == suite]
        if len(rows) != 32 or sum(row["pass_rate"] for row in rows) / 32 != coding["scores"][suite]:
            raise ValueError("Current coding scores differ from all 32 saved outcomes")
    if len(coding["records_sha256"]) != 2 or any(len(value) != 64 for value in coding["records_sha256"]):
        raise ValueError("Current coding evidence lacks its completed source records")
    release = pinned_record(config, "capability_release")
    labels = []
    for item in release["skills"]:
        skill = CodingSkill(item["label"])
        if item != {"label": skill.value, "description": SKILL_DESCRIPTIONS[skill]}:
            raise ValueError("Diversity selection needs canonical coding labels")
        labels.append(skill)
    review = pinned_record(config, "capability_review")
    binding = {
        "candidate_identity": candidate.checkpoint_identity,
        "coding_evidence_identity": config["current_coding_identity"],
        "coding_evidence_sha256": config["current_coding_sha256"],
        "coding_panel_sha256": original["coding_panel_sha256"],
        "capability_release_sha256": config["capability_release_sha256"],
        "source": "coding-development",
    }
    if (
        not 1 <= len(labels) <= 4
        or len(set(labels)) != len(labels)
        or review["decision"] != "approve"
        or review["binding"] != binding
        or config["selection"]["capabilities"] != release
    ):
        raise ValueError("Diversity labels require new reviewed current coding provenance")
    for key in (
        "parent",
        "runtime_bundle",
        "runtime_commit",
        "tokenizer_files",
        "student_context_amendment",
        "student_training_template_uri",
        "student_training_template_sha256",
    ):
        if config[key] != original[key]:
            raise ValueError(f"Diversity study changed its frozen {key}")
    if (
        config["runtime_commit"] != MARIN_SKYRL.commit
        or config["selection"]["teacher_model"] != original["selection"]["teacher_model"]
    ):
        raise ValueError("Diversity collection changed teacher settings or runtime")
    bank = diversity_bank(config, original)
    selected = selected_teacher_tasks(bank, release, tuple(row["task_id"] for row in config["selection"]["selected"]))
    if (
        [asdict(task) for task in selected] != config["selection"]["selected"]
        or len(selected) != NEW_FAMILIES
        or len({task.family for task in selected}) != NEW_FAMILIES
    ):
        raise ValueError("Diversity selection requires exactly six independent eligible families")
    excluded = pinned_record(config, "exclusion_review")
    if (
        excluded["decision"] != "approve"
        or excluded["bank_sha256"] != config["bank_record_sha256"]
        or excluded["family_map_sha256"] != compact_json_sha256(bank["family_by_task"])
    ):
        raise ValueError("Diversity exclusion review does not bind its bank")
    if {task.family for task in selected}.intersection(excluded["excluded_families"]):
        raise ValueError("Acceptance or final families cannot enter teacher diversity")
    decision = PinnedFile(**config["prospective_decision"]).read_json()
    expected = {
        "protocol": PROTOCOL,
        "original_study": config["original_study"],
        "current_selection_sha256": config["current_selection_sha256"],
        "current_coding_sha256": config["current_coding_sha256"],
        "capability_release_sha256": config["capability_release_sha256"],
        "capability_review_sha256": config["capability_review_sha256"],
        "bank_record_sha256": config["bank_record_sha256"],
        "train_sha256": config["selection"]["train_sha256"],
        "retained_collection": config["retained_collection"],
        "retained_train": config["retained_train"],
        "retained_canonical_proof": config["retained_canonical_proof"],
        "historical_attempts": config["historical_attempts"],
        "retained_rows": config["retained_rows"],
        "selected": config["selection"]["selected"],
        "maximum_new_trajectories": len(PERMITTED_SLOTS),
        "first_qualified_new_rows": 4,
        "sft": {
            "rows": ROWS,
            "passes": PASSES,
            "batch_size": ROWS,
            "updates": UPDATES,
            "context_tokens": CONTEXT_TOKENS,
            "learning_rate": SFT_LEARNING_RATE,
            "assistant_only_loss": True,
            "truncate": False,
        },
    }
    if decision != expected:
        raise ValueError("Diversity study differs from its prospective decision")
    return original, lineage, bank


def diversity_plan(config: dict, records: dict[str, str], tokenizer: MarinTokenizer) -> dict:
    """Recompute the four retained rows before any new teacher request."""
    bank = pinned_record(config, "bank_record")
    if set(records) != {entry["task_id"] for entry in bank["tasks"]}:
        raise ValueError("Diversity Parquet inventory differs from its complete admitted bank")
    for entry in bank["tasks"]:
        if digest(json.loads(records[entry["task_id"]])) != entry["task_sha256"]:
            raise ValueError("Diversity raw task differs from its admission hash")
    collection = PinnedFile(**config["retained_collection"]).read_json()
    canonical = PinnedFile(**config["retained_canonical_proof"]).read_json()
    entries = config["retained_rows"]
    train = PinnedFile(**config["retained_train"]).read_bytes()
    if (
        collection["status"] != "passed"
        or collection["protocol"] != RETAINED_PROTOCOL
        or len(collection["accepted"]) != RETAINED_ROWS
        or len(entries) != RETAINED_ROWS
        or tuple(entry["slot"] for entry in entries) != RETAINED_SLOTS
        or canonical["status"] != "canonical_runtime_rows_passed"
        or canonical["train_sha256"] != config["retained_train"]["sha256"]
    ):
        raise ValueError("Diversity study requires all four qualified retained rows")
    retained = []
    for entry, accepted, attested in zip(entries, collection["accepted"], canonical["rows"], strict=True):
        task = accepted["task"]
        slot = accepted["slot"]
        raw = records[task["task_id"]]
        if (
            digest(json.loads(raw)) != task["task_sha256"]
            or task["family"] != bank["family_by_task"][task["task_id"]]
            or entry["slot"] != slot
            or attested["slot"] != slot
            or accepted["attempt"] != int(slot.split("-")[1])
        ):
            raise ValueError("Retained row differs from its admitted task or frozen order")
        rollout = PinnedFile(**entry["rollout"]).read_json()
        saved = PinnedFile(**entry["student_row"]).read_json()
        grade = PinnedFile(**entry["qualification"]).read_json()
        row = qualified_row(rollout, TaskSpec.model_validate_json(raw), tokenizer)
        if (
            row is None
            or row["example"] != saved["example"]
            or row["example"] != accepted["row"]
            or compact_json_sha256(row["example"]) != accepted["row_sha256"]
            or len(row["input_ids"]) != attested["tokens"]
            or sum(row["assistant_mask"]) != attested["assistant_targets"]
            or grade["status"] != "accepted"
            or grade["task"] != task
            or grade["attempt"] != accepted["attempt"]
            or grade["rollout_sha256"] != compact_json_sha256(rollout)
        ):
            raise ValueError("Retained row differs from full trajectory, grade or canonical token witness")
        retained.append({**accepted, "witness": row})
    if "".join(json.dumps(row["row"], sort_keys=True) + "\n" for row in retained).encode() != train:
        raise ValueError("Retained training bytes differ from the four accepted examples")
    families = {row["task"]["family"] for row in retained}
    if (
        len(families) != RETAINED_ROWS
        or len({row["row_sha256"] for row in retained}) != RETAINED_ROWS
        or families.intersection(row["family"] for row in config["selection"]["selected"])
    ):
        raise ValueError("New diversity families overlap retained families or duplicate examples")
    history = PinnedFile(**config["historical_attempts"]).read_json()
    if (
        history["consumed_trajectories"] != collection["cumulative_trajectories"]
        or len(history["records"]) != history["consumed_trajectories"]
        or len({(entry["producer"], entry["slot"]) for entry in history["records"]}) != len(history["records"])
        or any(not entry["family"] for entry in history["records"])
    ):
        raise ValueError("Historical trajectory count differs from the retained collection")
    return {
        "protocol": PROTOCOL,
        "selection": config["selection"],
        "permitted_slots": list(PERMITTED_SLOTS),
        "consumed_trajectories": history["consumed_trajectories"],
        "historical_attempts": history,
        "retained": retained,
    }


def run_diversity_collection(config: DiversityCollectionConfig) -> None:
    study = config.study
    require_diversity_condition(study)
    write_once(StoragePath(config.collection.output_path) / "diversity-study.json", study)
    collect_chat_dataset(
        config.collection,
        train_sha256=study["selection"]["train_sha256"],
        template=StudentTrainingTemplate(
            study["student_training_template_uri"], study["student_training_template_sha256"]
        ),
        plan=lambda records, tokenizer: diversity_plan(study, records, tokenizer),
        required_rows=ROWS,
        passes=PASSES,
        batch_size=ROWS,
        updates=UPDATES,
    )


def run_diversity_remote(config: DiversityCollectionConfig) -> None:
    remote(
        run_diversity_collection,
        resources=ResourceConfig.with_cpu(cpu=8, ram="64GB", disk="64GB"),
        pip_packages=list(TEACHER_PIP_PACKAGES),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def diversity_workflow(config: dict) -> dict[str, ArtifactStep]:
    require_diversity_condition(config)
    scientific = {
        **config,
        "version": config["collection_version"],
        "dose_decision_uri": config["current_selection_uri"],
        "dose_decision_sha256": config["current_selection_sha256"],
    }
    return teacher_sft_steps(
        scientific,
        CollectionBinding(lambda base: DiversityCollectionConfig(base, config), run_diversity_remote),
        UPDATES,
        NAMESPACE,
        CONTEXT_TOKENS,
        training_version=config["version"],
    )


def run_diversity_calibration(config: DevelopmentEvaluationConfig) -> None:
    if not MINIMUM_BANK_TASKS <= config.limit <= MAXIMUM_BANK_TASKS:
        raise ValueError("Diversity calibration requires its complete bank of 32 to 36 tasks")
    journal = calibration_evaluation_journal(config)
    run_development_evaluation(config, journal=journal)


def diversity_post_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    trained = diversity_workflow(pinned_record(config, "sft_config"))["train"]
    return validated_diversity_post_workflow(config, stage, trained=trained)


def validated_diversity_post_workflow(config: dict, stage: str, *, trained: ArtifactStep) -> dict[str, ArtifactStep]:
    """Apply the original study gates to an explicitly bound training producer."""
    if config["protocol"] != PROTOCOL or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise ValueError("Diversity post-SFT protocol or runtime differs")
    study = pinned_record(config, "sft_config")
    original, lineage, bank_record = require_diversity_condition(study)
    for key in ("parent", "bank", "runtime_bundle"):
        if config[key] != study[key]:
            raise ValueError(f"Diversity post-SFT changed its {key}")
    source_config = pinned_record(original, "continuation_config")
    for key in ("retention", "machine_config", "panel_uri", "panel_sha256"):
        if config[key] != source_config[key]:
            raise ValueError(f"Diversity post-SFT changed its original {key}")
    completed = restored_round(pinned_record(source_config, "source_round"))
    source = pinned_record(source_config, "source_replay")
    if source != completed.replay_plan:
        raise ValueError("Diversity source replay differs from its sealed source round")
    validate_source_replay(source, completed.plan)
    schedule = pinned_record(source_config, "calibration_decision")["schedule"]
    bank, retention = adopted(config["bank"]), adopted(config["retention"])
    source_plan = replace(
        completed.plan,
        task_bank=tuple(QualifiedTask(**item) for item in bank_record["tasks"]),
        bank_identity=artifact_identity(bank),
    )
    qualification = pinned_record(config, "qualification")
    if qualification["source_config_sha256"] != config["sft_config_sha256"]:
        raise ValueError("Diversity qualification identifies different SFT inputs")
    export = qualified_four_update_sft(qualification, identity=artifact_identity(trained), root=config["sft_uri"])
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        config["version"],
        export,
        kind=LevanterCheckpoint,
        config={"sft": artifact_identity(trained), "qualification_sha256": config["qualification_sha256"]},
    )
    baseline = checkpoint_score(lineage["incumbent"])
    parent = checkpoint_score(lineage["original_parent"])
    ids = original_retention_task_ids(source_config, artifact_identity(retention))
    for label, score in (("incumbent", baseline), ("parent", parent)):
        if (
            evaluated_score(
                pinned_record(source_config, f"{label}_coding"),
                pinned_record(source_config, f"{label}_retention"),
                score.checkpoint_identity,
                original["coding_panel_sha256"],
                artifact_identity(retention),
                ids,
            )
            != score
        ):
            raise ValueError("Diversity baseline differs from original pinned evidence")
    return post_sft_stages(
        config,
        stage,
        model=model,
        bank=bank,
        retention=retention,
        source_plan=source_plan,
        source={**schedule, "family_by_task": bank_record["family_by_task"]},
        export_uri=export,
        study=StudyBaseline(PROTOCOL, baseline, parent, ids),
        calibration_runner=run_diversity_calibration,
    )
