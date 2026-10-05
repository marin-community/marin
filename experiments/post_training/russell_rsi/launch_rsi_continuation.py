# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate an expanded bank and run one four-update trial from the current champion."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import (
    PILOT_UPDATES,
    CheckpointScore,
    QualifiedTask,
    RoundPlan,
    calibration_measurements,
    promotes,
    qualified_bank,
    restored_round,
    write_once,
)
from experiments.post_training.russell_rsi.calibrated_trial import bounded_schedule, four_update_trial
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    CodingPanel,
    PanelItem,
    collect_coding_eval_evidence,
)
from experiments.post_training.russell_rsi.launch import (
    CALIBRATION_TEMPERATURE,
    CLUSTER,
    MODEL,
    MODEL_REVISION,
    development_step,
    evaluation_model,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import RETENTION_TASKS, adopted, evaluated_score
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.replay import (
    REPLAY_SEED,
    ReplayDatasetConfig,
    calibration_signal_failure,
    freeze_replay_dataset,
    sampled_replay_plan,
    validate_replay_plan,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model

PROTOCOL = "champion-rsi-r1"
UPDATES = PILOT_UPDATES
MAX_NEW_FAMILIES = 4
RETAINED_TASKS = 26
PARENT_CODING = (25 / 32, 25 / 32)
INCUMBENT_CODING = (25 / 32, 27 / 32)
BASELINE_RETENTION = 1 / 3


def qualified_champion(record: dict, source: dict) -> str:
    """Return the qualified export of the original training artifact."""
    identity = f"{source['name']}@{source['version']}:{source['fingerprint']}"
    export = prefix_join(source["output_path"], "exports/global_step_8/policy")
    reload = record["serving_reload"]
    if (
        source["result_type"] != "marin.rl.skyrl.SkyRLRun"
        or source["result"]["global_step"] != 8
        or source["result"]["hf_model_uri"] != export
        or source["result"]["tokenizer_uri"] != MODEL
        or source["result"]["tokenizer_revision"] != MODEL_REVISION
        or record["protocol"] != "rsi-champion-eight-update-qualification-v1"
        or record["model_identity"] != identity
        or record["model_root"] != source["output_path"]
        or record["hf_export_uri"] != export
        or record["optimizer_updates"] != 8
        or record["export_verified"] is not True
        or not record["export_evidence_uri"]
        or len(record["export_evidence_sha256"]) != 64
        or reload["verified"] is not True
        or reload["model_identity"] != identity
        or reload["model_uri"] != export
        or reload["suite"] != "mmlu-smoke"
        or reload["limit"] != 1
        or not reload["evidence_uri"]
        or len(reload["evidence_sha256"]) != 64
    ):
        raise ValueError("Continuation requires the qualified eight-update champion export")
    return export


def expanded_bank(
    record: dict, retained: tuple[QualifiedTask, ...], old_families: dict[str, str]
) -> tuple[QualifiedTask, ...]:
    """Keep all retained contracts and admit only new independent families."""
    tasks = tuple(QualifiedTask(**item) for item in record["tasks"])
    families = record["family_by_task"]
    additions = tasks[RETAINED_TASKS:]
    if (
        tasks[:RETAINED_TASKS] != retained
        or not 1 <= len(additions) <= MAX_NEW_FAMILIES
        or qualified_bank(tasks) != tasks
        or set(families) != {task.task_id for task in tasks}
        or any(families[task.task_id] != old_families[task.task_id] for task in retained)
        or any(
            task.relation not in {"independent", "new_contract"} or families[task.task_id] != task.contract_id
            for task in additions
        )
    ):
        raise ValueError("Continuation requires all 26 unchanged tasks and one to four new independent families")
    return tasks


def continuation_schedule(summary: dict, plan: RoundPlan, families: dict[str, str]) -> dict:
    """Use the existing family sampler with a fresh, bounded trial identity."""
    measurements = calibration_measurements(summary, plan.task_bank, plan.current_checkpoint, plan.bank_identity)
    failure = calibration_signal_failure(measurements)
    if failure is not None:
        return {"protocol": PROTOCOL, "signal_gate_passed": False, "reason": failure, "schedule": None}
    schedule = sampled_replay_plan(
        plan,
        measurements,
        plan.task_bank[RETAINED_TASKS:],
        pilot_number=2,
        bank_identity=plan.bank_identity,
        calibration_identity=plan.calibration_identity,
        frozen_identity=f"{PROTOCOL}-replay",
        parent_identity=plan.current_checkpoint,
        model_identity=plan.current_checkpoint,
        family_by_task=families,
        updates=UPDATES,
        seed=REPLAY_SEED,
    )
    if not schedule["signal_gate_passed"]:
        return {
            "protocol": PROTOCOL,
            "signal_gate_passed": False,
            "reason": "weighted_q4_below_threshold",
            "schedule": None,
        }
    schedule = bounded_schedule(
        schedule,
        PROTOCOL,
        [
            "Replay repeats contracts and creates no independent evaluation evidence.",
        ],
    )
    return {"protocol": PROTOCOL, "signal_gate_passed": True, "reason": None, "schedule": schedule}


@dataclass(frozen=True)
class ContinuationCalibrationConfig:
    summary_path: str
    plan: RoundPlan
    families: dict[str, str]
    qualification_sha256: str
    output_path: str


def seal_continuation_calibration(config: ContinuationCalibrationConfig) -> None:
    raw = StoragePath(prefix_join(config.summary_path, "failure_summary.json")).read_bytes()
    write_once(
        StoragePath(prefix_join(config.output_path, "calibration-decision.json")),
        {
            **continuation_schedule(json.loads(raw), config.plan, config.families),
            "plan": asdict(config.plan),
            "qualification_sha256": config.qualification_sha256,
            "summary_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )


@dataclass(frozen=True)
class ContinuationSelectionConfig:
    coding_path: str
    retention_path: str
    candidate_identity: str
    panel_sha256: str
    retention_identity: str
    retention_task_ids: tuple[str, ...]
    incumbent: CheckpointScore
    parent: CheckpointScore
    output_path: str


def seal_continuation_selection(config: ContinuationSelectionConfig) -> None:
    coding = json.loads(StoragePath(prefix_join(config.coding_path, "coding-evidence.json")).read_text())
    retention = json.loads(StoragePath(prefix_join(config.retention_path, "failure_summary.json")).read_text())
    candidate = evaluated_score(
        coding,
        retention,
        config.candidate_identity,
        config.panel_sha256,
        config.retention_identity,
        config.retention_task_ids,
    )
    selected = candidate if promotes(candidate, config.incumbent) else config.incumbent
    write_once(
        StoragePath(prefix_join(config.output_path, "continuation-selection.json")),
        {
            "protocol": PROTOCOL,
            "incumbent": asdict(config.incumbent),
            "candidate": asdict(candidate),
            "selected": asdict(selected),
            "original_parent": asdict(config.parent),
        },
    )


def continuation_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    """Build the requested stage from immutable champion and bank evidence."""
    version = config["version"]
    source_config = json.loads(pinned_bytes(config["source_config_uri"], config["source_config_sha256"]))
    for key in ("parent", "retention", "machine_config", "runtime_bundle", "panel_uri", "panel_sha256"):
        if config[key] != source_config[key]:
            raise ValueError(f"Continuation {key} differs from the frozen source protocol")
    completed = restored_round(json.loads(pinned_bytes(config["source_round_uri"], config["source_round_sha256"])))
    source_replay = json.loads(pinned_bytes(config["source_replay_uri"], config["source_replay_sha256"]))
    if completed.replay_plan is None or source_replay != completed.replay_plan:
        raise ValueError("Continuation source replay differs from its sealed round")
    validate_replay_plan(
        source_replay,
        completed.plan,
        **{
            key: source_replay[key]
            for key in (
                "pilot_number",
                "bank_identity",
                "calibration_identity",
                "frozen_identity",
                "parent_identity",
                "model_identity",
            )
        },
        family_by_task=source_replay["family_by_task"],
    )
    old_bank = json.loads(pinned_bytes(config["old_bank_record_uri"], config["old_bank_record_sha256"]))
    if (
        tuple(QualifiedTask(**item) for item in old_bank["tasks"]) != completed.plan.task_bank
        or config["old_bank_record_sha256"] != source_config["bank"]["identity_config"]["bank_sha256"]
        or artifact_identity(adopted(source_config["bank"])) != completed.plan.bank_identity
    ):
        raise ValueError("Continuation retained bank differs from its frozen source")
    bank_record = json.loads(pinned_bytes(config["bank_record_uri"], config["bank_record_sha256"]))
    if config["bank_record_sha256"] != config["bank"]["identity_config"]["bank_sha256"]:
        raise ValueError("Continuation bank record differs from its adopted manifest")
    tasks = expanded_bank(bank_record, completed.plan.task_bank, source_replay["family_by_task"])
    source_model = json.loads(pinned_bytes(config["incumbent_artifact_uri"], config["incumbent_artifact_sha256"]))
    qualification = json.loads(pinned_bytes(config["qualification_uri"], config["qualification_sha256"]))
    if (
        qualification["source_artifact_sha256"] != config["incumbent_artifact_sha256"]
        or qualification["source_artifact_uri"] != config["incumbent_artifact_uri"]
    ):
        raise ValueError("Champion qualification identifies a different source artifact")
    export = qualified_champion(qualification, source_model)
    original_identity = qualification["model_identity"]
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        version,
        export,
        kind=LevanterCheckpoint,
        config={
            "source_identity": original_identity,
            "source_artifact_sha256": config["incumbent_artifact_sha256"],
            "qualification_sha256": config["qualification_sha256"],
        },
    )
    bank, retention = adopted(config["bank"]), adopted(config["retention"])
    parent = adopted(config["parent"], LevanterCheckpoint)
    if completed.plan.current_checkpoint != artifact_identity(
        parent
    ) or completed.plan.retention_identity != artifact_identity(retention):
        raise ValueError("Continuation parent or retention differs from its source round")
    panel_value = json.loads(pinned_bytes(config["panel_uri"], config["panel_sha256"]))
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    panel_digest = compact_json_sha256(asdict(panel))
    scores = []
    task_ids: tuple[str, ...] = ()
    for label, identity in (("parent", artifact_identity(parent)), ("incumbent", original_identity)):
        coding = json.loads(pinned_bytes(config[f"{label}_coding_uri"], config[f"{label}_coding_sha256"]))
        retained = json.loads(pinned_bytes(config[f"{label}_retention_uri"], config[f"{label}_retention_sha256"]))
        ids = tuple(sorted(retained["task_rewards"]))
        if task_ids and ids != task_ids:
            raise ValueError("Continuation baselines use different retention contracts")
        task_ids = ids
        scores.append(evaluated_score(coding, retained, identity, panel_digest, artifact_identity(retention), ids))
    parent_score, incumbent_score = scores
    if (
        parent_score.development != PARENT_CODING
        or incumbent_score.development != INCUMBENT_CODING
        or parent_score.retention != BASELINE_RETENTION
        or incumbent_score.retention != BASELINE_RETENTION
    ):
        raise ValueError("Continuation requires the frozen parent and current champion baselines")
    runtime = RuntimeBundle(**config["runtime_bundle"])
    calibration = development_step(
        bank,
        model,
        version,
        runtime,
        f"{PROTOCOL}-calibration",
        relative_path="train.parquet",
        samples_per_task=8,
        temperature=CALIBRATION_TEMPERATURE,
        require_reward_variation=False,
        limit=len(tasks),
        startup_attempts=3,
    )
    plan = replace(
        completed.plan,
        name=PROTOCOL,
        current_checkpoint=artifact_identity(model),
        champion_checkpoint=artifact_identity(model),
        task_bank=tasks,
        bank_identity=artifact_identity(bank),
        calibration_identity=artifact_identity(calibration),
        feedback_labels=(),
        selected_tasks=tasks,
        retained_count=RETAINED_TASKS,
        fresh_count=len(tasks) - RETAINED_TASKS,
        absent_bands=(),
        seed=REPLAY_SEED,
        updates=UPDATES,
        max_glm_responses=0,
    )
    decision = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-calibration-decision",
        version=version,
        artifact_type=Artifact,
        deps=(calibration, bank, model),
        build_config=lambda ctx: ContinuationCalibrationConfig(
            ctx.artifact_path(calibration),
            plan,
            bank_record["family_by_task"],
            config["qualification_sha256"],
            ctx.output_path,
        ),
        run=seal_continuation_calibration,
    )
    if stage == "calibrate":
        return {"calibration": calibration, "decision": decision, "terminal": decision}
    record = json.loads(pinned_bytes(config["calibration_decision_uri"], config["calibration_decision_sha256"]))
    summary = json.loads(pinned_bytes(config["calibration_summary_uri"], record["summary_sha256"]))
    expected = {
        **continuation_schedule(summary, plan, bank_record["family_by_task"]),
        "plan": asdict(plan),
        "qualification_sha256": config["qualification_sha256"],
        "summary_sha256": record["summary_sha256"],
    }
    if compact_json_sha256(record) != compact_json_sha256(expected):
        raise ValueError("Continuation calibration decision does not match its complete evidence")
    saved_decision = ArtifactStep.adopt(
        f"documents/russell-rsi-{PROTOCOL}-pinned-decision",
        version,
        str(StoragePath(config["calibration_decision_uri"]).parent),
        config={"decision_sha256": config["calibration_decision_sha256"]},
    )
    if not record["signal_gate_passed"]:
        return {"terminal": saved_decision}
    data = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-replay",
        version=version,
        artifact_type=Artifact,
        deps=(bank, model, saved_decision),
        build_config=lambda ctx: ReplayDatasetConfig(
            ctx.artifact_path(bank), ctx.output_path, tasks, record["schedule"]
        ),
        run=remote(
            freeze_replay_dataset,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
            pip_packages=["./lib/taskcompendium"],
        ),
    )
    trial = four_update_trial(data, model, version, retention, config["machine_config"], PROTOCOL)
    trained, updates, reload = trial["rl"], trial["updates"], trial["reload"]
    evaluation = evaluation_model(f"russell-rsi-{PROTOCOL}", SKYRL_POLICY_LOCATION, None)
    outputs = {"rl": trained, "reload": reload}
    if stage == "train":
        return {**outputs, "terminal": reload}
    coding = eval_step(
        evaluation,
        "humanevalplus,mbppplus",
        version=version,
        deps=(trained, updates, reload),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, evaluation),
        limit=32,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )

    def evidence_config(ctx: StepContext):
        if ctx.is_fingerprint:
            return {"evaluation": artifact_identity(coding), "model": artifact_identity(trained), "panel": panel}
        result = ctx.resolved(coding)
        return CodingEvidenceConfig(
            result.records_prefix,
            result.run_ids,
            result.results_paths,
            artifact_identity(trained),
            panel,
            ctx.output_path,
        )

    evidence = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-coding",
        version=version,
        artifact_type=Artifact,
        deps=(coding, trained),
        build_config=evidence_config,
        run=collect_coding_eval_evidence,
    )
    retained = development_step(retention, trained, version, runtime, f"{PROTOCOL}-retention", limit=RETENTION_TASKS)
    retained = replace(retained, deps=tuple(dict.fromkeys((*retained.deps, updates, reload))))
    selection = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-selection",
        version=version,
        artifact_type=Artifact,
        deps=(evidence, retained),
        build_config=lambda ctx: ContinuationSelectionConfig(
            ctx.artifact_path(evidence),
            ctx.artifact_path(retained),
            artifact_identity(trained),
            panel_digest,
            artifact_identity(retention),
            task_ids,
            incumbent_score,
            parent_score,
            ctx.output_path,
        ),
        run=seal_continuation_selection,
    )
    return {**outputs, "coding": evidence, "retention": retained, "selection": selection, "terminal": selection}


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["calibrate", "train", "evaluate"]), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if (
        resolve_version("russell-rsi-continuation", None) != config["version"]
        or config["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise click.UsageError("Continuation config version or runtime pin differs")
    return [continuation_workflow(config, stage)["terminal"]]


if __name__ == "__main__":
    main()
