# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate one teacher SFT update, then run at most one four-update GRPO trial."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.evaluation.pipeline import EvalStepConfig, EvaluationResult, eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    QualifiedTask,
    RoundPlan,
    calibration_measurements,
    checkpoint_score,
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
    development_step,
    evaluation_model,
)
from experiments.post_training.russell_rsi.launch_dose_comparison import selected_dose
from experiments.post_training.russell_rsi.launch_teacher_sft import SFT_LEARNING_RATE
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.replay import (
    REPLAY_SEED,
    ReplayDatasetConfig,
    calibration_signal_failure,
    freeze_replay_dataset,
    sampled_replay_plan,
    validate_replay_plan,
)
from experiments.post_training.russell_rsi.rollout_eval import (
    CALIBRATION_SAMPLES,
    CALIBRATION_STARTUP_ATTEMPTS,
    DevelopmentEvaluationConfig,
    run_calibration_evaluation,
    run_development_evaluation,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model

PROTOCOL = "teacher-sft-r1"
UPDATES = 4
BANK_TASKS = 26
RETENTION_TASKS = 3


def qualified_sft(record: dict, *, identity: str, root: str) -> str:
    """Return the attested export of exactly one qualified SFT update."""
    return _qualified_sft(
        record, identity=identity, root=root, updates=1, protocol="teacher-sft-one-update-qualification-v1"
    )


def _qualified_sft(record: dict, *, identity: str, root: str, updates: int, protocol: str) -> str:
    export = prefix_join(root, f"hf/step-{updates - 1}")
    shards = record["hf_shards"]
    shard_paths = {item["path"] for item in shards}
    if (
        not shards
        or len(shard_paths) != len(shards)
        or shard_paths != set(record["hf_weight_map"].values())
        or any(item["size"] <= 0 for item in shards)
    ):
        raise ValueError("SFT shard inventory does not match its index weight map")
    if (
        record["protocol"] != protocol
        or record["sft_identity"] != identity
        or record["sft_root"] != root
        or record["hf_export_uri"] != export
        or record["optimizer_updates"] != updates
        or record["learning_rate"] != SFT_LEARNING_RATE
        or any(not math.isfinite(record[key]) for key in ("loss", "gradient_norm", "update_norm"))
        or any(record[key] <= 0 for key in ("gradient_norm", "update_norm"))
        or record["serving_reload"]["verified"] is not True
        or record["serving_reload"]["model_uri"] != export
        or record["serving_reload"]["suite"] != "mmlu-smoke"
        or record["serving_reload"]["limit"] != 1
        or not record["serving_reload"]["evidence_uri"]
        or len(record["serving_reload"]["evidence_sha256"]) != 64
        or not record["hf_files"]
        or any(not item["path"] or len(item["sha256"]) != 64 for item in record["hf_files"])
        or not all(record["hf_verified"][key] is True for key in ("shards", "config", "tokenizer", "eos"))
    ):
        raise ValueError(f"Post-SFT requires the pinned {updates}-update export and serving qualification")
    return export


def post_sft_plan(source: RoundPlan, *, model: str, calibration: str, protocol: str = PROTOCOL) -> RoundPlan:
    return replace(
        source,
        name=protocol,
        current_checkpoint=model,
        champion_checkpoint=model,
        calibration_identity=calibration,
        feedback_labels=(),
        selected_tasks=source.task_bank,
        retained_count=len(source.task_bank),
        fresh_count=0,
        absent_bands=(),
        seed=REPLAY_SEED,
        updates=UPDATES,
        max_glm_responses=0,
    )


def post_sft_schedule(summary: dict, plan: RoundPlan, source: dict, protocol: str = PROTOCOL) -> dict:
    """Keep the legacy family sampler, with a fresh trial namespace and limits."""
    measurements = calibration_measurements(summary, plan.task_bank, plan.current_checkpoint, plan.bank_identity)
    failure = calibration_signal_failure(measurements)
    if failure is not None:
        return {"protocol": protocol, "signal_gate_passed": False, "reason": failure, "schedule": None}
    by_id = {task.task_id: task for task in plan.task_bank}
    schedule = sampled_replay_plan(
        plan,
        measurements,
        tuple(by_id[key] for key in source["sampling_spec"]["targeted_task_ids"]),
        pilot_number=2,
        bank_identity=plan.bank_identity,
        calibration_identity=plan.calibration_identity,
        frozen_identity=f"{protocol}-replay",
        parent_identity=plan.current_checkpoint,
        model_identity=plan.current_checkpoint,
        family_by_task=source["family_by_task"],
        updates=UPDATES,
        seed=REPLAY_SEED,
    )
    if not schedule["signal_gate_passed"]:
        return {
            "protocol": protocol,
            "signal_gate_passed": False,
            "reason": "weighted_q4_below_threshold",
            "schedule": None,
        }
    schedule = bounded_schedule(
        schedule,
        protocol,
        [
            "Replay repeats existing contracts and creates no independent evidence.",
            "Calibration estimates do not establish a causal benefit of teacher SFT.",
        ],
        source["schedule_sha256"],
    )
    return {"protocol": protocol, "signal_gate_passed": True, "reason": None, "schedule": schedule}


@dataclass(frozen=True)
class CalibrationRecordConfig:
    summary_path: str
    plan: RoundPlan
    source_replay: dict
    qualification_sha256: str
    output_path: str


def seal_calibration(config: CalibrationRecordConfig) -> None:
    _seal_calibration(config, PROTOCOL)


def _seal_calibration(config: CalibrationRecordConfig, protocol: str) -> None:
    raw = StoragePath(prefix_join(config.summary_path, "failure_summary.json")).read_bytes()
    result = post_sft_schedule(json.loads(raw), config.plan, config.source_replay, protocol)
    write_once(
        StoragePath(prefix_join(config.output_path, "calibration-decision.json")),
        {
            **result,
            "plan": asdict(config.plan),
            "qualification_sha256": config.qualification_sha256,
            "summary_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )


@dataclass(frozen=True)
class SelectionConfig:
    coding_paths: tuple[str, ...]
    retention_paths: tuple[str, ...]
    model_identities: tuple[str, ...]
    panel_sha256: str
    retention_identity: str
    retention_task_ids: tuple[str, ...]
    parent: CheckpointScore
    output_path: str


def evaluated_score(
    coding: dict, retention: dict, identity: str, panel_sha256: str, retention_identity: str, task_ids: tuple[str, ...]
) -> CheckpointScore:
    rewards = retention["task_rewards"]
    scores = tuple(coding["scores"][key] for key in ("humanevalplus", "mbppplus"))
    if (
        coding["model_identity"] != identity
        or coding["panel_sha256"] != panel_sha256
        or retention["model_identity"] != identity
        or retention["tasks_identity"] != retention_identity
        or retention["count"] != RETENTION_TASKS
        or len(task_ids) != RETENTION_TASKS
        or set(rewards) != set(task_ids)
        or any(len(group) != 1 or group[0] not in (0, 1) for group in rewards.values())
        or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
    ):
        raise ValueError("Post-SFT selection requires complete identical coding and retention panels")
    return CheckpointScore(identity, scores, sum(group[0] for group in rewards.values()) / RETENTION_TASKS)


def selection_scores(config: SelectionConfig) -> list[CheckpointScore]:
    if len(config.model_identities) not in (1, 2):
        raise ValueError("Post-SFT selection requires SFT alone or SFT and its one RL trial")
    scores = []
    for coding_path, retention_path, identity in zip(
        config.coding_paths, config.retention_paths, config.model_identities, strict=True
    ):
        coding = json.loads(StoragePath(prefix_join(coding_path, "coding-evidence.json")).read_text())
        retention = json.loads(StoragePath(prefix_join(retention_path, "failure_summary.json")).read_text())
        scores.append(
            evaluated_score(
                coding, retention, identity, config.panel_sha256, config.retention_identity, config.retention_task_ids
            )
        )
    return scores


def selection_record(config: SelectionConfig) -> dict:
    scores = selection_scores(config)
    selected = scores[0] if len(scores) == 1 else selected_dose(*scores)
    return {
        "protocol": PROTOCOL,
        "parent": asdict(config.parent),
        "sft": asdict(scores[0]),
        "sft_rl": asdict(scores[1]) if len(scores) == 2 else None,
        "selected": asdict(selected),
        "promoted": asdict(selected_dose(config.parent, selected)),
    }


def seal_selection(config: SelectionConfig) -> None:
    write_once(StoragePath(prefix_join(config.output_path, "post-sft-selection.json")), selection_record(config))


def adopted(value: dict, kind: type = Artifact) -> ArtifactStep:
    return ArtifactStep.adopt(value["name"], value["version"], value["uri"], kind=kind, config=value["identity_config"])


def validate_source_replay(source: dict, plan: RoundPlan) -> None:
    """Check a pinned replay against its sealed source plan."""
    validate_replay_plan(
        source,
        plan,
        **{
            key: source[key]
            for key in (
                "pilot_number",
                "bank_identity",
                "calibration_identity",
                "frozen_identity",
                "parent_identity",
                "model_identity",
            )
        },
        family_by_task=source["family_by_task"],
    )


def post_sft_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    """Build only the requested stage from qualified, immutable inputs."""
    version = config["version"]
    source_config = json.loads(pinned_bytes(config["source_config_uri"], config["source_config_sha256"]))
    for key in ("parent", "retention", "bank", "machine_config", "runtime_bundle", "panel_uri", "panel_sha256"):
        if config[key] != source_config[key]:
            raise ValueError(f"Post-SFT {key} differs from the frozen source protocol")
    sft_config = json.loads(pinned_bytes(config["sft_config_uri"], config["sft_config_sha256"]))
    if any(sft_config[key] != config[key] for key in ("parent", "bank", "runtime_bundle")):
        raise ValueError("SFT training configuration differs from its frozen parent and bank")
    sft = adopted(config["sft"], LevanterCheckpoint)
    qualification = json.loads(pinned_bytes(config["qualification_uri"], config["qualification_sha256"]))
    if qualification["source_config_sha256"] != config["sft_config_sha256"]:
        raise ValueError("SFT qualification identifies a different training configuration")
    export_uri = qualified_sft(qualification, identity=artifact_identity(sft), root=config["sft"]["uri"])
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        version,
        export_uri,
        kind=LevanterCheckpoint,
        config={"sft": artifact_identity(sft), "qualification_sha256": config["qualification_sha256"]},
    )
    bank = adopted(config["bank"])
    retention = adopted(config["retention"])
    completed = restored_round(json.loads(pinned_bytes(config["source_round_uri"], config["source_round_sha256"])))
    source = json.loads(pinned_bytes(config["source_replay_uri"], config["source_replay_sha256"]))
    sealed_replay = completed.replay_plan
    if sealed_replay is None or source != sealed_replay:
        raise ValueError("Post-SFT source replay differs from its sealed pilot-two round")
    validate_source_replay(source, completed.plan)
    if config["bank_record_sha256"] != config["bank"]["identity_config"]["bank_sha256"]:
        raise ValueError("Post-SFT bank metadata is not the adopted bank manifest")
    bank_record = json.loads(pinned_bytes(config["bank_record_uri"], config["bank_record_sha256"]))
    if (
        tuple(QualifiedTask(**item) for item in bank_record["tasks"]) != completed.plan.task_bank
        or len(completed.plan.task_bank) != BANK_TASKS
        or artifact_identity(bank) != source["bank_identity"]
        or artifact_identity(bank) != completed.plan.bank_identity
        or completed.plan.current_checkpoint != artifact_identity(adopted(config["parent"], LevanterCheckpoint))
        or completed.plan.retention_identity != artifact_identity(retention)
        or len(source["sampling_spec"]["targeted_task_ids"]) != 2
    ):
        raise ValueError("Post-SFT requires the unchanged 26-task bank and two targeted source contracts")
    return post_sft_stages(
        config,
        stage,
        model=model,
        bank=bank,
        retention=retention,
        source_plan=completed.plan,
        source=source,
        export_uri=export_uri,
    )


@dataclass(frozen=True)
class StudyBaseline:
    protocol: str
    incumbent: CheckpointScore
    original_parent: CheckpointScore
    retention_task_ids: tuple[str, ...]


def post_sft_stages(
    config: dict,
    stage: str,
    *,
    model: ArtifactStep,
    bank: ArtifactStep,
    retention: ArtifactStep,
    source_plan: RoundPlan,
    source: dict,
    export_uri: str,
    study: StudyBaseline | None = None,
    calibration_runner: Callable[[DevelopmentEvaluationConfig], None] | None = None,
) -> dict[str, ArtifactStep]:
    """Use the already qualified model and validated bank, replay and baseline."""
    protocol = PROTOCOL if study is None else study.protocol
    version = config["version"]
    runtime = RuntimeBundle(**config["runtime_bundle"])
    runner = run_development_evaluation if study is None else run_calibration_evaluation
    if calibration_runner is not None:
        runner = calibration_runner
    calibration = development_step(
        bank,
        model,
        version,
        runtime,
        f"{protocol}-calibration",
        relative_path="train.parquet",
        samples_per_task=CALIBRATION_SAMPLES,
        temperature=CALIBRATION_TEMPERATURE,
        require_reward_variation=False,
        limit=len(source_plan.task_bank),
        startup_attempts=CALIBRATION_STARTUP_ATTEMPTS,
        evaluation_runner=runner,
    )
    plan = post_sft_plan(
        source_plan, model=artifact_identity(model), calibration=artifact_identity(calibration), protocol=protocol
    )

    def calibration_config(ctx: StepContext):
        record = CalibrationRecordConfig(
            ctx.artifact_path(calibration), plan, source, config["qualification_sha256"], ctx.output_path
        )
        return record if study is None else StudyCalibrationConfig(record, protocol)

    decision = ArtifactStep(
        name=f"documents/russell-rsi-{protocol}-calibration-decision",
        version=version,
        artifact_type=Artifact,
        deps=(calibration, bank, model),
        build_config=calibration_config,
        run=seal_calibration if study is None else seal_study_calibration,
    )
    if stage == "calibrate":
        return {"calibration": calibration, "decision": decision, "terminal": decision}
    record = json.loads(pinned_bytes(config["calibration_decision_uri"], config["calibration_decision_sha256"]))
    summary = json.loads(pinned_bytes(config["calibration_summary_uri"], record["summary_sha256"]))
    expected = post_sft_schedule(summary, plan, source, protocol)
    if compact_json_sha256(record) != compact_json_sha256(
        {
            **expected,
            "plan": asdict(plan),
            "qualification_sha256": config["qualification_sha256"],
            "summary_sha256": record["summary_sha256"],
        }
    ):
        raise ValueError("Post-SFT calibration decision does not match its complete evidence")
    saved_decision = ArtifactStep.adopt(
        f"documents/russell-rsi-{protocol}-pinned-decision",
        version,
        str(StoragePath(config["calibration_decision_uri"]).parent),
        config={"decision_sha256": config["calibration_decision_sha256"]},
    )
    outputs = {}
    checkpoints = [("sft", model)]
    barriers = ()
    if record["signal_gate_passed"]:
        data = ArtifactStep(
            name=f"documents/russell-rsi-{protocol}-replay",
            version=version,
            artifact_type=Artifact,
            deps=(bank, model, saved_decision),
            build_config=lambda ctx: ReplayDatasetConfig(
                ctx.artifact_path(bank), ctx.output_path, plan.task_bank, record["schedule"]
            ),
            run=remote(
                freeze_replay_dataset,
                resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
                pip_packages=["./lib/taskcompendium"],
            ),
        )
        trial = four_update_trial(data, model, version, retention, config["machine_config"], protocol)
        trained, updates, reload = trial["rl"], trial["updates"], trial["reload"]
        outputs.update({"rl": trained, "reload": reload})
        checkpoints.append(("sft-rl", trained))
        barriers = (trained, updates, reload)
    if stage == "train":
        return {**outputs, "terminal": outputs["reload"] if outputs else saved_decision}
    return post_sft_evaluation_stages(
        config,
        model=model,
        retention=retention,
        export_uri=export_uri,
        checkpoints=checkpoints,
        barriers=barriers,
        outputs=outputs,
        study=study,
    )


def post_sft_evaluation_stages(
    config: dict,
    *,
    model: ArtifactStep,
    retention: ArtifactStep,
    export_uri: str,
    checkpoints: list[tuple[str, ArtifactStep]],
    barriers: tuple[ArtifactStep, ...],
    outputs: dict[str, ArtifactStep],
    study: StudyBaseline | None,
    coding_runner: Callable[[EvalStepConfig], EvaluationResult] | None = None,
    retention_runner: Callable[[DevelopmentEvaluationConfig], None] | None = None,
) -> dict[str, ArtifactStep]:
    """Build the unchanged coding, retention, and selection stages."""
    protocol = PROTOCOL if study is None else study.protocol
    version = config["version"]
    runtime = RuntimeBundle(**config["runtime_bundle"])
    panel_value = json.loads(pinned_bytes(config["panel_uri"], config["panel_sha256"]))
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    panel_digest = compact_json_sha256(asdict(panel))
    if study is None:
        parent_retention = json.loads(pinned_bytes(config["parent_retention_uri"], config["parent_retention_sha256"]))
        retention_task_ids = tuple(sorted(parent_retention["task_rewards"]))
        parent_coding = json.loads(pinned_bytes(config["parent_coding_uri"], config["parent_coding_sha256"]))
        parent = adopted(config["parent"], LevanterCheckpoint)
        parent_score = evaluated_score(
            parent_coding,
            parent_retention,
            artifact_identity(parent),
            panel_digest,
            artifact_identity(retention),
            retention_task_ids,
        )
    else:
        parent_score = study.incumbent
        retention_task_ids = study.retention_task_ids
    # Both checkpoints wait for the same completed RL export and reload when RL is eligible.
    for label, checkpoint in checkpoints:
        evaluation = evaluation_model(
            f"russell-rsi-{protocol}-{label}", export_uri if label == "sft" else SKYRL_POLICY_LOCATION, None
        )

        def resolver(ctx: StepContext, item=checkpoint, selected=evaluation):
            if item.artifact_type is SkyRLRun:
                return resolve_skyrl_model(ctx, item, selected)
            return replace(selected, location=ctx.artifact_path(item), identity=artifact_identity(item))

        coding = eval_step(
            evaluation,
            "humanevalplus,mbppplus",
            version=version,
            deps=(checkpoint, *barriers),
            resolve_model=resolver,
            limit=32,
            accelerator="H100x8",
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
        )

        if coding_runner is not None:
            coding = replace(coding, run=coding_runner)

        def evidence_config(ctx: StepContext, result_step=coding, item=checkpoint):
            if ctx.is_fingerprint:
                return {"evaluation": artifact_identity(result_step), "model": artifact_identity(item), "panel": panel}
            result = ctx.resolved(result_step)
            return CodingEvidenceConfig(
                result.records_prefix,
                result.run_ids,
                result.results_paths,
                artifact_identity(item),
                panel,
                ctx.output_path,
            )

        evidence = ArtifactStep(
            name=f"documents/russell-rsi-{protocol}-{label}-coding",
            version=version,
            artifact_type=Artifact,
            deps=(coding, checkpoint),
            build_config=evidence_config,
            run=collect_coding_eval_evidence,
        )
        retained = development_step(
            retention, checkpoint, version, runtime, f"{protocol}-{label}-retention", limit=RETENTION_TASKS
        )
        if retention_runner is not None:
            retained = replace(retained, run=retention_runner)
        retained = replace(retained, deps=tuple(dict.fromkeys((*retained.deps, *barriers))))
        outputs.update({f"coding-{label}": evidence, f"retention-{label}": retained})
    return post_sft_selection_stages(
        version=version,
        checkpoints=checkpoints,
        outputs=outputs,
        panel_sha256=panel_digest,
        retention=retention,
        retention_task_ids=retention_task_ids,
        parent_score=parent_score,
        study=study,
    )


def post_sft_selection_stages(
    *,
    version: str,
    checkpoints: list[tuple[str, ArtifactStep]],
    outputs: dict[str, ArtifactStep],
    panel_sha256: str,
    retention: ArtifactStep,
    retention_task_ids: tuple[str, ...],
    parent_score: CheckpointScore,
    study: StudyBaseline | None,
) -> dict[str, ArtifactStep]:
    """Build the existing selection rule from coding and retention evidence."""
    protocol = PROTOCOL if study is None else study.protocol
    labels = tuple(label for label, _ in checkpoints)

    def selection_config(ctx: StepContext):
        record = SelectionConfig(
            tuple(ctx.artifact_path(outputs[f"coding-{label}"]) for label in labels),
            tuple(ctx.artifact_path(outputs[f"retention-{label}"]) for label in labels),
            tuple(artifact_identity(item) for _, item in checkpoints),
            panel_sha256,
            artifact_identity(retention),
            retention_task_ids,
            parent_score,
            ctx.output_path,
        )
        return record if study is None else StudySelectionConfig(record, protocol, study.original_parent)

    outputs["selection"] = ArtifactStep(
        name=f"documents/russell-rsi-{protocol}-selection",
        version=version,
        artifact_type=Artifact,
        deps=tuple(outputs[f"{kind}-{label}"] for label in labels for kind in ("coding", "retention")),
        build_config=selection_config,
        run=seal_selection if study is None else seal_study_selection,
    )
    return {**outputs, "terminal": outputs["selection"]}


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["calibrate", "train", "evaluate"]), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if (
        resolve_version("russell-rsi-post-teacher-sft", None) != config["version"]
        or config["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise click.UsageError("Post-SFT config version or runtime pin differs")
    outputs = post_sft_workflow(config, stage)
    return [outputs["terminal"]]


# Separate wrappers preserve the serialized configs of the old one-update artifacts.
@dataclass(frozen=True)
class StudyCalibrationConfig:
    record: CalibrationRecordConfig
    protocol: str


def seal_study_calibration(config: StudyCalibrationConfig) -> None:
    _seal_calibration(config.record, config.protocol)


@dataclass(frozen=True)
class StudySelectionConfig:
    record: SelectionConfig
    protocol: str
    original_parent: CheckpointScore


def seal_study_selection(config: StudySelectionConfig) -> None:
    result = selection_record(config.record)
    result["incumbent"] = result.pop("parent")
    result.update(
        {
            "protocol": config.protocol,
            "original_parent": asdict(config.original_parent),
            "original_parent_comparison": asdict(
                selected_dose(config.original_parent, checkpoint_score(result["selected"]))
            ),
        }
    )
    write_once(StoragePath(prefix_join(config.record.output_path, "post-sft-selection.json")), result)


def qualified_four_update_sft(record: dict, *, identity: str, root: str) -> str:
    """Require four real finite updates and their post-update export."""
    export = _qualified_sft(
        record, identity=identity, root=root, updates=4, protocol="teacher-sft-four-update-qualification-v1"
    )
    steps = record["optimizer_steps"]
    if record["serving_reload"]["model_identity"] != identity:
        raise ValueError("Four-pass serving reload identifies a different SFT model")
    if (
        [step["step"] for step in steps] != list(range(4))
        or any(step["skipped"] is not False or step["learning_rate"] != SFT_LEARNING_RATE for step in steps)
        or any(not math.isfinite(step[key]) for step in steps for key in ("loss", "gradient_norm", "update_norm"))
        or any(step[key] <= 0 for step in steps for key in ("gradient_norm", "update_norm"))
    ):
        raise ValueError("Four-pass SFT requires four complete finite optimizer updates")
    return export


if __name__ == "__main__":
    main()
