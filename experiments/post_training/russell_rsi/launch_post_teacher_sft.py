# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate one teacher SFT update, then run at most one four-update GRPO trial."""

import hashlib
import json
import math
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

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import (
    CheckpointScore,
    QualifiedTask,
    RoundPlan,
    calibration_measurements,
    restored_round,
    write_once,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    CodingPanel,
    PanelItem,
    collect_coding_eval_evidence,
)
from experiments.post_training.russell_rsi.launch import (
    CALIBRATION_TEMPERATURE,
    CLUSTER,
    OptimizerStepConfig,
    SamplingMode,
    development_step,
    evaluation_model,
    require_optimizer_updates,
    train_step,
)
from experiments.post_training.russell_rsi.launch_dose_comparison import selected_dose
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.replay import (
    GROUPS_PER_UPDATE,
    REPLAY_SEED,
    ROLLOUTS_PER_GROUP,
    ReplayDatasetConfig,
    calibration_signal_failure,
    freeze_replay_dataset,
    sampled_replay_plan,
    validate_replay_plan,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model

PROTOCOL = "teacher-sft-r1"
UPDATES = 4
BANK_TASKS = 26
RETENTION_TASKS = 3


def qualified_sft(record: dict, *, identity: str, root: str) -> str:
    """Return the attested export of exactly one qualified SFT update."""
    export = prefix_join(root, "hf/step-0")
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
        record["protocol"] != "teacher-sft-one-update-qualification-v1"
        or record["sft_identity"] != identity
        or record["sft_root"] != root
        or record["hf_export_uri"] != export
        or record["optimizer_updates"] != 1
        or record["learning_rate"] != 1e-6
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
        raise ValueError("Post-SFT requires the pinned one-update export and serving qualification")
    return export


def post_sft_plan(source: RoundPlan, *, model: str, calibration: str) -> RoundPlan:
    return replace(
        source,
        name=PROTOCOL,
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


def post_sft_schedule(summary: dict, plan: RoundPlan, source: dict) -> dict:
    """Keep the legacy family sampler, with a fresh trial namespace and limits."""
    measurements = calibration_measurements(summary, plan.task_bank, plan.current_checkpoint, plan.bank_identity)
    failure = calibration_signal_failure(measurements)
    if failure is not None:
        return {"protocol": PROTOCOL, "signal_gate_passed": False, "reason": failure, "schedule": None}
    by_id = {task.task_id: task for task in plan.task_bank}
    schedule = sampled_replay_plan(
        plan,
        measurements,
        tuple(by_id[key] for key in source["sampling_spec"]["targeted_task_ids"]),
        pilot_number=2,
        bank_identity=plan.bank_identity,
        calibration_identity=plan.calibration_identity,
        frozen_identity=f"{PROTOCOL}-replay",
        parent_identity=plan.current_checkpoint,
        model_identity=plan.current_checkpoint,
        family_by_task=source["family_by_task"],
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
    schedule.pop("schedule_sha256")
    schedule["legacy_sampler"] = {"protocol": schedule["protocol"], "pilot_number": schedule.pop("pilot_number")}
    schedule["source_schedule_sha256"] = source["schedule_sha256"]
    schedule["replay_namespace"] = schedule.pop("frozen_identity")
    schedule["protocol"] = PROTOCOL
    for entry in schedule["schedule"]:
        entry["occurrence_id"] = f"{PROTOCOL}-update-{entry['update']}-group-{entry['group']}"
    schedule["experiment_limits"] = {
        "runs": 1,
        "updates": UPDATES,
        "groups": GROUPS_PER_UPDATE * UPDATES,
        "rollouts": GROUPS_PER_UPDATE * UPDATES * ROLLOUTS_PER_GROUP,
        "additional_seeds": 0,
    }
    schedule["limits"] = [
        "Replay repeats existing contracts and creates no independent evidence.",
        "Calibration estimates do not establish a causal benefit of teacher SFT.",
    ]
    schedule["schedule_sha256"] = compact_json_sha256(schedule)
    return {"protocol": PROTOCOL, "signal_gate_passed": True, "reason": None, "schedule": schedule}


@dataclass(frozen=True)
class CalibrationRecordConfig:
    summary_path: str
    plan: RoundPlan
    source_replay: dict
    qualification_sha256: str
    output_path: str


def seal_calibration(config: CalibrationRecordConfig) -> None:
    raw = StoragePath(prefix_join(config.summary_path, "failure_summary.json")).read_bytes()
    result = post_sft_schedule(json.loads(raw), config.plan, config.source_replay)
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


def seal_selection(config: SelectionConfig) -> None:
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
    selected = scores[0] if len(scores) == 1 else selected_dose(*scores)
    promoted = selected_dose(config.parent, selected)
    write_once(
        StoragePath(prefix_join(config.output_path, "post-sft-selection.json")),
        {
            "protocol": PROTOCOL,
            "parent": asdict(config.parent),
            "sft": asdict(scores[0]),
            "sft_rl": asdict(scores[1]) if len(scores) == 2 else None,
            "selected": asdict(selected),
            "promoted": asdict(promoted),
        },
    )


def adopted(value: dict, kind: type = Artifact) -> ArtifactStep:
    return ArtifactStep.adopt(value["name"], value["version"], value["uri"], kind=kind, config=value["identity_config"])


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
    validate_replay_plan(
        source,
        completed.plan,
        **{
            key: sealed_replay[key]
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
        limit=BANK_TASKS,
        startup_attempts=3,
    )
    plan = post_sft_plan(completed.plan, model=artifact_identity(model), calibration=artifact_identity(calibration))
    decision = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-calibration-decision",
        version=version,
        artifact_type=Artifact,
        deps=(calibration, bank, model),
        build_config=lambda ctx: CalibrationRecordConfig(
            ctx.artifact_path(calibration), plan, source, config["qualification_sha256"], ctx.output_path
        ),
        run=seal_calibration,
    )
    if stage == "calibrate":
        return {"calibration": calibration, "decision": decision, "terminal": decision}
    record = json.loads(pinned_bytes(config["calibration_decision_uri"], config["calibration_decision_sha256"]))
    summary = json.loads(pinned_bytes(config["calibration_summary_uri"], record["summary_sha256"]))
    expected = post_sft_schedule(summary, plan, source)
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
        f"documents/russell-rsi-{PROTOCOL}-pinned-decision",
        version,
        str(StoragePath(config["calibration_decision_uri"]).parent),
        config={"decision_sha256": config["calibration_decision_sha256"]},
    )
    outputs = {}
    checkpoints = [("sft", model)]
    barriers = ()
    if record["signal_gate_passed"]:
        data = ArtifactStep(
            name=f"documents/russell-rsi-{PROTOCOL}-replay",
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
        trained = train_step(
            data,
            model,
            "pilot",
            version,
            retention,
            config["machine_config"],
            PROTOCOL,
            sampling_mode=SamplingMode.CALIBRATED_REPLAY,
        )

        def update_config(ctx: StepContext):
            if ctx.is_fingerprint:
                return {"trained": artifact_identity(trained), "updates": UPDATES}
            result = ctx.resolved(trained)
            return OptimizerStepConfig(UPDATES, result.global_step, result.hf_model_uri)

        updates = ArtifactStep(
            name=f"documents/russell-rsi-{PROTOCOL}-optimizer-gate",
            version=version,
            artifact_type=Artifact,
            deps=(trained,),
            build_config=update_config,
            run=require_optimizer_updates,
        )
        reload_model = evaluation_model(f"russell-rsi-{PROTOCOL}-reload", SKYRL_POLICY_LOCATION, None)
        reload = eval_step(
            reload_model,
            "mmlu-smoke",
            version=version,
            deps=(trained, updates),
            resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, reload_model),
            limit=1,
            accelerator="H100x8",
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
        )
        outputs.update({"rl": trained, "reload": reload})
        checkpoints.append(("sft-rl", trained))
        barriers = (trained, updates, reload)
    if stage == "train":
        return {**outputs, "terminal": outputs["reload"] if outputs else saved_decision}
    panel_value = json.loads(pinned_bytes(config["panel_uri"], config["panel_sha256"]))
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    parent_coding = json.loads(pinned_bytes(config["parent_coding_uri"], config["parent_coding_sha256"]))
    parent_retention = json.loads(pinned_bytes(config["parent_retention_uri"], config["parent_retention_sha256"]))
    parent = adopted(config["parent"], LevanterCheckpoint)
    panel_digest = compact_json_sha256(asdict(panel))
    rewards = parent_retention["task_rewards"]
    parent_score = evaluated_score(
        parent_coding,
        parent_retention,
        artifact_identity(parent),
        panel_digest,
        artifact_identity(retention),
        tuple(sorted(rewards)),
    )
    # Both checkpoints wait for the same completed RL export and reload when RL is eligible.
    for label, checkpoint in checkpoints:
        evaluation = evaluation_model(
            f"russell-rsi-{PROTOCOL}-{label}", export_uri if label == "sft" else SKYRL_POLICY_LOCATION, None
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
            name=f"documents/russell-rsi-{PROTOCOL}-{label}-coding",
            version=version,
            artifact_type=Artifact,
            deps=(coding, checkpoint),
            build_config=evidence_config,
            run=collect_coding_eval_evidence,
        )
        retained = development_step(
            retention, checkpoint, version, runtime, f"{PROTOCOL}-{label}-retention", limit=RETENTION_TASKS
        )
        retained = replace(retained, deps=tuple(dict.fromkeys((*retained.deps, *barriers))))
        outputs.update({f"coding-{label}": evidence, f"retention-{label}": retained})
    labels = tuple(label for label, _ in checkpoints)
    outputs["selection"] = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-selection",
        version=version,
        artifact_type=Artifact,
        deps=tuple(outputs[f"{kind}-{label}"] for label in labels for kind in ("coding", "retention")),
        build_config=lambda ctx: SelectionConfig(
            tuple(ctx.artifact_path(outputs[f"coding-{label}"]) for label in labels),
            tuple(ctx.artifact_path(outputs[f"retention-{label}"]) for label in labels),
            tuple(artifact_identity(item) for _, item in checkpoints),
            panel_digest,
            artifact_identity(retention),
            tuple(sorted(rewards)),
            parent_score,
            ctx.output_path,
        ),
        run=seal_selection,
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


if __name__ == "__main__":
    main()
