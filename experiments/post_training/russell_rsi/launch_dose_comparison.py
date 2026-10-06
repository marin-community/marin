# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare checkpoints four and eight in one separate bounded dose experiment."""

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
import yaml
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IRIS_HUB_CLUSTER_CONFIG, IrisSkyRLExecution, SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import (
    ATTEMPTS_PER_TASK,
    PILOT_UPDATES,
    CheckpointScore,
    Measurement,
    RoundPlan,
    StopReason,
    promotes,
    restored_round,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    CodingPanel,
    collect_coding_eval_evidence,
)
from experiments.post_training.russell_rsi.dose_qualification import qualified_dose_source
from experiments.post_training.russell_rsi.launch import (
    CLUSTER,
    OptimizerStepConfig,
    SamplingMode,
    Scale,
    adopted,
    development_step,
    evaluation_model,
    require_optimizer_updates,
    train_step,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.replay import (
    GROUPS_PER_UPDATE,
    REPLAY_ROWS,
    REPLAY_SEED,
    ReplayDatasetConfig,
    freeze_sampled_replay_dataset,
    sampled_replay_plan,
    validate_replay_plan,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model

DOSE_UPDATES = 8
DOSE_ROWS = DOSE_UPDATES * GROUPS_PER_UPDATE


def dose_replay_plan(source: dict, plan: RoundPlan) -> dict:
    """Extend the sealed family sampler without changing its first sixty-four rows."""
    identities = {
        key: source[key]
        for key in (
            "pilot_number",
            "bank_identity",
            "calibration_identity",
            "frozen_identity",
            "parent_identity",
            "model_identity",
        )
    }
    families = source["family_by_task"]
    validate_replay_plan(source, plan, **identities, family_by_task=families)
    if not source["signal_gate_passed"]:
        raise ValueError("Dose eligibility requires the existing calibration signal gate")
    measurements = tuple(
        Measurement(
            source["model_identity"],
            task.task_sha256,
            (1.0,) * source["successes_by_task"][task.task_id]
            + (0.0,) * (ATTEMPTS_PER_TASK - source["successes_by_task"][task.task_id]),
        )
        for task in plan.task_bank
    )
    by_id = {task.task_id: task for task in plan.task_bank}
    result = sampled_replay_plan(
        plan,
        measurements,
        tuple(by_id[task_id] for task_id in source["sampling_spec"]["targeted_task_ids"]),
        **identities,
        family_by_task=families,
        updates=DOSE_UPDATES,
        seed=REPLAY_SEED,
    )
    if result["schedule"][:REPLAY_ROWS] != source["schedule"]:
        raise ValueError("Dose schedule differs from the frozen four-update prefix")
    result.pop("schedule_sha256")
    result["protocol"] = "separate-eight-update-dose-v1"
    result["source_schedule_sha256"] = source["schedule_sha256"]
    result["experiment_limits"] = {"runs": 1, "updates": DOSE_UPDATES, "additional_seeds": 0}
    result["schedule_sha256"] = compact_json_sha256(result)
    return result


def freeze_dose_dataset(config: ReplayDatasetConfig) -> None:
    freeze_sampled_replay_dataset(config, rows_required=DOSE_ROWS)


def selected_dose(four: CheckpointScore, eight: CheckpointScore) -> CheckpointScore:
    """Select eight only with retained scores and a strict coding improvement."""
    if promotes(eight, four):
        return eight
    return four


@dataclass(frozen=True)
class SavedExportConfig:
    trained: SkyRLRun
    resolved_launch_uri: str
    output_path: str


def export_saved_four(config: SavedExportConfig) -> SkyRLRun:
    """Export only the immutable saved step-four request with the pinned launcher."""
    trained = config.trained
    require_optimizer_updates(OptimizerStepConfig(DOSE_UPDATES, trained.global_step, trained.hf_model_uri))
    checkpoint = prefix_join(trained.checkpoint_root, "global_step_4")
    request_uri = prefix_join(checkpoint, "hf_export_request.json")
    request = json.loads(StoragePath(request_uri).read_text())
    if request["step"] != 4 or request["checkpoint_path"] != checkpoint:
        raise ValueError("Saved export request does not identify dose checkpoint four")
    submission_args = (
        ["--target-cluster", CLUSTER, "--parent-cluster-config", str(Path(IRIS_HUB_CLUSTER_CONFIG).resolve(strict=True))]
        if get_job_info() is None
        else []
    )
    with tempfile.TemporaryDirectory() as directory:
        launch = Path(directory) / "resolved-launch.yaml"
        resolved = yaml.safe_load(StoragePath(config.resolved_launch_uri).read_text())
        launch.write_text(yaml.safe_dump(resolved["config"], sort_keys=False))
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--prerelease=allow",
                "--python",
                "3.12",
                "--with",
                MARIN_SKYRL.requirement(),
                "python",
                "-m",
                "cloud.iris.export_hf_checkpoint",
                "--request",
                checkpoint,
                "--launch-config",
                str(launch),
                "--cluster",
                CLUSTER,
                "--cluster-config",
                str(Path(f"lib/iris/config/{CLUSTER}.yaml").resolve(strict=True)),
                *submission_args,
                "--gpu-variant",
                "H100",
                "--allocation-gpus-per-node",
                "8",
                "--cpu",
                "32",
                "--memory",
                "512GB",
                "--disk",
                "2TB",
                "--priority",
                "interactive",
            ],
            check=True,
        )
    complete = json.loads(StoragePath(request_uri).read_text())
    if (
        complete["status"] != "complete"
        or complete["last_exit_code"] != 0
        or any(complete[key] != request[key] for key in ("step", "checkpoint_path", "export_path"))
    ):
        raise ValueError("Checkpoint four export did not complete")
    policy_uri = subprocess.check_output(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--prerelease=allow",
            "--python",
            "3.12",
            "--with",
            MARIN_SKYRL.requirement(),
            "python",
            "-c",
            "import sys; from marinskyrl.checkpoint_paths import policy_export_path; "
            "print(policy_export_path(sys.argv[1], int(sys.argv[2])))",
            complete["export_path"],
            "4",
        ],
        text=True,
    ).strip()
    if policy_uri == trained.hf_model_uri:
        raise ValueError("Dose checkpoint exports must have distinct policy URIs")
    evidence_uri = prefix_join(config.output_path, "saved-export.json")
    StoragePath(evidence_uri).write_text(
        json.dumps(
            {
                "step": 4,
                "source_terminal": trained.terminal_manifest_uri,
                "request_uri": request_uri,
                "request_sha256": compact_json_sha256(complete),
                "runtime_commit": MARIN_SKYRL.commit,
                "policy_uri": policy_uri,
                "status": "complete",
            },
            sort_keys=True,
        )
        + "\n"
    )
    return SkyRLRun(
        **{
            **trained.model_dump(),
            "hf_model_uri": policy_uri,
            "global_step": 4,
            "path": evidence_uri,
            "terminal_manifest_uri": evidence_uri,
        }
    )


@dataclass(frozen=True)
class DoseSelectionConfig:
    coding_paths: tuple[str, str]
    retention_paths: tuple[str, str]
    model_identities: tuple[str, str]
    retention_identity: str
    expected_retention_task_ids: tuple[str, ...]
    parent_score: CheckpointScore
    output_path: str


def seal_dose_selection(config: DoseSelectionConfig) -> None:
    """Seal checkpoint choice without applying the separate champion gate."""
    scores = []
    task_ids = None
    for coding_path, retention_path, identity in zip(
        config.coding_paths, config.retention_paths, config.model_identities, strict=True
    ):
        coding = json.loads(StoragePath(prefix_join(coding_path, "coding-evidence.json")).read_text())
        retained = json.loads(StoragePath(prefix_join(retention_path, "failure_summary.json")).read_text())
        rewards = retained["task_rewards"]
        if (
            coding["model_identity"] != identity
            or retained["model_identity"] != identity
            or retained["tasks_identity"] != config.retention_identity
            or not rewards
            or set(rewards) != set(config.expected_retention_task_ids)
            or retained["count"] != len(config.expected_retention_task_ids)
            or any(len(group) != 1 or group[0] not in (0, 1) for group in rewards.values())
            or (task_ids is not None and set(rewards) != task_ids)
        ):
            raise ValueError("Dose comparison does not identify the same complete evaluation panels")
        task_ids = set(rewards)
        scores.append(
            CheckpointScore(
                identity,
                tuple(coding["scores"][suite] for suite in ("humanevalplus", "mbppplus")),
                sum(group[0] for group in rewards.values()) / len(rewards),
            )
        )
    selected = selected_dose(*scores)
    StoragePath(prefix_join(config.output_path, "dose-selection.json")).write_text(
        json.dumps(
            {
                "parent": asdict(config.parent_score),
                "four": asdict(scores[0]),
                "eight": asdict(scores[1]),
                "selected": asdict(selected),
                "promotion": "requires_separate_unchanged_champion_gate",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def require_dose_source_launch(
    source: dict, expected: dict, execution: IrisSkyRLExecution, parent_identity: str
) -> None:
    """Require the recorded request to match the existing four-update recipe."""
    allocation = {
        **expected["iris"]["allocation"],
        "cpu": execution.cpu,
        "memory": execution.memory,
        "disk": execution.disk,
    }
    if (
        source["skyrl"] != expected["skyrl"]
        or source["runtime"] != expected["runtime"]
        or source["run"]["seed"] != expected["run"]["seed"]
        or source["iris"]["allocation"] != allocation
        or source["iris"]["cluster"] != execution.cluster
        or source["inputs"]["model"]["identity"] != parent_identity
    ):
        raise ValueError("Qualified four-update launch differs from the canonical dose source recipe")


def require_duration_only_launch(four: dict, eight: dict) -> None:
    """Require the dose launch to differ only in duration and checkpoint cadence."""
    source = json.loads(json.dumps(four["skyrl"]))
    dose = json.loads(json.dumps(eight["skyrl"]))
    if (
        source["trainer"]["max_steps"] != PILOT_UPDATES
        or dose["trainer"]["max_steps"] != DOSE_UPDATES
        or dose["trainer"]["ckpt_interval"] != PILOT_UPDATES
        or dose["trainer"]["hf_save_interval"] != PILOT_UPDATES
        or dose["trainer"]["eval_interval"] != DOSE_UPDATES
    ):
        raise ValueError("Dose launch does not identify the declared eight-update checkpoint cadence")
    for trainer in (source["trainer"], dose["trainer"]):
        for field in ("max_steps", "ckpt_interval", "hf_save_interval", "eval_interval"):
            trainer.pop(field, None)
    source_iris = {key: value for key, value in four["iris"].items() if key != "job_name"}
    dose_iris = {key: value for key, value in eight["iris"].items() if key != "job_name"}
    if (
        source != dose
        or four["runtime"] != eight["runtime"]
        or source_iris != dose_iris
        or four["inputs"]["model"] != eight["inputs"]["model"]
        or four["run"]["seed"] != eight["run"]["seed"]
    ):
        raise ValueError("Eight-update dose changes settings beyond its declared duration and checkpoint cadence")


@dataclass(frozen=True)
class SourceQualificationConfig:
    evidence: dict
    expected_launch: dict
    execution: IrisSkyRLExecution
    parent_identity: str
    expected_export_uri: str
    expected_reload_identity: str
    expected_replay_sha256: str
    output_path: str


def record_source_qualification(config: SourceQualificationConfig) -> None:
    """Revalidate completed source evidence before allocating the duration-only dose."""
    source = qualified_dose_source(config.evidence)
    require_dose_source_launch(source.requested, config.expected_launch, config.execution, config.parent_identity)
    if (
        source.export_uri != config.expected_export_uri
        or source.reload_identity != config.expected_reload_identity
        or source.source_replay["schedule_sha256"] != config.expected_replay_sha256
    ):
        raise ValueError("Dose qualification differs from the frozen source experiment")
    StoragePath(prefix_join(config.output_path, "source-qualification.json")).write_text(
        json.dumps(
            {
                "mode": "reused",
                "purpose": "duration_only_eight_update_dose",
                "source_rl": source.rl_identity,
                "source_reload": source.reload_identity,
                "export_uri": source.export_uri,
                "evidence_sha256": source.evidence_sha256,
                "requested_compatibility_sha256": compact_json_sha256(config.expected_launch["skyrl"]),
                "runtime_commit": MARIN_SKYRL.commit,
                "source_replay_sha256": config.expected_replay_sha256,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def dose_workflow(config: dict, plan: RoundPlan, schedule: dict, panel: CodingPanel) -> dict[str, ArtifactStep]:
    """Bind training and each evaluation to both completed checkpoint exports."""
    version = config["version"]

    parent = adopted(config["parent"], LevanterCheckpoint)
    bank = adopted(config["bank"])
    retention = adopted(config["retention"])
    if artifact_identity(parent) != schedule["parent_identity"] or artifact_identity(bank) != schedule["bank_identity"]:
        raise ValueError("Dose parent or bank differs from the sealed replay source")

    reference_four = train_step(
        bank, parent, "pilot", version, retention, config["machine_config"], sampling_mode=SamplingMode.CALIBRATED_REPLAY
    )
    expected_launch = yaml.safe_load(json.loads(reference_four.fingerprint_payload())["launch_config_yaml"])
    execution = next(value for value in reference_four.runtime_args.values() if isinstance(value, IrisSkyRLExecution))

    def eligibility(ctx: StepContext) -> SourceQualificationConfig:
        return SourceQualificationConfig(
            config["qualification"],
            expected_launch,
            execution,
            artifact_identity(parent),
            config["qualified_four_export_uri"],
            config["source_reload_identity"],
            schedule["source_schedule_sha256"],
            ctx.output_path,
        )

    gate = ArtifactStep(
        name="documents/russell-rsi-dose-eligibility",
        version=version,
        artifact_type=Artifact,
        build_config=eligibility,
        run=record_source_qualification,
    )
    data = ArtifactStep(
        name="documents/russell-rsi-dose-replay",
        version=version,
        artifact_type=Artifact,
        deps=(bank, gate),
        build_config=lambda ctx: ReplayDatasetConfig(ctx.artifact_path(bank), ctx.output_path, plan.task_bank, schedule),
        run=remote(
            freeze_dose_dataset,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
            pip_packages=["./lib/taskcompendium"],
        ),
    )
    trained = train_step(
        data,
        parent,
        "dose",
        version,
        retention,
        config["machine_config"],
        "separate-eight-update",
        sampling_mode=SamplingMode.CALIBRATED_REPLAY,
        bounded_scale=Scale(DOSE_UPDATES, 7),
        checkpoint_interval=4,
    )
    dose_launch = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    require_duration_only_launch(expected_launch, dose_launch)
    four = ArtifactStep(
        name="checkpoints/russell-rsi-dose-four-export",
        version=version,
        artifact_type=SkyRLRun,
        deps=(trained,),
        build_config=lambda ctx: (
            {"producer": artifact_identity(trained), "step": 4}
            if ctx.is_fingerprint
            else SavedExportConfig(
                ctx.resolved(trained), prefix_join(ctx.artifact_path(trained), "resolved-launch.yaml"), ctx.output_path
            )
        ),
        run=export_saved_four,
    )
    outputs: dict[str, ArtifactStep] = {"rl": trained, "export-four": four}
    for update, checkpoint in ((4, four), (8, trained)):
        label = f"dose-update-{update}"
        model = evaluation_model(f"russell-rsi-{label}", SKYRL_POLICY_LOCATION, None)
        reload = eval_step(
            model,
            "mmlu-smoke",
            version=version,
            deps=(trained, four),
            resolve_model=lambda ctx, source=checkpoint, selected=model: resolve_skyrl_model(ctx, source, selected),
            limit=1,
            accelerator="H100x8",
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
        )
        coding = eval_step(
            model,
            "humanevalplus,mbppplus",
            version=version,
            deps=(trained, four, reload),
            resolve_model=lambda ctx, source=checkpoint, selected=model: resolve_skyrl_model(ctx, source, selected),
            limit=32,
            accelerator="H100x8",
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
        )

        def evidence_config(ctx: StepContext, evaluation=coding, source=checkpoint):
            if ctx.is_fingerprint:
                return {"evaluation": artifact_identity(evaluation), "model": artifact_identity(source), "panel": panel}
            result = ctx.resolved(evaluation)
            return CodingEvidenceConfig(
                result.records_prefix,
                result.run_ids,
                result.results_paths,
                artifact_identity(source),
                panel,
                ctx.output_path,
            )

        evidence = ArtifactStep(
            name=f"documents/russell-rsi-{label}-coding-evidence",
            version=version,
            artifact_type=Artifact,
            deps=(coding, checkpoint),
            build_config=evidence_config,
            run=collect_coding_eval_evidence,
        )
        retained = development_step(
            retention,
            checkpoint,
            version,
            RuntimeBundle(**config["runtime_bundle"]),
            f"{label}-retention",
            limit=config["retention_count"],
        )
        retained = replace(retained, deps=tuple(dict.fromkeys((*retained.deps, trained, four, reload))))
        outputs.update({f"reload-{update}": reload, f"coding-{update}": evidence, f"retention-{update}": retained})
    selection_dependencies = tuple(outputs[key] for key in ("coding-4", "retention-4", "coding-8", "retention-8"))
    outputs["selection"] = ArtifactStep(
        name="documents/russell-rsi-dose-selection",
        version=version,
        artifact_type=Artifact,
        deps=(*selection_dependencies, four, trained, retention),
        build_config=lambda ctx: DoseSelectionConfig(
            tuple(ctx.artifact_path(outputs[f"coding-{step}"]) for step in (4, 8)),
            tuple(ctx.artifact_path(outputs[f"retention-{step}"]) for step in (4, 8)),
            (artifact_identity(four), artifact_identity(trained)),
            artifact_identity(retention),
            tuple(config["retention_task_ids"]),
            CheckpointScore(**config["parent_score"]),
            ctx.output_path,
        ),
        run=seal_dose_selection,
    )
    return outputs


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    version = resolve_version("russell-rsi-dose-comparison", None)
    if version != config["version"]:
        raise click.UsageError("Pinned dose config and artifact version differ")
    if config["runtime_commit"] != MARIN_SKYRL.commit or config["unresolved_correctness_failures"]:
        raise click.UsageError("Dose requires the pinned runtime and resolved correctness evidence")
    completed = restored_round(json.loads(pinned_bytes(config["round_uri"], config["round_sha256"])))
    if completed.state.stop_reason is not StopReason.NO_IMPROVEMENT:
        raise click.UsageError("Dose requires the sealed existing two-no-gain stop")
    if completed.result.optimizer_steps != 4:
        raise click.UsageError("Dose requires a completed exact four-update trial")
    source = json.loads(pinned_bytes(config["replay_uri"], config["replay_sha256"]))
    schedule = dose_replay_plan(source, completed.plan)
    panel_value = json.loads(pinned_bytes(config["panel_uri"], config["panel_sha256"]))
    panel = CodingPanel.from_dict(panel_value)
    source_loop = json.loads(pinned_bytes(config["source_loop_uri"], config["source_loop_sha256"]))
    for key in (
        "parent",
        "retention",
        "runtime_bundle",
        "machine_config",
        "panel_uri",
        "panel_sha256",
        "parent_retention_evidence_uri",
        "parent_retention_evidence_sha256",
    ):
        if config[key] != source_loop[key]:
            raise click.UsageError(f"Dose {key} differs from the qualified source loop")
    config["parent_score"] = asdict(completed.state.parent)
    config["qualified_four_export_uri"] = completed.result.checkpoint_uri
    config["source_reload_identity"] = completed.result.reload_identity
    # The entry check also applies when the executor reuses a successful qualification artifact.
    qualified = qualified_dose_source(config["qualification"])
    if (
        qualified.reload_identity != completed.result.reload_identity
        or qualified.export_uri != completed.result.checkpoint_uri
        or qualified.source_replay != source
    ):
        raise click.UsageError("Dose qualification differs from the sealed source trial")
    retained = json.loads(
        pinned_bytes(config["parent_retention_evidence_uri"], config["parent_retention_evidence_sha256"])
    )
    config["retention_task_ids"] = sorted(retained["task_rewards"])
    config["retention_count"] = retained["count"]
    if len(config["retention_task_ids"]) != config["retention_count"]:
        raise click.UsageError("Parent retention evidence is incomplete")
    outputs = dose_workflow(config, completed.plan, schedule, panel)
    return [outputs["selection"]]


if __name__ == "__main__":
    main()
