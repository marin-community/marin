# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the incumbent on the current bank and run one bounded GRPO trial."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle
from taskcompendium.parquet import read_task_records

from experiments.post_training.russell_rsi.bootstrap_loop import (
    PILOT_UPDATES,
    QualifiedTask,
    RoundPlan,
    qualified_bank,
    write_once,
)
from experiments.post_training.russell_rsi.calibrated_trial import calibrated_schedule, four_update_trial
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingPanel
from experiments.post_training.russell_rsi.launch import CALIBRATION_TEMPERATURE, adopted, development_step
from experiments.post_training.russell_rsi.launch_post_teacher_sft import evaluated_score
from experiments.post_training.russell_rsi.launch_rsi_continuation import (
    BASELINE_RETENTION,
    INCUMBENT_CODING,
    PARENT_CODING,
    ContinuationSelectionConfig,
    qualified_champion,
    selection_record,
    trial_evaluation_graph,
)
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256, pinned_bytes
from experiments.post_training.russell_rsi.replay import REPLAY_SEED, ReplayDatasetConfig, freeze_replay_dataset
from experiments.post_training.russell_rsi.rollout_eval import run_calibration_evaluation
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PROTOCOL = "incumbent-current-bank-r1"
TASKS = 32
FAMILIES = 26
PRIOR_TASKS = 28
TARGETS = 4


def pinned_config(config: dict, name: str) -> dict:
    return json.loads(pinned_bytes(config[f"{name}_uri"], config[f"{name}_sha256"]))


def current_bank_tasks(config: dict) -> tuple[tuple[QualifiedTask, ...], dict[str, str], tuple[QualifiedTask, ...]]:
    """Verify the frozen bank and select additions by identity in current-bank order."""
    bank = pinned_config(config, "bank_record")
    prior = pinned_config(config, "prior_bank_record")
    root = config["bank"]["uri"]
    pins = config["bank"]["identity_config"]
    if (
        config["bank_record_uri"] != prefix_join(root, "bank.json")
        or config["bank_record_sha256"] != pins["bank_sha256"]
    ):
        raise ValueError("Current bank record differs from its adopted artifact")
    manifest = json.loads(pinned_bytes(prefix_join(root, "repair-manifest.json"), pins["manifest_sha256"]))
    if manifest["status"] != "complete" or manifest["stage"] != "admit":
        raise ValueError("Current bank requires completed admission")
    if (
        manifest["files"]["bank.json"] != pins["bank_sha256"]
        or manifest["files"]["train.parquet"] != pins["train_sha256"]
    ):
        raise ValueError("Admission manifest differs from the bank or raw training rows")
    tasks = tuple(QualifiedTask(**row) for row in bank["tasks"])
    families = bank["family_by_task"]
    old = tuple(QualifiedTask(**row) for row in prior["tasks"])
    old_by_id = {task.task_id: task for task in old}
    by_id = {task.task_id: task for task in tasks}
    if (
        len(tasks) != TASKS
        or qualified_bank(tasks) != tasks
        or len(old) != PRIOR_TASKS
        or len(old_by_id) != PRIOR_TASKS
        or set(families) != set(by_id)
        or len(set(families.values())) != FAMILIES
        or not set(old_by_id) <= set(by_id)
        or any(by_id[key] != value or families[key] != prior["family_by_task"][key] for key, value in old_by_id.items())
    ):
        raise ValueError("Current bank must retain all 28 records and families unchanged")
    targets = tuple(task for task in tasks if task.task_id not in old_by_id)
    prior_families = set(prior["family_by_task"].values())
    if (
        len(targets) != TARGETS
        or len({task.contract_id for task in targets}) != TARGETS
        or any(
            task.relation not in {"independent", "new_contract"}
            or families[task.task_id] != task.contract_id
            or families[task.task_id] in prior_families
            for task in targets
        )
        or config["targeted_task_ids"] != [task.task_id for task in targets]
    ):
        raise ValueError("Targets must be the four independent additions in current-bank order")
    path = prefix_join(root, "train.parquet")
    with TemporaryDirectory(prefix="incumbent-bank-") as directory:
        local = Path(directory) / "train.parquet"
        local.write_bytes(pinned_bytes(path, pins["train_sha256"]))
        records = list(read_task_records(str(local)))
    values = [json.loads(record) for record in records]
    if len(values) != TASKS or {row["id"] for row in values} != set(by_id):
        raise ValueError("Raw training rows differ from the current bank")
    for value in values:
        task = by_id[value["id"]]
        if canonical_sha256(value) != task.task_sha256:
            raise ValueError("Raw TaskSpec content differs from admission identity")
        proof = json.loads(
            pinned_bytes(prefix_join(root, f"evidence/{task.admission_sha256}/proposal.json"), task.admission_sha256)
        )
        if proof["task_sha256"] != task.task_sha256 or proof["source_group"] != task.source_id:
            raise ValueError("Task admission evidence differs from the current bank")
    return tasks, families, targets


@dataclass(frozen=True)
class IncumbentCalibrationConfig:
    summary_path: str
    plan: RoundPlan
    families: dict[str, str]
    targeted_tasks: tuple[QualifiedTask, ...]
    qualification_sha256: str
    lineage_review_sha256: str
    output_path: str


def calibration_decision(
    *,
    summary: dict,
    summary_sha256: str,
    plan: RoundPlan,
    families: dict[str, str],
    targeted_tasks: tuple[QualifiedTask, ...],
    qualification_sha256: str,
    lineage_review_sha256: str,
) -> dict:
    return {
        **calibrated_schedule(summary, plan, families, targeted_tasks, PROTOCOL),
        "plan": asdict(plan),
        "targeted_task_ids": [task.task_id for task in targeted_tasks],
        "qualification_sha256": qualification_sha256,
        "lineage_review_sha256": lineage_review_sha256,
        "summary_sha256": summary_sha256,
    }


def seal_incumbent_calibration(config: IncumbentCalibrationConfig) -> None:
    raw = StoragePath(prefix_join(config.summary_path, "failure_summary.json")).read_bytes()
    write_once(
        StoragePath(prefix_join(config.output_path, "calibration-decision.json")),
        calibration_decision(
            summary=json.loads(raw),
            summary_sha256=hashlib.sha256(raw).hexdigest(),
            plan=config.plan,
            families=config.families,
            targeted_tasks=config.targeted_tasks,
            qualification_sha256=config.qualification_sha256,
            lineage_review_sha256=config.lineage_review_sha256,
        ),
    )


def seal_incumbent_selection(config: ContinuationSelectionConfig) -> None:
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
    write_once(
        StoragePath(prefix_join(config.output_path, "continuation-selection.json")),
        selection_record(candidate, config.incumbent, config.parent, PROTOCOL),
    )


def incumbent_bank_workflow(config: dict, stage: str) -> dict[str, ArtifactStep]:
    """Bind a new incumbent calibration and its conditional four-update trial."""
    if config["protocol"] != PROTOCOL:
        raise ValueError("Incumbent trial protocol differs")
    pinned_bytes(config["feedback_release_uri"], config["feedback_release_sha256"])
    if config["feedback_identity"] != config["feedback_release_sha256"]:
        raise ValueError("Feedback identity differs from its lineage pin")
    source = pinned_config(config, "source_config")
    for key in ("parent", "retention", "machine_config", "runtime_bundle", "panel_uri", "panel_sha256"):
        if config[key] != source[key]:
            raise ValueError(f"Incumbent trial {key} differs from the frozen source")
    tasks, families, targets = current_bank_tasks(config)
    producer = pinned_config(config, "incumbent_artifact")
    qualification = pinned_config(config, "qualification")
    if (
        qualification["source_artifact_uri"] != config["incumbent_artifact_uri"]
        or qualification["source_artifact_sha256"] != config["incumbent_artifact_sha256"]
    ):
        raise ValueError("Qualification identifies a different incumbent producer")
    export = qualified_champion(qualification, producer)
    lineage = pinned_config(config, "lineage_review")
    if (
        lineage["current_pair"]["model_identity"] != qualification["model_identity"]
        or lineage["current_pair"]["bank"]["sha256"] != config["bank_record_sha256"]
        or lineage["prior_incumbent_28_bank"]["bank_artifact"]["identity_config"]["bank_sha256"]
        != config["prior_bank_record_sha256"]
    ):
        raise ValueError("Lineage review identifies different incumbent or bank inputs")
    version = config["version"]
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        version,
        export,
        kind=LevanterCheckpoint,
        config={
            "source_identity": qualification["model_identity"],
            "source_artifact_sha256": config["incumbent_artifact_sha256"],
            "qualification_sha256": config["qualification_sha256"],
        },
    )
    bank, retention = adopted(config["bank"]), adopted(config["retention"])
    parent = adopted(config["parent"], LevanterCheckpoint)
    panel = CodingPanel.from_dict(pinned_config(config, "panel"))
    panel_digest = compact_json_sha256(asdict(panel))
    scores = []
    task_ids: tuple[str, ...] = ()
    for label, identity in (("parent", artifact_identity(parent)), ("incumbent", qualification["model_identity"])):
        coding, retained = pinned_config(config, f"{label}_coding"), pinned_config(config, f"{label}_retention")
        ids = tuple(sorted(retained["task_rewards"]))
        if task_ids and ids != task_ids:
            raise ValueError("Incumbent trial baselines use different retention contracts")
        task_ids = ids
        scores.append(evaluated_score(coding, retained, identity, panel_digest, artifact_identity(retention), ids))
    parent_score, incumbent_score = scores
    if (
        parent_score.development != PARENT_CODING
        or incumbent_score.development != INCUMBENT_CODING
        or parent_score.retention != BASELINE_RETENTION
        or incumbent_score.retention != BASELINE_RETENTION
    ):
        raise ValueError("Incumbent trial baselines differ from frozen scores")
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
        startup_attempts=3,
        require_reward_variation=False,
        limit=TASKS,
        evaluation_runner=run_calibration_evaluation,
    )
    plan = RoundPlan(
        name=PROTOCOL,
        current_checkpoint=artifact_identity(model),
        champion_checkpoint=artifact_identity(model),
        task_bank=tasks,
        bank_identity=artifact_identity(bank),
        calibration_identity=artifact_identity(calibration),
        feedback_labels=(),
        selected_tasks=tasks,
        retained_count=PRIOR_TASKS,
        fresh_count=TARGETS,
        absent_bands=(),
        development_identity=panel_digest,
        retention_identity=artifact_identity(retention),
        feedback_identity=config["feedback_identity"],
        runtime_identity=compact_json_sha256(
            {"qemu": runtime.archive_sha256, "skyrl": config["runtime_commit"], "machine": config["machine_config"]}
        ),
        seed=REPLAY_SEED,
        updates=PILOT_UPDATES,
        max_glm_responses=0,
    )
    decision = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-calibration-decision",
        version=version,
        artifact_type=Artifact,
        deps=(calibration, bank, model),
        build_config=lambda ctx: IncumbentCalibrationConfig(
            ctx.artifact_path(calibration),
            plan,
            families,
            targets,
            config["qualification_sha256"],
            config["lineage_review_sha256"],
            ctx.output_path,
        ),
        run=seal_incumbent_calibration,
    )
    if stage == "calibrate":
        return {"calibration": calibration, "decision": decision, "terminal": decision}
    record = pinned_config(config, "calibration_decision")
    raw = pinned_bytes(config["calibration_summary_uri"], record["summary_sha256"])
    expected = calibration_decision(
        summary=json.loads(raw),
        summary_sha256=record["summary_sha256"],
        plan=plan,
        families=families,
        targeted_tasks=targets,
        qualification_sha256=config["qualification_sha256"],
        lineage_review_sha256=config["lineage_review_sha256"],
    )
    if compact_json_sha256(record) != compact_json_sha256(expected):
        raise ValueError("Incumbent decision differs from its complete calibration evidence")
    saved = ArtifactStep.adopt(
        f"documents/russell-rsi-{PROTOCOL}-pinned-decision",
        version,
        str(StoragePath(config["calibration_decision_uri"]).parent),
        config={"decision_sha256": config["calibration_decision_sha256"]},
    )
    if not record["signal_gate_passed"]:
        return {"terminal": saved}
    data = ArtifactStep(
        name=f"documents/russell-rsi-{PROTOCOL}-replay",
        version=version,
        artifact_type=Artifact,
        deps=(bank, model, saved),
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
    if stage == "train":
        return {**trial, "terminal": trial["reload"]}
    return trial_evaluation_graph(
        trained=trial["rl"],
        updates=trial["updates"],
        reload=trial["reload"],
        panel=panel,
        retention=retention,
        incumbent_score=incumbent_score,
        parent_score=parent_score,
        runtime=runtime,
        protocol=PROTOCOL,
        version=version,
        task_ids=task_ids,
        selection_writer=seal_incumbent_selection,
    )


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--stage", type=click.Choice(["calibrate", "train", "evaluate"]), required=True)
@rl_build_options
def main(config_uri: str, config_sha256: str, stage: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if resolve_version(PROTOCOL, None) != config["version"] or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise click.UsageError("Incumbent config version or installed runtime pin differs")
    return [incumbent_bank_workflow(config, stage)["terminal"]]


if __name__ == "__main__":
    main()
