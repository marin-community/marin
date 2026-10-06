# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare one selected Russell checkpoint with its parent on the sealed acceptance panel."""

import json
from dataclasses import asdict, dataclass, replace

import click
from marin.execution.artifact import Artifact, artifact_record_identity
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import RemoteCallable
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.bootstrap_loop import checkpoint_score, promotes, write_once
from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION, adopted, development_step
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.rollout_eval import (
    SUPPLEMENTARY_TASKS,
    SupplementaryEvaluationConfig,
    run_supplementary_evaluation,
)


@dataclass(frozen=True)
class SelectedCheckpoint:
    export_uri: str
    source_identity: str
    record_sha256: str
    decision_sha256: str


def selected_checkpoint(config: dict) -> SelectedCheckpoint:
    """Require an exported checkpoint with the fixed Snowball tokenizer."""
    decision = json.loads(pinned_bytes(config["decision_uri"], config["decision_sha256"]))
    parent = checkpoint_score(decision["parent"])
    selected = checkpoint_score(decision["selected"])
    if parent.checkpoint_identity != config["parent"]["artifact_identity"] or not promotes(selected, parent):
        raise ValueError("The selected checkpoint does not pass the unchanged parent promotion gate")
    record = json.loads(pinned_bytes(config["checkpoint_record_uri"], config["checkpoint_record_sha256"]))
    identity = artifact_record_identity(record)
    if identity != selected.checkpoint_identity:
        raise ValueError("The export record does not identify the selected checkpoint")
    match record["result_type"]:
        case "marin.rl.skyrl.SkyRLRun":
            export = record["result"]["hf_model_uri"]
            if (record["result"]["tokenizer_uri"], record["result"]["tokenizer_revision"]) != (MODEL, MODEL_REVISION):
                raise ValueError("Acceptance requires the unchanged Snowball tokenizer")
        case "marin.training.training.LevanterCheckpoint":
            export = record["source"]
        case _:
            raise ValueError("Unsupported selected checkpoint artifact type")
    if not export:
        raise ValueError("Acceptance requires a completed HF export")
    return SelectedCheckpoint(export, identity, config["checkpoint_record_sha256"], config["decision_sha256"])


@dataclass(frozen=True)
class AcceptanceResultConfig:
    paths: tuple[str, str]
    model_identities: tuple[str, str]
    panel_identity: str
    task_ids: tuple[str, ...]
    selected_source: SelectedCheckpoint
    output_path: str


def seal_acceptance_result(config: AcceptanceResultConfig) -> None:
    """Report all matched outcomes without another checkpoint choice or model attempt."""
    outcomes = []
    for path, identity in zip(config.paths, config.model_identities, strict=True):
        summary = json.loads(StoragePath(prefix_join(path, "failure_summary.json")).read_text())
        rewards = summary["task_rewards"]
        if (
            summary["model_identity"] != identity
            or summary["tasks_identity"] != config.panel_identity
            or summary["count"] != SUPPLEMENTARY_TASKS
            or summary["samples_per_task"] != 1
            or set(rewards) != set(config.task_ids)
            or any(len(group) != 1 or group[0] not in (0, 1) for group in rewards.values())
        ):
            raise ValueError("Acceptance requires the same four tasks and one valid grade per checkpoint")
        outcomes.append({task: rewards[task][0] for task in config.task_ids})
    parent, selected = outcomes
    gained = [task for task in config.task_ids if selected[task] > parent[task]]
    lost = [task for task in config.task_ids if selected[task] < parent[task]]
    write_once(
        StoragePath(config.output_path) / "acceptance-result.json",
        {
            "selected_source": asdict(config.selected_source),
            "model_identities": config.model_identities,
            "panel_identity": config.panel_identity,
            "parent": parent,
            "selected": selected,
            "gained": gained,
            "lost": lost,
            "paired_net_gain": len(gained) - len(lost),
            "acceptance_passed": len(gained) > len(lost),
            "scope": "Four admitted Marin tasks. This panel does not measure unseen-repository generalization.",
        },
    )


def supplementary_workflow(config: dict, selected: SelectedCheckpoint) -> ArtifactStep:
    """Build the matched pair with one durable request journal for each checkpoint."""
    version = config["evaluation_version"]
    panel_spec = config["panel"]
    panel = adopted(panel_spec)
    parent_spec = config["parent"]
    parent = adopted(parent_spec, LevanterCheckpoint)
    candidate = ArtifactStep.adopt(
        "checkpoints/russell-rsi-acceptance-candidate",
        version,
        selected.export_uri,
        kind=LevanterCheckpoint,
        config=asdict(selected),
    )
    if (
        artifact_identity(parent) != parent_spec["artifact_identity"]
        or artifact_identity(panel) != panel_spec["artifact_identity"]
    ):
        raise ValueError("Acceptance parent or panel identity differs from the sealed comparison")
    model_identities = (artifact_identity(parent), artifact_identity(candidate))
    evaluations = []
    for index, (label, model) in enumerate(
        (("supplementary-parent-v1", parent), ("supplementary-selected-v1", candidate))
    ):
        evaluation = development_step(
            panel,
            model,
            version,
            RuntimeBundle(**config["runtime_bundle"]),
            label,
            relative_path="supplementary.parquet",
            samples_per_task=1,
            temperature=0.0,
            require_reward_variation=False,
            limit=SUPPLEMENTARY_TASKS,
            startup_attempts=3,
        )
        original_config = evaluation.build_config
        if not isinstance(evaluation.run, RemoteCallable):
            raise TypeError("Supplementary evaluation requires the development remote worker")

        def build_config(ctx: StepContext, original=original_config, role=index) -> SupplementaryEvaluationConfig:
            return SupplementaryEvaluationConfig(
                original(ctx),
                config["journal_path"],
                role,
                model_identities,
                prefix_join(ctx.artifact_path(panel), "panel-manifest.json"),
                panel_spec["identity_config"]["manifest_sha256"],
            )

        evaluations.append(
            replace(
                evaluation,
                name=f"evals/russell-rsi-{label}-acceptance",
                build_config=build_config,
                run=replace(evaluation.run, fn=run_supplementary_evaluation),
            )
        )
    return ArtifactStep(
        name="documents/russell-rsi-supplementary-acceptance",
        version=version,
        artifact_type=Artifact,
        deps=tuple(evaluations),
        build_config=lambda ctx: AcceptanceResultConfig(
            (ctx.artifact_path(evaluations[0]), ctx.artifact_path(evaluations[1])),
            model_identities,
            artifact_identity(panel),
            tuple(config["task_ids"]),
            selected,
            ctx.output_path,
        ),
        run=seal_acceptance_result,
    )


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str) -> list[ArtifactStep]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if resolve_version("russell-rsi-supplementary", None) != config["evaluation_version"]:
        raise click.UsageError("Acceptance config and artifact version differ")
    return [supplementary_workflow(config, selected_checkpoint(config))]


if __name__ == "__main__":
    main()
