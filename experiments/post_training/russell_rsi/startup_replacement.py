# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue one calibration replacement for a task that stopped before model inference."""

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.launch import adopted, development_step
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256, pinned_bytes
from experiments.post_training.russell_rsi.rollout_eval import DevelopmentEvaluationConfig, run_development_evaluation


@dataclass(frozen=True)
class StartupReplacementConfig:
    evaluation: DevelopmentEvaluationConfig
    original_traces_uri: str
    original_traces_sha256: str
    original_line_index: int
    original_line_sha256: str
    original_task_sha256: str
    tasks_sha256: str
    runtime_module_hashes: dict[str, str]


def validate_startup_replacement(config: StartupReplacementConfig) -> dict:
    """Validate the original interrupted attempt and the exact replacement task."""
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415

    lines = pinned_bytes(config.original_traces_uri, config.original_traces_sha256).splitlines(keepends=True)
    line = lines[config.original_line_index]
    if hashlib.sha256(line).hexdigest() != config.original_line_sha256:
        raise ValueError("Replacement identifies a different original attempt")
    original = json.loads(line)
    if (
        original["interrupted_operation"] != "start"
        or original["steps"]
        or original["response_token_ids"]
        or original["grade"]["reward"] is not None
    ):
        raise ValueError("Only an interrupted startup without model output permits a new candidate")
    pinned_bytes(config.evaluation.tasks_path, config.tasks_sha256)
    tasks = list(read_tasks(config.evaluation.tasks_path))
    if len(tasks) != 1 or tasks[0].id != original["task_id"]:
        raise ValueError("Replacement must contain exactly the original task")
    task_sha256 = canonical_sha256(tasks[0].model_dump(mode="json"))
    if task_sha256 != config.original_task_sha256:
        raise ValueError("Replacement changed the frozen task")
    if config.evaluation.samples_per_task != 1 or config.evaluation.limit != 1:
        raise ValueError("Replacement permits one rollout")
    return original


def issue_startup_replacement(config: StartupReplacementConfig) -> None:
    """Validate and record the single issuance before model startup."""
    validate_startup_replacement(config)
    for module_name, expected in config.runtime_module_hashes.items():
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Replacement runtime differs from the original: {module_name}")
    issuance = StoragePath(prefix_join(config.evaluation.output_path, "replacement-issued.json"))
    if issuance.exists():
        raise ValueError("The single replacement was already issued; retain its existing result")
    issuance.write_text(json.dumps(asdict(config), sort_keys=True) + "\n")


def run_startup_replacement(config: StartupReplacementConfig) -> None:
    """Consume the single issuance before model startup and preserve an incomplete result."""
    issue_startup_replacement(config)
    run_development_evaluation(config.evaluation)
    summary = json.loads(StoragePath(prefix_join(config.evaluation.output_path, "failure_summary.json")).read_text())
    rewards = [reward for group in summary["task_rewards"].values() for reward in group]
    if len(rewards) != 1:
        raise ValueError("The single replacement has no grade; do not issue another candidate")


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str) -> ArtifactStep[Artifact]:
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    version = resolve_version("russell-rsi-startup-replacement", None)
    if version != config["version"]:
        raise click.UsageError("The replacement config and artifact versions differ")
    data = ArtifactStep.adopt(
        "documents/russell-rsi-startup-replacement-input",
        version,
        config["data_uri"],
        config={"tasks_sha256": config["tasks_sha256"], "original_line_sha256": config["original_line_sha256"]},
    )
    parent = config["parent"]
    model = adopted(parent, LevanterCheckpoint)
    evaluation = development_step(
        data,
        model,
        version,
        RuntimeBundle(**config["runtime_bundle"]),
        "startup-replacement",
        relative_path="train.parquet",
        samples_per_task=config["sampling"]["samples_per_task"],
        temperature=config["sampling"]["temperature"],
        limit=1,
    )

    def build_config(ctx: StepContext) -> StartupReplacementConfig:
        assert evaluation.build_config is not None
        return StartupReplacementConfig(
            evaluation=evaluation.build_config(ctx),
            original_traces_uri=config["original_traces_uri"],
            original_traces_sha256=config["original_traces_sha256"],
            original_line_index=config["original_line_index"],
            original_line_sha256=config["original_line_sha256"],
            original_task_sha256=config["task_sha256"],
            tasks_sha256=config["tasks_sha256"],
            runtime_module_hashes=config["runtime_module_hashes"],
        )

    # The Iris root owns the eight GPUs and disables failure and preemption retries.
    return replace(evaluation, build_config=build_config, run=run_startup_replacement)


if __name__ == "__main__":
    main()
