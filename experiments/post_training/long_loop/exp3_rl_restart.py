# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restart Exp3 RL from the pinned adopted SFT export, with a fresh lineage."""

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import click
import draccus
import yaml
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRolePlan, SkyRLRunConfig
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.exp3_production import OUTPUT_PREFIX, TEMPORARY_PREFIX
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision
from experiments.post_training.long_loop.rl import configured_rl_step
from experiments.post_training.long_loop.spec_pipeline import runtime_execution

SOCKET_OVERRIDES = ("NCCL_NET", "NCCL_IB_DISABLE", "NCCL_SOCKET_FAMILY", "NCCL_IB_HCA")


def restart_recipe(reference: dict) -> dict:
    recipe = copy.deepcopy(reference["plan"]["rl"]["config"])
    for name in SOCKET_OVERRIDES:
        recipe["extra_env"].pop(name, None)
    trainer = recipe["trainer"]
    trainer["resume_mode"] = "none"
    trainer["ckpt_interval"] = -1
    for callback in trainer["callbacks"]:
        if callback["type"] == "checkpoint":
            callback.update(save_steps=-1, save_interval_seconds=7200, save_on_train_end=True)
    trainer["callbacks"].append(
        {"type": "wall_time_budget", "seconds": 43200, "drain_margin": 1800, "groups_per_step": 64}
    )
    trainer["rollout_buffer"]["source_routes"] = copy.deepcopy(
        reference["launch"]["config"]["skyrl"]["trainer"]["rollout_buffer"]["source_routes"]
    )
    recipe["terminal_bench"]["harbor"]["eval_override_storage_mb"] = 131072
    return recipe


@click.command(help=__doc__)
@click.option("--reference", required=True)
@click.option("--reference-sha256", required=True)
@click.option("--source-revision", required=True)
@click.option("--validation-output", required=True, type=click.Path(path_type=Path))
@rl_build_options
def main(reference: str, reference_sha256: str, source_revision: str, validation_output: Path) -> ArtifactStep:
    configure_coreweave_s3()
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    payload = StoragePath(reference).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference_sha256:
        raise ValueError("Frozen .51 restart reference differs from its pinned hash")
    baseline = json.loads(payload)
    original = baseline["launch"]["config"]
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX=TEMPORARY_PREFIX)
    name = "mochi-exp3-rl-ib"
    version = resolve_version(name, None)
    model_input = original["inputs"]["model"]
    model = ArtifactHfModel(
        ArtifactStep.adopt(f"{name}/model", "2026.11.16.51", source=model_input["uri"], kind=LevanterCheckpoint),
        relative_path="",
        tokenizer_uri=model_input["tokenizer_uri"],
        tokenizer_revision=model_input["tokenizer_revision"],
    )

    def data_sources(key: str) -> tuple[ArtifactDataSource, ...]:
        return tuple(
            ArtifactDataSource(
                ArtifactStep.adopt(f"{name}/{key}-{index}", "2026.11.16.51", source=row["uri"], kind=Artifact),
                relative_path=row["relative_path"],
            )
            for index, row in enumerate(original["inputs"][key])
        )

    rl = baseline["plan"]["rl"]
    execution = runtime_execution(rl["execution"])
    execution = replace(execution, coordinator_name=f"mochi-exp3-rl-ib-{version}")
    step = configured_rl_step(
        version=version,
        name=name,
        model=model,
        train_data=data_sources("train_data"),
        validation_data=data_sources("validation_data"),
        config=restart_recipe(baseline),
        plan=draccus.decode(SkyRLRolePlan, rl["role_plan"]),
        execution=execution,
        retained=rl["retained"],
        seed=rl["seed"],
        ray_timeouts=rl["ray_timeouts"],
    )
    original_builder = step.build_config

    def bind_names(context: StepContext) -> SkyRLRunConfig:
        launch = original_builder(context)
        document = yaml.safe_load(launch.launch_config_yaml)
        document["iris"]["job_name"] = f"mochi-exp3-rl-ib-{version}-{launch.attempt_id}"
        return replace(launch, launch_config_yaml=yaml.safe_dump(document, sort_keys=False))

    step = replace(step, build_config=bind_names)
    validation_output.mkdir(parents=True, exist_ok=True)
    validate_graph(
        step,
        validation_output,
        OUTPUT_PREFIX,
        model_config=(StoragePath(model_input["uri"]) / "config.json").read_bytes(),
    )
    return step


if __name__ == "__main__":
    main()
