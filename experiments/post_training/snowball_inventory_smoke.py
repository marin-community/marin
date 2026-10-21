# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ten optimizer updates over the versioned Snowball curriculum inventory."""

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.async_rl import DEFAULT, SNOWBALL_RECIPE, training_config

TOKENIZER_MODEL = "open-athena/Snowball-Smartie-RL-2026.10.15.32-step14"
TOKENIZER_REVISION = "6783e9743478bd25a9ce03ab3460e84d6a9814b0"
CLUSTER = "cw-us-east-02a"
CONTAINER_PROFILE = "CONTAINER_PROFILE_PRIVILEGED"
ARTIFACT_NAME = "checkpoints/snowball-inventory-smoke"
SEED = 20261020
ROLE_PLAN = replace(SNOWBALL_RECIPE.role_plan, train_batch_size=32, policy_mini_batch_size=32)


@dataclass(frozen=True)
class ModelSourceIdentity:
    repository: str
    revision: str


@dataclass(frozen=True)
class InventoryRunConfig:
    training: SkyRLRunConfig
    pool_sha256: str
    validation_sha256: str | None = None
    validation_family_counts: dict[str, int] | None = None
    model_source: ModelSourceIdentity | None = None


def inventory_training_execution() -> IrisSkyRLExecution:
    """Execution settings for the full continuation and initializer comparison."""
    return IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=32,
        memory=SNOWBALL_RECIPE.host_memory,
        disk="2TB",
        priority="interactive",
        max_retries=0,
        target_cluster=CLUSTER,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=36,
        job_timeout_seconds=24 * 60 * 60,
        container_profile=CONTAINER_PROFILE,
        wandb_entity="marin-community",
    )


def runtime_environment() -> dict[str, str]:
    """Keep the vendored runtime independent from the parent Python environment."""
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    return environment


def verify_pool_runtime(
    launch_config_yaml: str,
    expected_sha256: str,
    output_uri: str,
    validation_sha256: str | None,
    validation_family_counts: dict[str, int] | None = None,
    model_source: ModelSourceIdentity | None = None,
) -> None:
    with TemporaryDirectory(prefix="snowball-launch-preflight-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(launch_config_yaml)
        command = [
            "uv",
            "run",
            "--project",
            "vendor/skyrl",
            "--frozen",
            "--extra",
            "cpu",
            "--no-group",
            "dev",
            "python",
            "experiments/post_training/snowball_inventory_probe.py",
            "--launch-config",
            str(path),
            "--expected-sha256",
            expected_sha256,
            "--output-uri",
            output_uri,
        ]
        if validation_sha256 is not None:
            command.extend(["--expected-validation-sha256", validation_sha256])
        if validation_family_counts is not None:
            command.extend(["--validation-family-counts", json.dumps(validation_family_counts, sort_keys=True)])
        if model_source is not None:
            command.extend(["--model-repository", model_source.repository, "--model-revision", model_source.revision])
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=30 * 60,
            env=runtime_environment(),
        )


def run_inventory(config: InventoryRunConfig) -> SkyRLRun:
    report_uri = prefix_join(config.training.output.terminal_manifest_uri.rsplit("/", 1)[0], "probe/report.json")
    remote(
        verify_pool_runtime,
        resources=ResourceConfig.with_cpu(cpu=4, ram="32GB", disk="32GB", container_profile=CONTAINER_PROFILE),
    )(
        config.training.launch_config_yaml,
        config.pool_sha256,
        report_uri,
        config.validation_sha256,
        config.validation_family_counts,
        config.model_source,
    )
    return run_skyrl(config.training)


def smoke_recipe(
    request_window_tokens: int,
    max_new_tokens: int,
    *,
    role_plan: SkyRLRolePlan = ROLE_PLAN,
    max_steps: int = 10,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
) -> dict:
    preset = replace(
        DEFAULT,
        label="inventory-smoke",
        max_steps=max_steps,
        eval_interval=-1,
        max_staleness_steps=0,
        generation_workers=role_plan.train_batch_size,
        request_window_tokens=request_window_tokens,
        max_new_tokens=max_new_tokens,
        batch_policy="full_batch",
    )
    config = training_config(preset)
    config["entrypoint"] = "taskcompendium"
    config["data"] = {"kind": "parquet", "shuffle": True}
    config["environment"] = {"task_sessions": {"local": {}}}
    if (judge_base_url is None) != (judge_model is None):
        raise ValueError("Judge base URL and model must be configured together")
    if judge_base_url is not None:
        config["environment"]["task_sessions"]["remote_judge"] = {
            "base_url": judge_base_url,
            "api_key_env": "JUDGE_API_KEY",
        }
        config["extra_env"]["VERIFYIT_JUDGE_MODEL"] = judge_model
    config["trainer"]["max_ckpts_to_keep"] = 1
    config["trainer"]["seed"] = SEED
    config["trainer"]["train_batch_size"] = role_plan.train_batch_size
    config["trainer"]["policy_mini_batch_size"] = role_plan.policy_mini_batch_size
    config["generator"]["n_samples_per_prompt"] = role_plan.n_samples_per_prompt
    config["trainer"]["project_name"] = "marin-snowball-curriculum-smoke"
    config["extra_env"].update({"NCCL_NET": "Socket", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_FAMILY": "AF_INET"})
    return config


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-version", required=True)
@click.option("--pool-uri", required=True)
@click.option("--pool-version", required=True)
@click.option("--pool-sha256", required=True)
@click.option("--request-window-tokens", type=int, required=True)
@click.option("--max-new-tokens", type=int, default=1024, show_default=True)
@click.option("--judge-base-url", required=True)
@click.option("--judge-model", required=True)
@rl_build_options
def main(
    model_uri: str,
    model_version: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    request_window_tokens: int,
    max_new_tokens: int,
    judge_base_url: str,
    judge_model: str,
) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    model = ArtifactStep.adopt(
        "models/snowball-inventory-smoke-input", model_version, model_uri, kind=LevanterCheckpoint
    )
    pool = ArtifactStep.adopt("documents/snowball-inventory-runtime", pool_version, pool_uri)
    recipe = smoke_recipe(request_window_tokens, max_new_tokens, judge_base_url=judge_base_url, judge_model=judge_model)
    rl = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=TOKENIZER_MODEL,
                tokenizer_revision=TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path="train.parquet"),),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=ROLE_PLAN.num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory=SNOWBALL_RECIPE.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=12,
            job_timeout_seconds=6 * 60 * 60,
            container_profile=CONTAINER_PROFILE,
            wandb_entity="marin-community",
        ),
        export_hf=True,
    )

    def build_config(ctx: StepContext) -> InventoryRunConfig:
        return InventoryRunConfig(rl.build_config(ctx), pool_sha256)

    return replace(rl, run=run_inventory, build_config=build_config)


if __name__ == "__main__":
    main()
