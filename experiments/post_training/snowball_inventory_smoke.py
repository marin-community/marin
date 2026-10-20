# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ten optimizer updates over the versioned Snowball curriculum inventory."""

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
class InventorySmokeConfig:
    training: SkyRLRunConfig
    pool_sha256: str


def verify_pool_runtime(launch_config_yaml: str, expected_sha256: str, output_uri: str) -> None:
    with TemporaryDirectory(prefix="snowball-launch-preflight-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(launch_config_yaml)
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "config/external/MarinSkyRL",
                "--frozen",
                "--no-dev",
                "python",
                "experiments/post_training/snowball_inventory_probe.py",
                "--launch-config",
                str(path),
                "--expected-sha256",
                expected_sha256,
                "--output-uri",
                output_uri,
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=30 * 60,
        )


def run_inventory_smoke(config: InventorySmokeConfig) -> SkyRLRun:
    report_uri = prefix_join(config.training.output.terminal_manifest_uri.rsplit("/", 1)[0], "probe/report.json")
    remote(
        verify_pool_runtime,
        resources=ResourceConfig.with_cpu(cpu=4, ram="32GB", disk="32GB", container_profile=CONTAINER_PROFILE),
    )(config.training.launch_config_yaml, config.pool_sha256, report_uri)
    return run_skyrl(config.training)


def smoke_recipe(request_window_tokens: int, max_new_tokens: int) -> dict:
    preset = replace(
        DEFAULT,
        label="inventory-smoke",
        max_steps=10,
        eval_interval=-1,
        max_staleness_steps=0,
        generation_workers=ROLE_PLAN.train_batch_size,
        request_window_tokens=request_window_tokens,
        max_new_tokens=max_new_tokens,
        batch_policy="full_batch",
    )
    config = training_config(preset)
    config["entrypoint"] = "taskcompendium"
    config["data"] = {"kind": "parquet", "shuffle": True}
    config["environment"] = {
        "task_sessions": {
            "local": {},
            "token_prefix": {
                "tokenizer_model": TOKENIZER_MODEL,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
            "remote_judge": {
                "base_url": "https://api.together.xyz/v1",
                "api_key_env": "TOGETHER_API_KEY",
            },
        }
    }
    config["trainer"]["max_ckpts_to_keep"] = 1
    config["trainer"]["seed"] = SEED
    config["trainer"]["train_batch_size"] = ROLE_PLAN.train_batch_size
    config["trainer"]["policy_mini_batch_size"] = ROLE_PLAN.policy_mini_batch_size
    config["trainer"]["project_name"] = "marin-snowball-curriculum-smoke"
    config["extra_env"]["VERIFYIT_JUDGE_MODEL"] = "openai/gpt-oss-120b"
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
@rl_build_options
def main(
    model_uri: str,
    model_version: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    request_window_tokens: int,
    max_new_tokens: int,
) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    model = ArtifactStep.adopt(
        "models/snowball-inventory-smoke-input", model_version, model_uri, kind=LevanterCheckpoint
    )
    pool = ArtifactStep.adopt("documents/snowball-inventory-runtime", pool_version, pool_uri)
    recipe = smoke_recipe(request_window_tokens, max_new_tokens)
    rl = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            runtime=SkyRLRuntime(profile=SNOWBALL_RECIPE.profile),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=TOKENIZER_MODEL,
                tokenizer_revision=TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path="train.parquet"),),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=SNOWBALL_RECIPE.num_nodes,
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

    def build_config(ctx: StepContext) -> InventorySmokeConfig:
        return InventorySmokeConfig(rl.build_config(ctx), pool_sha256)

    return replace(rl, run=run_inventory_smoke, build_config=build_config)


if __name__ == "__main__":
    main()
