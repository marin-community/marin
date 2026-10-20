# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ten optimizer updates over the versioned Snowball curriculum inventory."""

import subprocess
from dataclasses import replace
from pathlib import Path

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
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
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.async_rl import DEFAULT, SNOWBALL_RECIPE, training_config

TOKENIZER_MODEL = "open-athena/Snowball-Smartie-RL-2026.10.15.32-step14"
TOKENIZER_REVISION = "6783e9743478bd25a9ce03ab3460e84d6a9814b0"
CLUSTER = "cw-us-east-02a"
ARTIFACT_NAME = "checkpoints/snowball-inventory-smoke"
SEED = 20261020
ROLE_PLAN = replace(SNOWBALL_RECIPE.role_plan, train_batch_size=32, policy_mini_batch_size=32)


def verify_pool_runtime(pool_uri: str, expected_sha256: str, output_uri: str) -> None:
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
            "--pool-uri",
            prefix_join(pool_uri, "train.parquet"),
            "--expected-sha256",
            expected_sha256,
            "--output-uri",
            prefix_join(output_uri, "report.json"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=30 * 60,
    )


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
    preflight = apply(
        user_owned_name("documents/snowball-inventory-runtime-probe"),
        remote(
            verify_pool_runtime,
            resources=ResourceConfig.with_cpu(cpu=4, ram="32GB", disk="32GB", target_cluster=CLUSTER),
        ),
        version=resolve_version(name, None),
        pool_uri=pool,
        expected_sha256=pool_sha256,
        output_uri=OUT,
    )
    rl = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(smoke_recipe(request_window_tokens, max_new_tokens), sort_keys=False),
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
            max_retries=1,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=12,
            job_timeout_seconds=6 * 60 * 60,
            wandb_entity="marin-community",
        ),
        export_hf=True,
    )
    return replace(rl, deps=(*rl.deps, preflight))


if __name__ == "__main__":
    main()
