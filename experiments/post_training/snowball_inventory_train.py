# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball on the full curriculum with a conservative Smartie-derived recipe."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
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

from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.snowball_inventory_smoke import (
    CLUSTER,
    CONTAINER_PROFILE,
    SEED,
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    InventoryRunConfig,
    run_inventory,
    smoke_recipe,
)

ARTIFACT_NAME = "checkpoints/snowball-inventory-train"
ROLE_PLAN = SNOWBALL_RECIPE.role_plan
CHECKPOINT_INTERVAL = 25


def training_recipe(max_steps: int, *, request_window_tokens: int = 20_480, max_new_tokens: int = 4_096) -> dict:
    recipe = smoke_recipe(request_window_tokens, max_new_tokens, role_plan=ROLE_PLAN, max_steps=max_steps)
    trainer = recipe["trainer"]
    trainer["project_name"] = "marin-snowball-curriculum"
    trainer["ckpt_interval"] = CHECKPOINT_INTERVAL
    trainer["hf_save_interval"] = max_steps
    trainer["max_ckpts_to_keep"] = 2
    trainer["eval_interval"] = CHECKPOINT_INTERVAL
    trainer["eval_before_train"] = True
    trainer["eval_batch_size"] = 243
    trainer["algorithm"].update(
        advantage_estimator="rloo_n",
        group_advantage_min_size=2,
        advantage_batch_normalize=False,
        rloo_n_filter_zero_reward_groups=True,
        use_entropy_loss=False,
    )
    trainer["policy"]["optimizer_config"].update(
        lr=1e-6,
        adam_betas=[0.9, 0.999],
        weight_decay=0.0,
        max_grad_norm=0.5,
        scheduler="constant_with_warmup",
        num_warmup_steps=0,
    )
    return recipe


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-version", required=True)
@click.option("--pool-uri", required=True)
@click.option("--pool-version", required=True)
@click.option("--pool-sha256", required=True)
@click.option("--validation-uri", required=True)
@click.option("--validation-sha256", required=True)
@click.option("--max-steps", type=click.IntRange(min=25), required=True)
@rl_build_options
def main(
    model_uri: str,
    model_version: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    validation_uri: str,
    validation_sha256: str,
    max_steps: int,
) -> ArtifactStep[SkyRLRun]:
    if max_steps % CHECKPOINT_INTERVAL:
        raise click.BadParameter("max-steps must be a multiple of the checkpoint interval (25)")
    name = user_owned_name(ARTIFACT_NAME)
    model = ArtifactStep.adopt(
        "models/snowball-inventory-smoke-input", model_version, model_uri, kind=LevanterCheckpoint
    )
    pool = ArtifactStep.adopt("documents/snowball-inventory-runtime", pool_version, pool_uri)
    validation = ArtifactStep.adopt("documents/snowball-structural-validation", "2026.10.20.1", validation_uri)
    rl = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(training_recipe(max_steps), sort_keys=False),
            runtime=SkyRLRuntime(profile=SNOWBALL_RECIPE.profile),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=TOKENIZER_MODEL,
                tokenizer_revision=TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(validation, relative_path="validation.parquet"),),
            topology=SkyRLTopology(
                num_nodes=SNOWBALL_RECIPE.num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
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
            coordinator_timeout_hours=36,
            job_timeout_seconds=24 * 60 * 60,
            container_profile=CONTAINER_PROFILE,
            wandb_entity="marin-community",
        ),
        export_hf=True,
    )

    def build_config(ctx: StepContext) -> InventoryRunConfig:
        return InventoryRunConfig(rl.build_config(ctx), pool_sha256, validation_sha256)

    return replace(rl, run=run_inventory, build_config=build_config)


if __name__ == "__main__":
    main()
