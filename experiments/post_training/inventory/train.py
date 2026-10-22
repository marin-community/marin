# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Datakit SFT on the Snowball inventory and evaluate before and after RL."""

from dataclasses import replace

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.async_rl import DEFAULT, SNOWBALL_RECIPE, training_config
from experiments.post_training.inventory.data import TRAIN_FILENAME, VALIDATION_FILENAME, VALIDATION_ROWS, data_step

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_VERSION = "2026.09.21"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
DATA_VERSION = "2026.10.22.2"
ARTIFACT_NAME = "checkpoints/snowball-inventory"
CLUSTER = "cw-us-east-02a"
SEED = 20261020
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192


def training_recipe(steps: int) -> dict:
    """Use the maintained Snowball geometry with synchronous RLOO-N updates."""
    plan = SNOWBALL_RECIPE.role_plan
    preset = replace(
        DEFAULT,
        label="inventory",
        max_steps=steps,
        eval_interval=steps,
        max_staleness_steps=0,
        generation_workers=plan.train_batch_size,
        request_window_tokens=REQUEST_WINDOW_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_policy="full_batch",
    )
    config = training_config(preset)
    config["entrypoint"] = "taskcompendium"
    config["data"] = {"kind": "parquet", "shuffle": True}
    config["environment"] = {
        "task_sessions": {
            "local": {},
            "remote_judge": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
        }
    }
    config["extra_env"].update(
        VERIFYIT_JUDGE_MODEL="openai/gpt-oss-120b",
        NCCL_NET="Socket",
        NCCL_IB_DISABLE="1",
        NCCL_SOCKET_FAMILY="AF_INET",
    )
    trainer = config["trainer"]
    trainer.update(
        seed=SEED,
        project_name="marin-snowball-curriculum",
        eval_num_prompts=None,
        eval_batch_size=VALIDATION_ROWS,
        dump_eval_results=True,
    )
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
    return config


def build_run(
    version: str,
    *,
    training_uri: str,
    validation_uri: str,
    steps: int = 25,
    data_version: str = DATA_VERSION,
) -> ArtifactStep[SkyRLRun]:
    """Bind the frozen data, pinned SFT model and terminal checkpoint/export."""
    data = data_step(data_version, training_uri, validation_uri)
    model = ArtifactStep.adopt(
        f"models/snowball-datakit-sft-{MODEL_REVISION}",
        MODEL_VERSION,
        MODEL_URI,
        kind=LevanterCheckpoint,
    )
    plan = SNOWBALL_RECIPE.role_plan
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(ARTIFACT_NAME),
            version=version,
            config_yaml=yaml.safe_dump(training_recipe(steps), sort_keys=False),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=MODEL_ID,
                tokenizer_revision=MODEL_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=plan.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=plan,
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
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            wandb_entity="marin-community",
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--train-source", required=True)
@click.option("--validation-source", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=25, show_default=True)
@click.option("--data-version", default=DATA_VERSION, show_default=True)
@rl_build_options
def main(train_source: str, validation_source: str, steps: int, data_version: str) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    return build_run(
        resolve_version(name, None),
        training_uri=train_source,
        validation_uri=validation_source,
        steps=steps,
        data_version=data_version,
    )


if __name__ == "__main__":
    main()
