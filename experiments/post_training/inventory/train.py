# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Datakit SFT on the Snowball inventory and evaluate before and after RL."""

from dataclasses import replace

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
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
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_VERSION = "2026.09.21"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
DEFAULT_STEPS = 25
ARTIFACT_NAME = "checkpoints/snowball-inventory"
CLUSTER = "cw-us-east-02a"
SEED = 20261020
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192
EVALUATION_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
JUDGE_API_KEY_ENV = "TOGETHER_API_KEY"


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
            "daytona": {"api_key_env": EVALUATION_API_KEY_ENV, "ttl_minutes": 60},
            "remote_judge": {"base_url": "https://api.together.xyz/v1", "api_key_env": JUDGE_API_KEY_ENV},
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
    selection: ArtifactStep[Artifact],
    steps: int = DEFAULT_STEPS,
) -> ArtifactStep[SkyRLRun]:
    """Bind selected canonical tasks, pinned SFT weights and terminal outputs."""
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
            train_data=(ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(selection, relative_path=VALIDATION_FILENAME),),
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
            required_env_vars=(EVALUATION_API_KEY_ENV, JUDGE_API_KEY_ENV),
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--selection-name", required=True, help="Name in the task-selection recipe.")
@click.option("--selection-version", required=True, help="Version in the task-selection recipe.")
@click.option("--selection-uri", required=True, help="Task-selection artifact root.")
@click.option("--steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True)
@rl_build_options
def main(selection_name: str, selection_version: str, selection_uri: str, steps: int) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    selection = ArtifactStep.adopt(f"rl-data/task-selection/{selection_name}", selection_version, selection_uri)
    return build_run(
        resolve_version(name, None),
        selection=selection,
        steps=steps,
    )


if __name__ == "__main__":
    main()
