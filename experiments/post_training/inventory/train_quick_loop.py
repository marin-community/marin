# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Datakit SFT on the Snowball 100k selection and evaluate before and after RL."""

from dataclasses import replace
from enum import StrEnum

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
from experiments.post_training.inventory.loop_publication import publication_step, run_recorded_loop
from experiments.post_training.inventory.loop_result import LoopRunConfig
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_VERSION = "2026.09.21"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
DEFAULT_STEPS = 25
ARTIFACT_NAME = "checkpoints/snowball-inventory-batch-fixed"
SELECTION_PREFIX = "rl-data/task-selection"
DEFAULT_SELECTION_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.10.23.2"
CLUSTER = "cw-us-east-02a"
SEED = 20261020
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192
EVALUATION_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
JUDGE_API_KEY_ENV = "TOGETHER_API_KEY"
ATTENTION_HELPER_SHA256 = "fe39ea5a707a1979f9576595f580a2f70506a5868a389ee150202b45963b5f03"


class Arm(StrEnum):
    CONTROL = "a"
    LEARNING_RATE = "b"
    SEQUENCE_MEAN = "c"
    ASYNC = "d"


def training_recipe(steps: int, *, arm: Arm = Arm.CONTROL) -> dict:
    """Render one treatment with fixed training and greedy evaluation budgets."""
    preset = replace(
        DEFAULT,
        label="snowball-100k",
        max_steps=steps,
        eval_interval=steps,
        max_staleness_steps=2 if arm == Arm.ASYNC else 0,
        request_window_tokens=REQUEST_WINDOW_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_policy="full_batch",
    )
    config = training_config(preset)
    config["entrypoint"] = "taskcompendium"
    config["data"] = {"kind": "parquet", "shuffle": True, "expected_eval_rows": 100}
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
        resume_mode="none",
        loop_evidence_enabled=True,
    )
    trainer["algorithm"].update(
        batch_invariant=True,
        advantage_estimator="rloo_n",
        group_advantage_min_size=2,
        advantage_batch_normalize=False,
        rloo_n_filter_zero_reward_groups=True,
        use_entropy_loss=False,
        use_kl_loss=False,
        use_kl_in_reward=False,
        loss_reduction="token_mean",
        off_policy_correction="none",
        off_policy_correction_rules=None,
    )
    trainer["rollout_buffer"]["max_in_flight"] = 192 if arm == Arm.ASYNC else SNOWBALL_RECIPE.role_plan.train_batch_size
    config["generator"].update(
        eval_sampling_params={"seed": SEED, "temperature": 0.0},
        max_num_seqs=1024,
        vllm_attention_backend="TRITON_ATTN",
        required_vllm_attention_helper_sha256=ATTENTION_HELPER_SHA256,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        enforce_eager=False,
    )
    trainer["policy"]["optimizer_config"].update(
        lr=1e-6,
        adam_betas=[0.9, 0.999],
        weight_decay=0.0,
        max_grad_norm=0.5,
        scheduler="constant_with_warmup",
        num_warmup_steps=0,
    )
    if arm == Arm.LEARNING_RATE:
        trainer["policy"]["optimizer_config"]["lr"] = 2e-6
    if arm == Arm.SEQUENCE_MEAN:
        trainer["algorithm"]["loss_reduction"] = "sequence_mean"
    return config


def build_run(
    version: str,
    *,
    selection: ArtifactStep[Artifact],
    steps: int = DEFAULT_STEPS,
    arm: Arm = Arm.CONTROL,
) -> ArtifactStep[SkyRLRun]:
    """Bind selected canonical tasks, pinned SFT weights and terminal outputs."""
    model = ArtifactStep.adopt(
        f"models/snowball-datakit-sft-{MODEL_REVISION}",
        MODEL_VERSION,
        MODEL_URI,
        kind=LevanterCheckpoint,
    )
    plan = SNOWBALL_RECIPE.role_plan
    training = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"{ARTIFACT_NAME}-{arm.value}"),
            version=version,
            config_yaml=yaml.safe_dump(training_recipe(steps, arm=arm), sort_keys=False),
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
        export_hf=False,
    )

    def build_config(ctx):
        return LoopRunConfig(
            skyrl=training.build_config(ctx),
            label=f"Fixed runtime · {arm.value.upper()}{steps}",
            expected_steps=steps,
        )

    return replace(training, build_config=build_config, run=run_recorded_loop)


@click.command(help=__doc__)
@click.option(
    "--selection-uri", default=DEFAULT_SELECTION_URI, show_default=True, help="Published task-selection artifact root."
)
@click.option("--steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True)
@click.option("--arm", type=click.Choice([arm.value for arm in Arm]), default=Arm.CONTROL.value, show_default=True)
@rl_build_options
def main(selection_uri: str, steps: int, arm: str) -> ArtifactStep:
    selected_arm = Arm(arm)
    name = user_owned_name(f"{ARTIFACT_NAME}-{selected_arm.value}")
    selection_uri = selection_uri.rstrip("/")
    _, marker, identity = selection_uri.rpartition(f"/{SELECTION_PREFIX}/")
    parts = identity.split("/")
    if not marker or len(parts) != 2 or not all(parts):
        raise click.BadParameter(
            "Expected a published .../rl-data/task-selection/<name>/<version> artifact root",
            param_hint="--selection-uri",
        )
    selection_name, selection_version = parts
    selection = ArtifactStep.adopt(f"{SELECTION_PREFIX}/{selection_name}", selection_version, selection_uri)
    training = build_run(
        resolve_version(name, None),
        selection=selection,
        steps=steps,
        arm=selected_arm,
    )
    return publication_step(training)


if __name__ == "__main__":
    main()
