# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Datakit SFT on the Snowball 100k selection and evaluate before and after RL."""

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
from skyrlrecipe import RecipePatch, RLEntrypoint, SkyRLRecipe, sections
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training.async_rl import DEFAULT, SNOWBALL_RECIPE, training_config
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_VERSION = "2026.09.21"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
DEFAULT_STEPS = 25
ARTIFACT_NAME = "checkpoints/snowball-inventory"
SELECTION_PREFIX = "rl-data/task-selection"
DEFAULT_SELECTION_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04"
CLUSTER = "cw-us-east-02a"
SEED = 20261020
MAX_NUM_SEQS = 1
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192
EVALUATION_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
JUDGE_API_KEY_ENV = "TOGETHER_API_KEY"


def training_recipe(steps: int, *, reproducible_serving: bool = False) -> SkyRLRecipe:
    """Use the maintained Snowball geometry with synchronous RLOO-N updates."""
    preset = replace(
        DEFAULT,
        label="snowball-100k",
        max_steps=steps,
        eval_interval=steps,
        max_staleness_steps=0,
        request_window_tokens=REQUEST_WINDOW_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_policy="full_batch",
    )
    base = training_config(preset)
    config = SkyRLRecipe(
        entrypoint=RLEntrypoint.TASKCOMPENDIUM,
        context_budget=base.context_budget,
        trainer=base.trainer,
        generator=base.generator,
        data=sections.Data(kind="parquet", shuffle=True),
        environment=sections.Environment(
            task_sessions={
                "local": {},
                "daytona": {"api_key_env": EVALUATION_API_KEY_ENV, "ttl_minutes": 60},
                "remote_judge": {"base_url": "https://api.together.xyz/v1", "api_key_env": JUDGE_API_KEY_ENV},
            }
        ),
        extra_env={
            **base.extra_env,
            "VERIFYIT_JUDGE_MODEL": "openai/gpt-oss-120b",
            "NCCL_NET": "Socket",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_FAMILY": "AF_INET",
        },
    )
    config = config.merge(
        RecipePatch(
            generator=sections.Generator(
                trajectory_reward_shaping=sections.TrajectoryRewardShaping(
                    enabled=True,
                    # Enabling shaping would also apply the DAPO overlong penalty over the derived l_cache window.
                    overlong=sections.Overlong(penalty_scale=0.0),
                    reasoning_span=sections.ReasoningSpan(penalty=0.1, target="marker"),
                ),
            ),
            trainer=sections.Trainer(
                project_name="marin-snowball-curriculum",
                eval_num_prompts=None,
                dump_eval_results=True,
                algorithm=sections.Algorithm(
                    advantage_estimator="rloo_n",
                    group_advantage_min_size=2,
                    advantage_batch_normalize=False,
                    rloo_n_filter_zero_reward_groups=True,
                    use_entropy_loss=False,
                ),
                policy=sections.Policy(
                    optimizer_config=sections.PolicyOptimizerConfig(
                        lr=1e-6,
                        adam_betas=(0.9, 0.999),
                        weight_decay=0.0,
                        max_grad_norm=0.5,
                        scheduler="constant_with_warmup",
                        num_warmup_steps=0,
                    )
                ),
            ),
        )
    )

    if not reproducible_serving:
        return config

    plan = SNOWBALL_RECIPE.role_plan
    return config.merge(
        RecipePatch(
            generator=sections.Generator(
                eval_sampling_params=EvalSamplingParams(seed=SEED),
                enable_prefix_caching=True,
                max_num_seqs=MAX_NUM_SEQS,
                vllm_attention_backend="TRITON_ATTN",
                engine_init_kwargs={
                    "compilation_config": {
                        "inductor_compile_config": {
                            "deterministic": True,
                            "batch_invariant": False,
                            "combo_kernels": True,
                            "benchmark_combo_kernel": False,
                        }
                    }
                },
            ),
            trainer=sections.Trainer(algorithm=sections.Algorithm(batch_invariant=True)),
            trajectory_runner=sections.TrajectoryRunner(
                max_concurrent_tasks=plan.num_inference_engines
                * plan.inference_engine_data_parallel_size
                * MAX_NUM_SEQS,
            ),
        )
    )


def build_run(
    version: str,
    *,
    selection: ArtifactStep[Artifact],
    steps: int = DEFAULT_STEPS,
    export_hf: bool = False,
    reproducible_serving: bool = False,
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
            config_yaml=yaml.safe_dump(
                training_recipe(steps, reproducible_serving=reproducible_serving).to_skyrl(),
                sort_keys=False,
            ),
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
        export_hf=export_hf,
    )


@click.command(help=__doc__)
@click.option(
    "--selection-uri", default=DEFAULT_SELECTION_URI, show_default=True, help="Published task-selection artifact root."
)
@click.option("--steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True)
@click.option("--export-hf", is_flag=True, help="Export the final Hugging Face model after training.")
@click.option(
    "--reproducible-serving",
    is_flag=True,
    help="Use the one-sequence-per-rank serving profile from the fixed100 qualification.",
)
@rl_build_options
def main(selection_uri: str, steps: int, export_hf: bool, reproducible_serving: bool) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
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
    return build_run(
        resolve_version(name, None),
        selection=selection,
        steps=steps,
        export_hf=export_hf,
        reproducible_serving=reproducible_serving,
    )


if __name__ == "__main__":
    main()
