# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Datakit SFT on the Snowball 100k selection and evaluate before and after RL."""

from dataclasses import replace
from typing import Literal

import click
import pyarrow.parquet as pq
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
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch, RLEntrypoint, SkyRLRecipe, sections
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training.async_rl import DEFAULT, SNOWBALL_RECIPE, training_config
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_VERSION = "2026.09.21"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
TEMPORARY_STORAGE_PREFIX = "s3://marin-us-east-02a"
DEFAULT_STEPS = 25
ARTIFACT_NAME = "checkpoints/snowball-inventory"
SELECTION_PREFIX = "rl-data/task-selection"
DEFAULT_SELECTION_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04"
CLUSTER = "cw-us-east-02a"
SEED = 20261020
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95
MAX_NUM_SEQS = 1
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192
EVALUATION_API_KEY_ENV = "DAYTONA_EVAL_API_KEY"
JUDGE_API_KEY_ENV = "TOGETHER_API_KEY"


def training_recipe(
    steps: int,
    *,
    reproducible_serving: bool = False,
    learner: Literal["megatron", "levanter"] = "megatron",
    verify_publication: bool = False,
    validate_native_checkpoint: bool = False,
    deterministic_gpu_ops: bool = False,
    diagnostic_shutdown_observer: bool = False,
) -> SkyRLRecipe:
    """Use the maintained Snowball geometry with synchronous RLOO-N updates."""
    if validate_native_checkpoint and learner != "levanter":
        raise ValueError("Native checkpoint validation requires the Levanter learner")
    if (deterministic_gpu_ops or diagnostic_shutdown_observer) and learner != "levanter":
        raise ValueError("Native compiler and shutdown diagnostics require the Levanter learner")
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
            # The privileged profile below exposes the front-end Ethernet NIC's RoCE functions
            # (mlx5_*) beside the InfiniBand rails. NCCL routes some channels over them, which fails
            # with IBV_WC_RETRY_EXC_ERR(12). Pin NCCL to the rails (ibp0-ibp7).
            "NCCL_IB_HCA": "ibp",
        },
    )
    config = config.merge(
        RecipePatch(
            generator=sections.Generator(
                eval_sampling_params=EvalSamplingParams(seed=SEED, temperature=EVAL_TEMPERATURE, top_p=EVAL_TOP_P),
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

    if verify_publication or learner == "levanter":
        config = config.merge(
            RecipePatch(generator=sections.Generator(expert_block_sync=sections.ExpertBlockSync(verify=True)))
        )
    if learner == "levanter":
        plan = SNOWBALL_RECIPE.role_plan
        experts = SNOWBALL_RECIPE.megatron.expert_model_parallel_size
        config = config.merge(
            RecipePatch(
                trainer=sections.Trainer(
                    strategy="levanter",
                    batch_builder="driver",
                    offload_optimizer_during_rollouts=False,
                    policy=sections.Policy(
                        levanter_config=sections.PolicyLevanterConfig(
                            python_executable="/app/.levanter-policy-env/bin/python",
                            coordinator_port=38679,
                            command_timeout_seconds=1800,
                            initialization_timeout_seconds=3600,
                            mesh_axes=sections.LevanterMeshAxes(
                                replica_dcn=1,
                                replica=1,
                                data=plan.policy_num_nodes * plan.policy_num_gpus_per_node // experts,
                                expert=experts,
                                model=1,
                                context=1,
                            ),
                            projection_block_size=1024,
                            projection_logits_dtype="bfloat16",
                            attention_implementation="gpu_fa4_cute",
                            moe_implementation="ring_gather_combine",
                            router_dot_fp32=True,
                            moe_capacity_factor=experts,
                            fp32_activations=False,
                            head_gate_fp32=True,
                            xsa_fp32=True,
                            device_memory_fraction=0.90,
                            allocator="vmm",
                            numpy_madvise_hugepage=False,
                            preallocate=False,
                            validate_checkpoint_roundtrip=validate_native_checkpoint,
                            deterministic_gpu_ops=deterministic_gpu_ops,
                            diagnostic_shutdown_observer=diagnostic_shutdown_observer,
                        )
                    ),
                )
            )
        )

    if not reproducible_serving:
        return config

    plan = SNOWBALL_RECIPE.role_plan
    return config.merge(
        RecipePatch(
            generator=sections.Generator(
                # Exact repeats were qualified with greedy evaluation.
                eval_sampling_params=EvalSamplingParams(seed=SEED, temperature=0.0, top_p=1.0),
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
    learner: Literal["megatron", "levanter"] = "megatron",
    verify_publication: bool = False,
    validate_native_checkpoint: bool = False,
    deterministic_gpu_ops: bool = False,
    diagnostic_shutdown_observer: bool = False,
    max_retries_preemption: int | None = None,
    coordinator_route: Literal["hub", "regional"] = "hub",
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
                training_recipe(
                    steps,
                    reproducible_serving=reproducible_serving,
                    learner=learner,
                    verify_publication=verify_publication,
                    validate_native_checkpoint=validate_native_checkpoint,
                    deterministic_gpu_ops=deterministic_gpu_ops,
                    diagnostic_shutdown_observer=diagnostic_shutdown_observer,
                ).to_skyrl(),
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
            retention=SkyRLRetentionPolicy(
                resume_checkpoint_count=2,
                temporary_storage_prefix=TEMPORARY_STORAGE_PREFIX,
            ),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory=SNOWBALL_RECIPE.host_memory,
            disk="2TB",
            priority="batch",
            max_retries=0,
            max_retries_preemption=max_retries_preemption,
            target_cluster=CLUSTER if coordinator_route == "hub" else None,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG if coordinator_route == "hub" else None,
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
@click.option("--learner", type=click.Choice(["megatron", "levanter"]), default="megatron", show_default=True)
@click.option(
    "--coordinator-route",
    type=click.Choice(["hub", "regional"]),
    default="hub",
    show_default=True,
    help="Submit the coordinator through the Marin hub or directly to the CoreWeave regional controller.",
)
@click.option("--verify-publication", is_flag=True, help="Replay received weight bytes after every publication.")
@click.option(
    "--validate-native-checkpoint",
    is_flag=True,
    help="Validate a real update and exact checkpoint restoration in fresh native learner processes.",
)
@click.option(
    "--deterministic-gpu-ops",
    is_flag=True,
    help="Use the deterministic XLA GPU profile in every native learner process.",
)
@click.option(
    "--diagnostic-shutdown-observer",
    is_flag=True,
    help="Observe native rank-zero shutdown with the experimental Linux debugger.",
)
@click.option(
    "--max-retries-preemption",
    type=click.IntRange(min=0),
    default=None,
    help="Override preemption retries for the coordinator and its descendant jobs.",
)
@click.option(
    "--reproducible-serving",
    is_flag=True,
    help="Use batch-invariant serving with one sequence per rank and greedy evaluation.",
)
@rl_build_options
def main(
    selection_uri: str,
    steps: int,
    export_hf: bool,
    reproducible_serving: bool,
    learner: Literal["megatron", "levanter"],
    verify_publication: bool,
    validate_native_checkpoint: bool,
    deterministic_gpu_ops: bool,
    diagnostic_shutdown_observer: bool,
    max_retries_preemption: int | None,
    coordinator_route: Literal["hub", "regional"],
) -> ArtifactStep[SkyRLRun]:
    if learner == "levanter" and export_hf:
        raise click.BadParameter("Levanter supports native checkpoints; terminal HF export is not implemented")
    if validate_native_checkpoint and learner != "levanter":
        raise click.BadParameter("--validate-native-checkpoint requires --learner levanter")
    if (deterministic_gpu_ops or diagnostic_shutdown_observer) and learner != "levanter":
        raise click.BadParameter("Native compiler and shutdown diagnostics require --learner levanter")
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
    # SkyRL trains each task at most once per update, so a smaller selection fails only after GPU allocation.
    batch_size = training_recipe(steps).trainer.train_batch_size
    fs, train_path = filesystem_for(str(StoragePath(selection_uri) / TRAIN_FILENAME))
    with fs.open(train_path, "rb") as stream:
        train_rows = pq.ParquetFile(stream).metadata.num_rows
    if train_rows < batch_size:
        raise click.BadParameter(
            f"{TRAIN_FILENAME} has {train_rows} tasks; one update needs {batch_size}",
            param_hint="--selection-uri",
        )
    selection = ArtifactStep.adopt(f"{SELECTION_PREFIX}/{selection_name}", selection_version, selection_uri)
    return build_run(
        resolve_version(name, None),
        selection=selection,
        steps=steps,
        export_hf=export_hf,
        reproducible_serving=reproducible_serving,
        learner=learner,
        verify_publication=verify_publication,
        validate_native_checkpoint=validate_native_checkpoint,
        deterministic_gpu_ops=deterministic_gpu_ops,
        diagnostic_shutdown_observer=diagnostic_shutdown_observer,
        max_retries_preemption=max_retries_preemption,
        coordinator_route=coordinator_route,
    )


if __name__ == "__main__":
    main()
