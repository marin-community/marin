# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-train Qwen3-0.6B directly from a packed TaskTrove Clean release.

The run proves the release drives a Harbor RL loop end to end without an exploded staging
artifact. Each Iris node caches the clean Parquet file, and MarinSkyRL extracts a task only when
its rollout batch is about to construct the Harbor trial.

Plan or run::

    python -m experiments.post_training.tasktrove.rl_smoke --version 2026.09.10.2
    python -m experiments.post_training.tasktrove.rl_smoke --version 2026.09.10.2 --run

``--run`` validates the graph and submits a CPU coordinator to Iris. Set ``DAYTONA_API_KEY`` and
``HF_TOKEN`` on the submit host; the coordinator forwards both without embedding them in its command.
"""

from __future__ import annotations

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLHardware,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    TaskTroveDataSource,
    TaskTroveSelection,
    skyrl_step,
)
from marin.skyrl_recipe import (
    Algorithm,
    ConfigGroups,
    ContextBudget,
    Data,
    Generator,
    Placement,
    Policy,
    PolicyOptimizerConfig,
    RecipePatch,
    RLEntrypoint,
    RolloutWorkers,
    SamplingParams,
    SkyRLRecipe,
    Trainer,
    TrajectoryRunner,
)

from experiments.post_training.curriculum_rl.launch import HF_EXPORT_SUBDIR, model_step
from experiments.post_training.curriculum_rl.pool import QWEN3_MODEL, QWEN3_REVISION
from experiments.post_training.tasktrove.pipeline import build_workflow, launch_commit

RL_ARTIFACT_NAME = "checkpoints/tasktrove-rl-smoke"
# The curriculum experiment's mirrored Qwen3-0.6B snapshot; reused rather than mirrored again.
MODEL_VERSION = "2026.08.29"
TASKTROVE_SOURCE = "DCAgent2__nl2bash-tasks-cleaned-oracle-v2"
SELECTED_TASKS = 8
CLUSTER = "cw-rno2a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
NUM_NODES = 1
WANDB_PROJECT = "marin-tasktrove"
SEED = 17
MAX_STEPS = 1
REQUEST_WINDOW_TOKENS = 4096
MAX_NEW_TOKENS_PER_TURN = 256
MAX_TURNS = 4

# One H100 node with colocated policy and inference actors. Qwen3-0.6B is the smallest mirrored
# policy that exercises the real vLLM, Harbor, weight-sync, optimizer, checkpoint, and export path.
ROLE_PART = RecipePatch(
    trainer=Trainer(
        placement=Placement(
            colocate_all=True,
            colocate_policy_ref=True,
            policy_num_nodes=1,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            ref_num_nodes=1,
            ref_num_gpus_per_node=GPUS_PER_NODE,
        ),
        train_batch_size=SELECTED_TASKS,
        policy_mini_batch_size=SELECTED_TASKS,
        micro_train_batch_size_per_gpu=1,
    ),
    generator=Generator(
        num_inference_engines=GPUS_PER_NODE,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        n_samples_per_prompt=2,
    ),
)


def rl_recipe() -> SkyRLRecipe:
    return SkyRLRecipe.combine(
        base=SkyRLRecipe(
            entrypoint=RLEntrypoint.TERMINAL_BENCH,
            config_groups=ConfigGroups(terminal_bench_config="terminal_bench"),
            context_budget=ContextBudget(
                request_window_tokens=REQUEST_WINDOW_TOKENS,
                max_new_tokens_per_turn=MAX_NEW_TOKENS_PER_TURN,
                max_turns=MAX_TURNS,
            ),
            terminal_bench={
                "harbor": {
                    "name": "terminus-2",
                    "enable_summarize": False,
                    "store_all_messages": True,
                    "strict_json_parser": True,
                    "interleaved_thinking": False,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    "override_timeout_sec": 600,
                    "override_cpus": 1,
                    "override_memory_mb": 2048,
                    "override_storage_mb": 2048,
                    "auto_snapshot": True,
                    "verifier_override_timeout_sec": 300,
                    "max_retries": 2,
                    "min_wait_sec": 30.0,
                    "max_wait_sec": 300.0,
                    "wait_multiplier": 2.0,
                    "exclude_exceptions": (
                        "VerifierTimeoutError",
                        "VerifierRuntimeError",
                        "RewardFileNotFoundError",
                        "RewardFileEmptyError",
                        "VerifierOutputParseError",
                    ),
                    "n_concurrent_trials": 16,
                    "log_level": "INFO",
                    "enable_reward_shaping": False,
                    "collect_rollout_details": False,
                    "enable_error_classification": True,
                    "mask_exceptions": (
                        "DaytonaError",
                        "EnvironmentStartTimeoutError",
                        "NetworkError",
                        "ConnectionError",
                        "RewardFileNotFoundError",
                        "RewardFileEmptyError",
                        "AgentEnvironmentTimeoutError",
                        "ContextLengthExceededError",
                    ),
                    "default_error_treatment": "zero",
                    "passthrough_exceptions": ("AgentTimeoutError",),
                    "zero_exceptions": (),
                },
                "model_info": None,
                "archiving": {"enabled": False},
                "trace_upload": {"enabled": False},
            },
            trainer=Trainer(
                strategy="megatron",
                flash_attn=True,
                use_sample_packing=False,
                algorithm=Algorithm(advantage_estimator="grpo", use_kl_loss=False),
                epochs=1,
                max_steps=MAX_STEPS,
                update_epochs_per_batch=1,
                eval_batch_size=SELECTED_TASKS,
                eval_before_train=False,
                eval_interval=-1,
                ckpt_interval=MAX_STEPS,
                resume_mode="latest",
                enable_db_registration=False,
                logger="console",
                project_name=WANDB_PROJECT,
                hf_hub_repo_id=None,
                policy=Policy(optimizer_config=PolicyOptimizerConfig(lr=2e-06, max_grad_norm=1.0)),
            ),
            generator=Generator(
                backend="vllm",
                model_dtype="bfloat16",
                vllm_attention_backend="FLASH_ATTN",
                gpu_memory_utilization=0.75,
                enforce_eager=False,
                run_engines_locally=True,
                weight_sync_backend="nccl",
                enable_http_endpoint=True,
                sampling_params=SamplingParams(temperature=1.0, top_p=1.0),
            ),
            data=Data(kind="tasks", train_data=(), val_data=()),
            trajectory_runner=TrajectoryRunner(rollout_workers=RolloutWorkers(num_workers=2, cpus_per_worker=4)),
        ),
        geometry=ROLE_PART,
    )


def smoke_step(release: ArtifactStep) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(RL_ARTIFACT_NAME)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            recipe=rl_recipe(),
            model=ArtifactHfModel(
                step=model_step(MODEL_VERSION),
                tokenizer_uri=QWEN3_MODEL,
                tokenizer_revision=QWEN3_REVISION,
                relative_path=HF_EXPORT_SUBDIR,
            ),
            train_data=(
                TaskTroveDataSource(
                    release,
                    TaskTroveSelection(
                        sources=(TASKTROVE_SOURCE,),
                        tags=("bash", "terminal"),
                        modes=("script",),
                        limit=SELECTED_TASKS,
                        seed=SEED,
                    ),
                ),
            ),
            validation_data=(),
            hardware=SkyRLHardware(gpus_per_node=GPUS_PER_NODE, gpu_variant=GPU_VARIANT),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            memory="128GB",
            disk="1TB",
            priority="interactive",
            max_retries=1,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=12,
            wandb_entity=None,
        ),
    )


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep:
    release = build_workflow(launch_commit()).release
    return smoke_step(release)


if __name__ == "__main__":
    main()
