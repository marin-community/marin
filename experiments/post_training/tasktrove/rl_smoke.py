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
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    TaskTroveDataSource,
    TaskTroveSelection,
    skyrl_step,
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
MAX_TURNS = 1

# One H100 node with colocated policy and inference actors. Qwen3-0.6B is the smallest mirrored
# policy that exercises the real vLLM, Harbor, weight-sync, optimizer, checkpoint, and export path.
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=True,
    policy_num_nodes=1,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=GPUS_PER_NODE,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=1,
    inference_engine_expert_parallel_size=1,
    train_batch_size=SELECTED_TASKS,
    policy_mini_batch_size=SELECTED_TASKS,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=2,
)


def rl_config_yaml(plan: SkyRLRolePlan) -> str:
    return f"""\
entrypoint: terminal_bench

config_groups:
  terminal_bench_config: terminal_bench

context_budget:
  request_window_tokens: {REQUEST_WINDOW_TOKENS}
  max_new_tokens_per_turn: {MAX_NEW_TOKENS_PER_TURN}
  max_turns: {MAX_TURNS}

terminal_bench:
  harbor:
    name: terminus-2
    enable_summarize: false
    store_all_messages: true
    strict_json_parser: true
    interleaved_thinking: false
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
    override_timeout_sec: 600
    override_cpus: 1
    override_memory_mb: 2048
    override_storage_mb: 2048
    auto_snapshot: true
    verifier_override_timeout_sec: 300
    max_retries: 2
    min_wait_sec: 30.0
    max_wait_sec: 300.0
    wait_multiplier: 2.0
    exclude_exceptions:
      - VerifierTimeoutError
      - VerifierRuntimeError
      - RewardFileNotFoundError
      - RewardFileEmptyError
      - VerifierOutputParseError
    n_concurrent_trials: 16
    log_level: INFO
    enable_reward_shaping: false
    # Harbor's exact-token continuation asks the inference server for /tokenize, which the SkyRL
    # HTTP endpoint does not serve; without rollout details Harbor counts tokens locally instead.
    collect_rollout_details: false
    enable_error_classification: true
    mask_exceptions:
      - DaytonaError
      - EnvironmentStartTimeoutError
      - NetworkError
      - ConnectionError
      - RewardFileNotFoundError
      - RewardFileEmptyError
      - AgentEnvironmentTimeoutError
      - ContextLengthExceededError
    default_error_treatment: zero
    passthrough_exceptions:
      - AgentTimeoutError
    zero_exceptions: []
  model_info:
  archiving:
    enabled: false
  trace_upload:
    enabled: false

trainer:
  strategy: fsdp2
  flash_attn: true
  use_sample_packing: false
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
  epochs: 1
  max_steps: {MAX_STEPS}
  update_epochs_per_batch: 1
  eval_batch_size: {plan.train_batch_size}
  micro_forward_batch_size_per_gpu: 8
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: {MAX_STEPS}
  resume_mode: latest
  enable_db_registration: false
  logger: console
  project_name: {WANDB_PROJECT}
  hf_hub_repo_id: null
  policy:
    optimizer_config:
      lr: 2.0e-6
      max_grad_norm: 1.0
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  gpu_memory_utilization: 0.75
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  enable_http_endpoint: true
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: tasks
  train_data: []
  val_data: []

trajectory_runner:
  process_pool:
    num_coordinators: 2
    cpus_per_coordinator: 4
"""


def smoke_step(release: ArtifactStep) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(RL_ARTIFACT_NAME)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=rl_config_yaml(ROLE_PLAN),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
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
            topology=SkyRLTopology(
                num_nodes=NUM_NODES,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=ROLE_PLAN,
            ),
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
