# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one Snowball GRPO update over mixed chat and stateful tool tasks.

The smoke uses 16 answer-only arithmetic tasks and 16 stateful Workplace tasks.
Every prompt receives four sampled rollouts. A successful run proves that the
same optimizer batch can consume both task shapes through TaskCompendium's
Harbor lowering, preserve semantic verifier outcomes, update the policy, and
export a step-1 checkpoint.

Plan or run::

    python -m experiments.post_training.taskcompendium.rl_smoke --version 2026.09.15.1
    python -m experiments.post_training.taskcompendium.rl_smoke --version 2026.09.15.1 --run
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    TaskCompendiumDataSource,
    skyrl_step,
)
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.curriculum_rl.launch import (
    GPUS_PER_NODE,
    MARIN_TOKENIZER,
    MARIN_TOKENIZER_REVISION,
    SNOWBALL_MODEL,
    SNOWBALL_MUONH_OVERRIDES,
    SNOWBALL_POLICY,
)

TASKCOMPENDIUM_REQUIREMENT = (
    "taskcompendium @ git+https://github.com/marin-community/marin.git@"
    "dae4f5b961a8e3431d77d09a282867009418d5c8#subdirectory=lib/taskcompendium"
)
DATA_ARTIFACT_NAME = "documents/taskcompendium-snowball-mixed-smoke"
RL_ARTIFACT_NAME = "checkpoints/taskcompendium-snowball-mixed-smoke"
LOWERINGS = "lowerings"
CHAT_TASKS = 16
TOOL_TASKS = 16
TRAIN_BATCH_SIZE = CHAT_TASKS + TOOL_TASKS
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
NUM_NODES = 5
SEED = 17
MAX_STEPS = 1
REQUEST_WINDOW_TOKENS = 2048
MAX_NEW_TOKENS_PER_TURN = 512
MAX_TURNS = 3

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    train_batch_size=TRAIN_BATCH_SIZE,
    policy_mini_batch_size=TRAIN_BATCH_SIZE,
    micro_train_batch_size_per_gpu=4,
    n_samples_per_prompt=4,
)


@dataclass(frozen=True)
class BuildLoweringsConfig:
    output_path: str


def build_lowerings(config: BuildLoweringsConfig) -> None:
    """Create native lowerings with placeholder policy bindings."""
    # TaskCompendium is intentionally isolated from Marin's base environment and
    # installed only for this CPU artifact step.
    from taskcompendium.execution import (  # noqa: PLC0415
        HarborExecutionConfig,
        HarborLaunchConfig,
        HarborTaskBinding,
        NoEnvironment,
    )
    from taskcompendium.importers.nemo_workplace import INTERFACE, provider_binding  # noqa: PLC0415
    from taskcompendium.lowering import lower_to_harbor  # noqa: PLC0415
    from taskcompendium.models import (  # noqa: PLC0415
        AnswerRequirements,
        AssistantFinal,
        ProviderStateVerifier,
        Rendering,
        Source,
        StepSpecification,
        TaskMetadata,
        TaskRequirements,
        TaskSpec,
        TaskTroveVerifier,
    )
    from taskcompendium.providers.nemo_workplace.provider import ADAPTER  # noqa: PLC0415
    from tasktrove_verify.spec import Mode  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as temporary:
        lowerings = Path(temporary) / LOWERINGS
        lowerings.mkdir()
        chat_binding = HarborTaskBinding(NoEnvironment())
        chat_rendering = Rendering("plain", AssistantFinal())
        provider_rendering = Rendering("provider-chat", AssistantFinal())
        provider = provider_binding()

        for index in range(CHAT_TASKS):
            left = 17 + index
            right = 31 + 2 * index
            specification = TaskSpec(
                id=f"smoke/chat/{index}",
                requirements=TaskRequirements(),
                resources=(),
                metadata=TaskMetadata(Source("taskcompendium-smoke", "1", f"chat-{index}", "1")),
                steps=(
                    StepSpecification(
                        instructions=f"Compute {left} + {right}. Return only the integer answer.",
                        verifier=TaskTroveVerifier(Mode.MATH, {"expected": str(left + right)}),
                    ),
                ),
            )
            lower_to_harbor(
                specification,
                (chat_rendering,),
                chat_binding,
                lowerings / f"chat-{index:02d}",
                reference_execution=HarborExecutionConfig(chat_binding, HarborLaunchConfig("chat")),
                agent_kwargs={"api_base": "http://127.0.0.1:1/v1", "max_tokens": 512, "temperature": 1.0},
                model_name="policy-placeholder",
            )

        for index in range(TOOL_TASKS):
            body = f"TaskCompendium mixed-training smoke reply {index}."
            specification = TaskSpec(
                id=f"smoke/workplace/{index}",
                requirements=TaskRequirements(action_interfaces=(INTERFACE,)),
                resources=(),
                metadata=TaskMetadata(
                    Source("taskcompendium-smoke", "1", f"workplace-{index}", "1"),
                    competencies=("stateful_tool_use",),
                    task_shape="stateful_domain",
                ),
                steps=(
                    StepSpecification(
                        instructions=f"Reply to email 00000057 with exactly this body: {body}",
                        verifier=ProviderStateVerifier(
                            INTERFACE,
                            ADAPTER,
                            {
                                "ground_truth": [
                                    {
                                        "name": "email_reply_email",
                                        "arguments": json.dumps({"email_id": "00000057", "body": body}),
                                    }
                                ]
                            },
                        ),
                        answer_requirements=AnswerRequirements("text"),
                    ),
                ),
            )
            lower_to_harbor(
                specification,
                (provider_rendering,),
                provider,
                lowerings / f"workplace-{index:02d}",
                reference_execution=HarborExecutionConfig(provider, HarborLaunchConfig("provider_chat")),
                agent_kwargs={
                    "api_base": "http://127.0.0.1:1/v1",
                    "max_tokens": 512,
                    "max_turns": MAX_TURNS,
                    "temperature": 1.0,
                },
                model_name="policy-placeholder",
            )

        StoragePath(config.output_path).upload_from(f"{temporary}/", recursive=True)


def lowerings_step(version: str | None = None) -> ArtifactStep[Artifact]:
    name = user_owned_name(DATA_ARTIFACT_NAME)
    return ArtifactStep(
        name=name,
        version=version or resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            build_lowerings,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            pip_packages=[TASKCOMPENDIUM_REQUIREMENT],
        ),
        build_config=lambda ctx: BuildLoweringsConfig(output_path=ctx.output_path),
    )


def rl_config_yaml(plan: SkyRLRolePlan) -> str:
    return f"""\
entrypoint: taskcompendium

config_groups:
  terminal_bench_config: terminal_bench

context_budget:
  request_window_tokens: {REQUEST_WINDOW_TOKENS}
  max_new_tokens_per_turn: {MAX_NEW_TOKENS_PER_TURN}
  max_turns: {MAX_TURNS}

terminal_bench:
  model_info:
  trace_upload:
    enabled: false

trainer:
  strategy: fsdp2
  flash_attn: true
  use_sample_packing: false
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: true
  epochs: 1
  max_steps: {MAX_STEPS}
  update_epochs_per_batch: 1
  train_batch_size: {plan.train_batch_size}
  policy_mini_batch_size: {plan.policy_mini_batch_size}
  eval_batch_size: {plan.train_batch_size}
  micro_forward_batch_size_per_gpu: 2
  micro_train_batch_size_per_gpu: {plan.micro_train_batch_size_per_gpu}
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: {MAX_STEPS}
  resume_mode: latest
  enable_db_registration: false
  logger: console
  project_name: marin-taskcompendium-smoke
  policy:
    optimizer_config:
      lr: 1.0e-5
      max_grad_norm: 1.0
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
  placement:
    colocate_all: false
    colocate_policy_ref: true
    policy_num_nodes: 4
    policy_num_gpus_per_node: {GPUS_PER_NODE}
    ref_num_nodes: 4
    ref_num_gpus_per_node: {GPUS_PER_NODE}

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: {plan.inference_engine_tensor_parallel_size}
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  num_inference_engines: {plan.num_inference_engines}
  n_samples_per_prompt: {plan.n_samples_per_prompt}
  gpu_memory_utilization: 0.75
  max_num_seqs: 32
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  enable_http_endpoint: true
  sampling_params:
    temperature: 1.0
    top_p: 1.0
    repetition_penalty: 1.1
  engine_init_kwargs:
    enable_auto_tool_choice: true
    tool_call_parser: hermes

data:
  kind: tasks
  train_data: []
  val_data: []

trajectory_runner:
  process_pool:
    num_coordinators: 2
    cpus_per_coordinator: 4
    executor_workers: 64
"""


def smoke_step(lowerings: ArtifactStep[Artifact], version: str | None = None) -> ArtifactStep[SkyRLModel]:
    name = user_owned_name(RL_ARTIFACT_NAME)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version or resolve_version(name, None),
            config_yaml=rl_config_yaml(ROLE_PLAN),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
            model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=MARIN_TOKENIZER,
                tokenizer_revision=MARIN_TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(TaskCompendiumDataSource(lowerings, relative_path=LOWERINGS),),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=NUM_NODES,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=1),
            seed=SEED,
            overrides=(
                "++trainer.hf_hub_repo_id=null",
                *SNOWBALL_POLICY.overrides,
                *SNOWBALL_MUONH_OVERRIDES,
            ),
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            wandb_entity=None,
        ),
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[SkyRLModel]:
    version = resolve_version(DATA_ARTIFACT_NAME, None)
    return smoke_step(lowerings_step(version), version)


if __name__ == "__main__":
    main()
