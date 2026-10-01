# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run eight staged ShellSim tasks through vLLM and one optimizer step."""

import click
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.curriculum_rl.launch import HF_EXPORT_SUBDIR, model_step
from experiments.post_training.curriculum_rl.pool import QWEN3_MODEL, QWEN3_REVISION

TASK_COUNT = 8
TASK_FILENAME = "tasks.parquet"
MODEL_VERSION = "2026.08.29"
CLUSTER = "cw-rno2a"
GPUS_PER_NODE = 8
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=1,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=GPUS_PER_NODE,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=1,
    inference_engine_expert_parallel_size=1,
    train_batch_size=TASK_COUNT,
    policy_mini_batch_size=TASK_COUNT,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
)


def write_smoke_tasks(output_path: str) -> None:
    """Write tasks whose second stage reads the first stage's file."""
    # The remote data step installs the optional task packages with the SkyRL runtime.
    from taskcompendium.environment import (  # noqa: PLC0415 -- optional task dependencies
        EnvironmentFile,
        EnvironmentKind,
        EnvironmentSpec,
        ShellVerifierSpec,
    )
    from taskcompendium.grading import exact_answer  # noqa: PLC0415 -- optional task dependencies
    from taskcompendium.models import (  # noqa: PLC0415 -- optional task dependencies
        AnswerType,
        ConversationInput,
        EnvironmentRequirements,
        Source,
        StageRewardStrategy,
        StageVerifierSpec,
        TaskSpec,
        TaskStage,
        TextMessage,
        VerifierKind,
        VerifierSpec,
    )
    from taskcompendium.parquet import write_tasks  # noqa: PLC0415 -- optional task dependencies

    tasks = []
    for index in range(TASK_COUNT):
        value = str(20 + index)
        shell_grade = ShellVerifierSpec(
            argv=("sh", "/private/grade.sh"),
            files=(
                EnvironmentFile(
                    path="/private/grade.sh",
                    content=f'if [ "$(cat /workspace/value)" = {value} ]; then echo 1; else echo 0; fi'.encode(),
                ),
            ),
            timeout=10,
        )
        tasks.append(
            TaskSpec(
                id=f"file-roundtrip-{index}",
                context=ConversationInput(
                    events=(
                        TextMessage(
                            role="user",
                            content=f"Use the shell tool to run `echo {value} > /workspace/value`. Then reply done.",
                        ),
                    )
                ),
                environment_requirements=EnvironmentRequirements(capabilities=("filesystem", "shell")),
                environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
                answer_type=AnswerType.TEXT,
                verifier=VerifierSpec(
                    kind=VerifierKind.STAGED,
                    parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
                ),
                stages=(
                    TaskStage(
                        name="write",
                        verifier=VerifierSpec(kind=VerifierKind.SHELL, parameters_json=shell_grade.model_dump_json()),
                    ),
                    TaskStage(
                        name="read",
                        context=ConversationInput(
                            events=(
                                TextMessage(
                                    role="user",
                                    content="Use the shell tool to read /workspace/value. Reply with only the number.",
                                ),
                            )
                        ),
                        verifier=exact_answer(value),
                    ),
                ),
                attempt_timeout=180,
                source=Source(dataset="shellsim-file-roundtrip", revision="1", row=str(index), importer_revision="1"),
            )
        )
    write_tasks(prefix_join(output_path, TASK_FILENAME), iter(tasks))


CONFIG_YAML = """\
entrypoint: taskcompendium
context_budget:
  request_window_tokens: 4096
  max_new_tokens_per_turn: 256
  max_turns: 4
trainer:
  strategy: megatron
  flash_attn: true
  use_sample_packing: false
  algorithm:
    advantage_estimator: reward
    use_kl_loss: false
    off_policy_correction: tis
  epochs: 1
  max_steps: 1
  update_epochs_per_batch: 1
  micro_forward_batch_size_per_gpu: 1
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: 1
  resume_mode: latest
  logger: console
  project_name: marin-task-rollouts
  policy:
    optimizer_config:
      lr: 2.0e-6
      max_grad_norm: 1.0
generator:
  backend: vllm
  engine_init_kwargs:
    enable_auto_tools: true
    tool_parser: hermes
  chat_template_kwargs:
    enable_thinking: false
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  gpu_memory_utilization: 0.75
  enforce_eager: true
  run_engines_locally: true
  weight_sync_backend: nccl
  sampling_params:
    temperature: 1.0
    top_p: 1.0
data:
  kind: tasks
  train_data: []
  val_data: []
trajectory_runner:
  rollout_workers:
    num_workers: 2
    cpus_per_worker: 4
"""


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep[SkyRLRun]:
    tasks = apply(
        user_owned_name("documents/task-rollout-smoke"),
        remote(
            write_smoke_tasks,
            resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="16g"),
            pip_packages=[MARIN_SKYRL.requirement()],
            env_vars={"UV_PRERELEASE": "allow"},
        ),
        output_path=OUT,
    )
    name = user_owned_name("checkpoints/task-rollout-smoke")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=CONFIG_YAML,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=model_step(MODEL_VERSION),
                tokenizer_uri=QWEN3_MODEL,
                tokenizer_revision=QWEN3_REVISION,
                relative_path=HF_EXPORT_SUBDIR,
            ),
            train_data=(ArtifactDataSource(tasks, relative_path=TASK_FILENAME),),
            validation_data=(),
            topology=SkyRLTopology(num_nodes=2, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=1),
            seed=17,
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
            coordinator_timeout_hours=1,
            wandb_entity=None,
        ),
        export_hf=True,
    )


if __name__ == "__main__":
    main()
