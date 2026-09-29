# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-step mixed TaskCompendium RL smoke on the pinned September Grug SFT model.

Plan first::

    uv run python -m experiments.post_training.taskcompendium_grug_smoke \
      --version 2026.09.28.1 --runtime-commit <MarinSkyRL-commit>

The source task packages contain private verifier material. The data artifact is
stored in Marin's private artifact store and staged only for the RL environment.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    IrisSkyRLExecution,
    PinnedHfModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.storage_path import StoragePath, prefix_join

MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "b8c07f7df1df65525abbfdbcd1572318ba11c42f"
TASKCOMPENDIUM_COMMIT = "f0ca039722031315a4548a46b093c03096cc4a7b"
TASKCOMPENDIUM_REQUIREMENT = (
    "taskcompendium[harbor,workplace] @ "
    f"git+https://github.com/marin-community/marin.git@{TASKCOMPENDIUM_COMMIT}#subdirectory=lib/taskcompendium"
)
CLUSTER = "cw-us-east-02a"
GPUS_PER_NODE = 8
SEED = 17
TASK_COUNT = 2
MAX_STEPS = 1
MAX_TURNS = 3

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=True,
    policy_num_nodes=1,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=GPUS_PER_NODE,
    inference_engine_expert_parallel_size=GPUS_PER_NODE,
    train_batch_size=TASK_COUNT,
    policy_mini_batch_size=TASK_COUNT,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=2,
)


@dataclass(frozen=True)
class TaskPackagesConfig:
    output_path: str
    taskcompendium_commit: str


def write_task_packages(config: TaskPackagesConfig) -> None:
    """Lower one direct chat task and the pinned Workplace row into a private artifact."""
    from taskcompendium.grading import exact_answer  # noqa: PLC0415
    from taskcompendium.importers.nemo_workplace import load_fixture  # noqa: PLC0415
    from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor  # noqa: PLC0415
    from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec  # noqa: PLC0415
    from taskcompendium.submission import AnswerFormat, SubmissionConvention  # noqa: PLC0415

    if config.taskcompendium_commit != TASKCOMPENDIUM_COMMIT:
        raise ValueError("Task package builder requires the pinned TaskCompendium revision")
    chat = TaskSpec(
        id="arithmetic-7-plus-5",
        instructions="What is 7 + 5?",
        verifier=exact_answer("12"),
        source=Source(dataset="hand-authored", revision="2026-09-28", row="arithmetic-7-plus-5", importer_revision="1"),
        requirements=TaskRequirements(),
        answer_type=AnswerType.NUMBER,
    )
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)
    workplace, workplace_convention, workplace_binding = load_fixture()
    with tempfile.TemporaryDirectory(prefix="taskcompendium-smoke-") as temporary:
        root = Path(temporary)
        lower_to_harbor(chat, convention, HarborEnvironmentConfig(), root / "chat")
        lower_to_harbor(workplace, workplace_convention, workplace_binding, root / "workplace")
        for source in sorted(root.rglob("*")):
            if source.is_file():
                destination = prefix_join(config.output_path, source.relative_to(root).as_posix())
                with StoragePath(destination).open("wb") as output:
                    output.write(source.read_bytes())


def task_packages_step(version: str) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=user_owned_name("documents/taskcompendium-grug-smoke"),
        version=version,
        artifact_type=Artifact,
        run=remote(
            write_task_packages,
            resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="8g", target_cluster=CLUSTER),
            pip_packages=[TASKCOMPENDIUM_REQUIREMENT],
        ),
        build_config=lambda ctx: TaskPackagesConfig(
            output_path=ctx.output_path, taskcompendium_commit=TASKCOMPENDIUM_COMMIT
        ),
    )


def rl_config_yaml(plan: SkyRLRolePlan) -> str:
    return f"""\
entrypoint: taskcompendium

config_groups:
  terminal_bench_config: terminal_bench
  taskcompendium_config: taskcompendium

context_budget:
  request_window_tokens: 4096
  max_new_tokens_per_turn: 256
  max_turns: {MAX_TURNS}

taskcompendium:
  concurrency: 2
  max_turns: {MAX_TURNS}
  timeout: 300

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
  micro_forward_batch_size_per_gpu: 1
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: 0
  resume_mode: none
  enable_db_registration: false
  logger: console
  project_name: marin-taskcompendium-smoke
  hf_hub_repo_id: null
  policy:
    optimizer_config:
      lr: 1.0e-6
      max_grad_norm: 1.0
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
      expert_model_parallel_size: {GPUS_PER_NODE}
      use_grouped_mm: true
      ep_comm_backend: torch

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  gpu_memory_utilization: 0.65
  max_num_batched_tokens: 4096
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  enable_http_endpoint: true
  engine_init_kwargs:
    enable_auto_tool_choice: true
    tool_call_parser: hermes
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: tasks
  train_data: []
  val_data: []
"""


def smoke_step(packages: ArtifactStep[Artifact], runtime_commit: str, version: str) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name("checkpoints/taskcompendium-grug-smoke")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=rl_config_yaml(ROLE_PLAN),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP, commit=runtime_commit),
            model=PinnedHfModel(
                repository=MODEL,
                revision=MODEL_REVISION,
                tokenizer_repository=MODEL,
                tokenizer_revision=MODEL_REVISION,
            ),
            train_data=(ArtifactDataSource(packages),),
            validation_data=(),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1, temporary_storage_ttl_days=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512GB",
            disk="500GB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=4,
            wandb_entity=None,
        ),
    )


@click.command(help=__doc__)
@click.option("--runtime-commit", required=True, help="Immutable MarinSkyRL commit containing TaskCompendium routing")
@rl_build_options
def main(runtime_commit: str) -> ArtifactStep[SkyRLRun]:
    packages = task_packages_step(resolve_version("documents/taskcompendium-grug-smoke", None))
    return smoke_step(packages, runtime_commit, resolve_version("checkpoints/taskcompendium-grug-smoke", None))


if __name__ == "__main__":
    main()
