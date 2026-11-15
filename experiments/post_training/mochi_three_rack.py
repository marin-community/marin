# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Mochi RL at full context with one trainer rack and two inference racks."""

import hashlib
from dataclasses import replace
from pathlib import Path, PurePosixPath

import click
import yaml
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe.locality import RoleTopologyMode

CONTEXT_CAP = 262144
DEFAULT_MAX_NEW_TOKENS = 8192
RUN_NAME = user_owned_name("checkpoints/mochi-three-rack")
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=16,
    policy_num_gpus_per_node=4,
    num_inference_engines=2,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=64,
    inference_engine_expert_parallel_size=64,
    train_batch_size=2,
    policy_mini_batch_size=2,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)
MEGATRON_GEOMETRY = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 16,
    "context_parallel_size": 4,
    "expert_model_parallel_size": 4,
    "expert_tensor_parallel_size": 1,
}


def build_run(
    version: str,
    *,
    model_uri: str,
    tokenizer_revision: str,
    input_version: str,
    data_root: str,
    data_sha256: str,
    tokenizer_uri: str | None = None,
    updates: int = 3,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    resume_uri: str | None = None,
) -> ArtifactStep[SkyRLRun]:
    if 48 % MEGATRON_GEOMETRY["pipeline_model_parallel_size"]:
        raise ValueError("Mochi's 48 layers must divide evenly across pipeline stages")
    if 12 % MEGATRON_GEOMETRY["context_parallel_size"]:
        raise ValueError("Mochi's sliding-window all-to-all attention requires CP to divide its 12 KV heads")
    resume_step = (
        int(PurePosixPath(resume_uri.rstrip("/")).name.removeprefix("global_step_")) if resume_uri is not None else 0
    )
    model = ArtifactHfModel(
        step=ArtifactStep.adopt(
            f"{RUN_NAME}/inputs/model",
            input_version,
            source=model_uri,
            kind=LevanterCheckpoint,
        ),
        relative_path="",
        tokenizer_uri=tokenizer_uri or model_uri,
        tokenizer_revision=tokenizer_revision,
    )
    data = ArtifactStep.adopt(
        user_owned_name(f"documents/mochi-three-rack/{data_sha256}"),
        input_version,
        source=data_root,
        kind=Artifact,
    )
    recipe = yaml.safe_load(Path(__file__).with_name("mochi_rl").joinpath("recipe.yaml").read_text())
    recipe["context_budget"] = {
        "request_window_tokens": CONTEXT_CAP,
        "max_new_tokens_per_turn": max_new_tokens,
        "max_turns": 4,
    }
    trainer = recipe["trainer"]
    trainer.update(
        project_name="mochi-three-rack",
        max_steps=resume_step + updates,
        ckpt_interval=resume_step + updates,
        epochs=resume_step + updates,
    )
    for role in ("policy", "ref"):
        role_config = trainer.setdefault(role, {})
        # Keep per-peer communication buffers small beside the full-context backward graph.
        role_config["nccl_buffer_size_bytes"] = 262144
        megatron = role_config.setdefault("megatron_config", {})
        megatron.update(MEGATRON_GEOMETRY)
        # The unfused expert SwiGLU exhausted memory during 261K backward recomputation.
        # Earlier pipeline stages also need their checkpoint inputs in pinned host memory.
        megatron.setdefault("transformer_config_kwargs", {}).update(
            bias_activation_fusion=True, grug_checkpoint_cpu_offload=True
        )
    for callback in trainer["callbacks"]:
        if callback["type"] == "checkpoint":
            callback["save_steps"] = resume_step + updates
    recipe["environment"]["task_sessions"]["task_spec"]["session"]["model_turn_timeout"] = 600
    recipe["generator"]["expert_block_sync"].update(verify=True, timeout_seconds=1800)
    # At 0.75 the full-context KV cache missed its startup requirement by 0.47 GiB.
    recipe["generator"]["gpu_memory_utilization"] = 0.80
    recipe["generator"]["trajectory_retention"].update(
        max_bytes_per_step=256 * 1024**2,
        max_bytes_per_run=1024**3,
    )
    recipe["extra_env"].update(
        NCCL_NET="IB",
        NCCL_IB_DISABLE="0",
        NCCL_IB_HCA="ibp",
        NCCL_DEBUG="INFO",
        NCCL_DEBUG_SUBSYS="INIT,NET",
    )
    step = skyrl_step(
        SkyRLSpec(
            name=RUN_NAME,
            version=version,
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            model=model,
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=ROLE_PLAN.num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="GB200",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3),
            seed=20261020,
            starting_checkpoint=(
                ArtifactStep.adopt(f"{RUN_NAME}/inputs/resume", version, source=resume_uri) if resume_uri else None
            ),
        ),
        IrisSkyRLExecution(
            cluster="cw-us-east-08a",
            cluster_config="lib/iris/config/cw-us-east-08a.yaml",
            cpu=120,
            memory="900GB",
            disk="2TB",
            priority="production",
            max_retries=0,
            target_cluster="cw-us-east-08a",
            parent_cluster_config="lib/iris/config/marin.yaml",
            coordinator_timeout_hours=72,
            job_timeout_seconds=21600,
            wandb_entity="marin-community",
            required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY", "MARIN_PREFIX"),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            role_topology=RoleTopologyMode.NVLINK_DOMAIN,
        ),
        export_hf=False,
    )
    original_build_config = step.build_config

    def bind_runtime(ctx: StepContext) -> SkyRLRunConfig:
        if not ctx.is_fingerprint:
            with (StoragePath(data_root) / "train.parquet").open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != data_sha256:
                raise ValueError("The mixed-task Parquet input differs from its pinned SHA-256")
        return original_build_config(ctx)

    return replace(step, build_config=bind_runtime)


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Completed schema-3 BF16 SFT export; no SFT job is submitted.")
@click.option("--tokenizer-uri", help="Pinned tokenizer artifact; defaults to the model export.")
@click.option("--tokenizer-revision", required=True)
@click.option("--input-version", required=True, help="Immutable version identifying the adopted model and data.")
@click.option("--data-root", required=True, help="Immutable mixed-task Parquet artifact directory.")
@click.option(
    "--data-sha256", required=True, help="SHA-256 of train.parquet; checked before launch and in its artifact name."
)
@click.option("--updates", type=click.IntRange(min=1), default=3)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1, max=CONTEXT_CAP - 1),
    default=DEFAULT_MAX_NEW_TOKENS,
    show_default=True,
    help="Response tokens per turn within the 262,144-token context window; use 1024 for short smoke runs.",
)
@click.option("--resume-uri", help="Completed native checkpoint directory; recovery prefers this run's saved progress.")
@click.option("--config-output", type=click.Path(dir_okay=False, path_type=Path), help="Write the source launch YAML.")
@rl_build_options
def main(
    model_uri: str,
    tokenizer_uri: str | None,
    tokenizer_revision: str,
    input_version: str,
    data_root: str,
    data_sha256: str,
    updates: int,
    max_new_tokens: int,
    resume_uri: str | None,
    config_output: Path | None,
) -> ArtifactStep[SkyRLRun]:
    step = build_run(
        resolve_version("mochi-three-rack", None),
        model_uri=model_uri,
        tokenizer_uri=tokenizer_uri,
        tokenizer_revision=tokenizer_revision,
        input_version=input_version,
        data_root=data_root,
        data_sha256=data_sha256,
        updates=updates,
        max_new_tokens=max_new_tokens,
        resume_uri=resume_uri,
    )
    if config_output is not None and get_job_info() is None:
        config_output.write_text(materialized_config(step, marin_prefix()).launch_config_yaml)
    return step


if __name__ == "__main__":
    main()
