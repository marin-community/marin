# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Mochi RL at full context with one trainer rack and two inference racks."""

import hashlib
from dataclasses import replace
from pathlib import Path

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
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
from rigging.filesystem.storage_path import StoragePath

TRIAL_ROOT = "s3://marin-us-east-08a/tmp/ttl=3d/romain/mochi-three-rack/2026.11.13.01a1c926.a1"
INPUT_VERSION = "2026.11.13.9400"
CONTEXT_CAP = 262144
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
    data_root: str,
    data_sha256: str,
    updates: int = 3,
    max_new_tokens: int = 1024,
    resume_version: str | None = None,
    resume_step: int = 0,
) -> ArtifactStep[SkyRLRun]:
    if 48 % MEGATRON_GEOMETRY["pipeline_model_parallel_size"]:
        raise ValueError("Mochi's 48 layers must divide evenly across pipeline stages")
    if 12 % MEGATRON_GEOMETRY["context_parallel_size"]:
        raise ValueError("Mochi's sliding-window all-to-all attention requires CP to divide its 12 KV heads")
    hf_root = str(StoragePath(TRIAL_ROOT) / "hf-sft")
    model = ArtifactHfModel(
        step=ArtifactStep.adopt(
            user_owned_name("checkpoints/mochi-sft-400407-schema3"),
            INPUT_VERSION,
            source=hf_root,
            kind=LevanterCheckpoint,
        ),
        relative_path="",
        tokenizer_uri=hf_root,
        tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    )
    data = ArtifactStep.adopt(
        user_owned_name(f"documents/mochi-three-rack/{data_sha256}"),
        INPUT_VERSION,
        source=data_root,
        kind=Artifact,
    )
    recipe = yaml.safe_load(Path(__file__).with_name("mochi_one_rack").joinpath("recipe.yaml").read_text())
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
        resume_mode="from_path" if resume_version else "none",
        restore_dataloader_state=False,
    )
    for role in ("policy", "ref"):
        trainer[role]["megatron_config"].update(MEGATRON_GEOMETRY)
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
            role_topology="nvlink.domain",
        ),
        export_hf=False,
    )
    previous = (
        ArtifactStep.adopt(
            RUN_NAME,
            resume_version,
            source=str(StoragePath(TRIAL_ROOT) / RUN_NAME / resume_version),
            kind=SkyRLRun,
        )
        if resume_version
        else None
    )
    original_build_config = step.build_config

    def bind_runtime(ctx: StepContext) -> SkyRLRunConfig:
        if not ctx.is_fingerprint:
            with (StoragePath(data_root) / "train.parquet").open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != data_sha256:
                raise ValueError("The mixed-task Parquet input differs from its pinned SHA-256")
        config = original_build_config(ctx)
        launch = yaml.safe_load(config.launch_config_yaml)
        traces = StoragePath(config.output.attempts_root) / "traces"
        launch["skyrl"]["generator"]["trajectory_retention"].update(
            output_path=str(traces),
            run_id=launch["run"]["id"],
        )
        launch["skyrl"]["terminal_bench"]["installed_agent"]["output_path"] = str(traces / "agent")
        if previous is not None:
            if ctx.is_fingerprint:
                checkpoint_root = f"<resume-checkpoints:{resume_version}>"
            else:
                checkpoint_root = ctx.resolved(previous).checkpoint_root
                latest = int((StoragePath(checkpoint_root) / "latest_ckpt_global_step.txt").read_text().strip())
                if latest != resume_step:
                    raise ValueError(f"Requested checkpoint {resume_step}, completed artifact records {latest}")
            launch["skyrl"]["trainer"]["resume_path"] = str(StoragePath(checkpoint_root) / f"global_step_{resume_step}")
        return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

    return replace(step, build_config=bind_runtime, deps=(*step.deps, previous) if previous else step.deps)


@click.command(help=__doc__)
@click.option("--data-root", required=True, help="Immutable mixed-task Parquet artifact directory.")
@click.option(
    "--data-sha256", required=True, help="SHA-256 of train.parquet; checked before launch and in its artifact name."
)
@click.option("--updates", type=click.IntRange(min=1), default=3)
@click.option("--max-new-tokens", type=click.IntRange(min=1, max=CONTEXT_CAP - 1), default=1024)
@click.option("--resume-version", default=None, help="Completed three-rack run artifact; starts fresh when omitted.")
@click.option("--resume-step", type=click.IntRange(min=1), default=None)
@rl_build_options
def main(
    data_root: str,
    data_sha256: str,
    updates: int,
    max_new_tokens: int,
    resume_version: str | None,
    resume_step: int | None,
) -> ArtifactStep[SkyRLRun]:
    if (resume_version is None) != (resume_step is None):
        raise click.UsageError("Continuation requires both --resume-version and --resume-step")
    return build_run(
        resolve_version("mochi-three-rack", None),
        data_root=data_root,
        data_sha256=data_sha256,
        updates=updates,
        max_new_tokens=max_new_tokens,
        resume_version=resume_version,
        resume_step=resume_step or 0,
    )


if __name__ == "__main__":
    main()
