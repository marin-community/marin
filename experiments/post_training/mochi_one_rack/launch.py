# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restart Hero MuonH at 4K before running five near-limit 32K updates."""

from dataclasses import dataclass, replace
from pathlib import Path

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
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

TRIAL_ROOT = "s3://marin-us-east-08a/tmp/ttl=7d/romain/mochi-rl-smoke/2026.11.11.01a1c926.a1"
INPUT_VERSION = "2026.11.11.9263"
DATA_VERSION = "2026.11.11.9270"
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=12,
    policy_num_gpus_per_node=4,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=12,
    inference_engine_expert_parallel_size=12,
    train_batch_size=2,
    policy_mini_batch_size=2,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


@dataclass(frozen=True)
class MochiSmoke:
    initial_four_k: ArtifactStep[SkyRLRun]
    four_k: ArtifactStep[SkyRLRun]
    thirty_two_k: ArtifactStep[SkyRLRun]


def build_smoke(
    version: str, *, initial_four_k_version: str | None = None, four_k_version: str | None = None
) -> MochiSmoke:
    trial_root = StoragePath(TRIAL_ROOT)
    hf_root = str(trial_root / "hf-sft")
    model = ArtifactHfModel(
        step=ArtifactStep.adopt(
            "users/romain/checkpoints/mochi-sft-395161-schema3",
            INPUT_VERSION,
            source=hf_root,
            kind=LevanterCheckpoint,
        ),
        relative_path="",
        tokenizer_uri=hf_root,
        tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    )
    execution = IrisSkyRLExecution(
        cluster="cw-us-east-08a",
        cluster_config="lib/iris/config/cw-us-east-08a.yaml",
        cpu=120,
        memory="940GB",
        disk="2TB",
        priority="production",
        max_retries=0,
        target_cluster="cw-us-east-08a",
        parent_cluster_config="lib/iris/config/marin.yaml",
        coordinator_timeout_hours=8,
        job_timeout_seconds=18000,
        wandb_entity="marin-community",
        required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY", "MARIN_PREFIX"),
        container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    )

    def training_step(stage: str, cap: int, max_steps: int, stage_version: str) -> ArtifactStep[SkyRLRun]:
        recipe = yaml.safe_load(Path(__file__).with_name("recipe.yaml").read_text())
        recipe["context_budget"] = {
            "request_window_tokens": cap,
            "max_new_tokens_per_turn": 2048 if cap == 4096 else 1024,
            "max_turns": 4,
        }
        recipe["trainer"]["max_steps"] = max_steps
        recipe["trainer"]["ckpt_interval"] = max_steps
        for callback in recipe["trainer"]["callbacks"]:
            if callback["type"] == "checkpoint":
                callback["save_steps"] = max_steps
        if stage != "4k-initial":
            recipe["trainer"].update(resume_mode="from_path", restore_dataloader_state=False)
        if stage == "32k":
            recipe["environment"]["task_sessions"]["task_spec"]["session"]["model_turn_timeout"] = 600
        data_stage = "32k" if stage == "32k" else "4k"
        data = ArtifactStep.adopt(
            f"users/romain/documents/mochi-one-rack/{data_stage}",
            DATA_VERSION,
            source=str(trial_root / "data" / DATA_VERSION / data_stage),
            kind=Artifact,
        )
        step = skyrl_step(
            SkyRLSpec(
                name=user_owned_name(f"checkpoints/mochi-muonh/{stage}"),
                version=stage_version,
                config_yaml=yaml.safe_dump(recipe, sort_keys=False),
                model=model,
                train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
                validation_data=(),
                topology=SkyRLTopology(
                    num_nodes=ROLE_PLAN.num_nodes,
                    gpus_per_node=4,
                    gpu_variant="GB200",
                    role_plan=ROLE_PLAN,
                ),
                retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=7),
                seed=20261020,
            ),
            execution,
            export_hf=False,
        )
        build_config = step.build_config

        def bind_traces(ctx: StepContext) -> SkyRLRunConfig:
            config = build_config(ctx)
            launch = yaml.safe_load(config.launch_config_yaml)
            traces = StoragePath(launch["artifacts"]["attempts_root"]) / "traces"
            launch["skyrl"]["generator"]["trajectory_retention"]["output_path"] = str(traces)
            launch["skyrl"]["generator"]["trajectory_retention"]["run_id"] = launch["run"]["id"]
            launch["skyrl"]["terminal_bench"]["installed_agent"]["output_path"] = str(traces / "agent")
            return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

        return replace(step, build_config=bind_traces)

    def continuation(
        previous: ArtifactStep[SkyRLRun], previous_step: int, stage: str, cap: int, max_steps: int, stage_version: str
    ) -> ArtifactStep[SkyRLRun]:
        step = training_step(stage, cap, max_steps, stage_version)
        build_config = step.build_config

        def resume_config(ctx: StepContext) -> SkyRLRunConfig:
            config = build_config(ctx)
            checkpoint_root = (
                f"<resume-checkpoints:{artifact_identity(previous)}>"
                if ctx.is_fingerprint
                else ctx.resolved(previous).checkpoint_root
            )
            launch = yaml.safe_load(config.launch_config_yaml)
            launch["skyrl"]["trainer"]["resume_path"] = str(
                StoragePath(checkpoint_root) / f"global_step_{previous_step}"
            )
            return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

        return replace(step, build_config=resume_config, deps=(*step.deps, previous))

    initial_four_k = training_step("4k-initial", 4096, 4, initial_four_k_version or four_k_version or version)
    four_k = continuation(initial_four_k, 4, "4k", 4096, 5, four_k_version or version)
    thirty_two_k = continuation(four_k, 5, "32k", 32768, 10, version)
    return MochiSmoke(initial_four_k, four_k, thirty_two_k)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(("4k-initial", "4k", "32k")), default="32k")
@click.option("--initial-four-k-version", default=None, help="Reuse a completed fresh-MuonH checkpoint-4 artifact.")
@click.option("--four-k-version", default=None, help="Reuse a completed 4K artifact when retrying the continuation.")
@rl_build_options
def main(stage: str, initial_four_k_version: str | None, four_k_version: str | None) -> ArtifactStep:
    run = build_smoke(
        resolve_version("mochi-one-rack", None),
        initial_four_k_version=initial_four_k_version,
        four_k_version=four_k_version,
    )
    if stage == "4k-initial":
        return run.initial_four_k
    return run.four_k if stage == "4k" else run.thirty_two_k


if __name__ == "__main__":
    main()
