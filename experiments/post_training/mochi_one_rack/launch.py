# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Mochi MuonH resume ladder and measured context trials."""

from dataclasses import dataclass, replace
from pathlib import Path

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, artifact_identity
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

from experiments.post_training.mochi_one_rack.data import write_context_trial_data

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


_TRIAL_PATH = StoragePath(TRIAL_ROOT)
_HF_ROOT = str(_TRIAL_PATH / "hf-sft")
_MODEL = ArtifactHfModel(
    step=ArtifactStep.adopt(
        "users/romain/checkpoints/mochi-sft-395161-schema3",
        INPUT_VERSION,
        source=_HF_ROOT,
        kind=LevanterCheckpoint,
    ),
    relative_path="",
    tokenizer_uri=_HF_ROOT,
    tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
)
_EXECUTION = IrisSkyRLExecution(
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
    job_timeout_seconds=18000,
    wandb_entity="marin-community",
    required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY", "MARIN_PREFIX"),
    container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
)


def _training_step(
    stage: str, cap: int, max_steps: int, stage_version: str, *, data: ArtifactStep[Artifact], max_new_tokens: int
) -> ArtifactStep[SkyRLRun]:
    recipe = yaml.safe_load(Path(__file__).with_name("recipe.yaml").read_text())
    recipe["context_budget"] = {
        "request_window_tokens": cap,
        "max_new_tokens_per_turn": max_new_tokens,
        "max_turns": 4,
    }
    recipe["trainer"]["max_steps"] = max_steps
    recipe["trainer"]["ckpt_interval"] = max_steps
    for callback in recipe["trainer"]["callbacks"]:
        if callback["type"] == "checkpoint":
            callback["save_steps"] = max_steps
    if stage != "4k-initial":
        recipe["trainer"].update(resume_mode="from_path", restore_dataloader_state=False)
    if cap >= 32768:
        recipe["environment"]["task_sessions"]["task_spec"]["session"]["model_turn_timeout"] = 600
    step = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/mochi-muonh/{stage}"),
            version=stage_version,
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            model=_MODEL,
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
        _EXECUTION,
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


def _continuation(
    previous: ArtifactStep[SkyRLRun],
    previous_step: int,
    stage: str,
    cap: int,
    max_steps: int,
    stage_version: str,
    *,
    data: ArtifactStep[Artifact],
    max_new_tokens: int,
) -> ArtifactStep[SkyRLRun]:
    step = _training_step(stage, cap, max_steps, stage_version, data=data, max_new_tokens=max_new_tokens)
    build_config = step.build_config

    def resume_config(ctx: StepContext) -> SkyRLRunConfig:
        config = build_config(ctx)
        checkpoint_root = (
            f"<resume-checkpoints:{artifact_identity(previous)}>"
            if ctx.is_fingerprint
            else ctx.resolved(previous).checkpoint_root
        )
        launch = yaml.safe_load(config.launch_config_yaml)
        launch["skyrl"]["trainer"]["resume_path"] = str(StoragePath(checkpoint_root) / f"global_step_{previous_step}")
        return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

    return replace(step, build_config=resume_config, deps=(*step.deps, previous))


def _retained_data(stage: str) -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(
        f"users/romain/documents/mochi-one-rack/{stage}",
        DATA_VERSION,
        source=str(_TRIAL_PATH / "data" / DATA_VERSION / stage),
        kind=Artifact,
    )


def build_smoke(
    version: str, *, initial_four_k_version: str | None = None, four_k_version: str | None = None
) -> MochiSmoke:
    four_k_data = _retained_data("4k")
    initial_four_k = _training_step(
        "4k-initial",
        4096,
        4,
        initial_four_k_version or four_k_version or version,
        data=four_k_data,
        max_new_tokens=2048,
    )
    four_k = _continuation(
        initial_four_k,
        4,
        "4k",
        4096,
        5,
        four_k_version or version,
        data=four_k_data,
        max_new_tokens=2048,
    )
    thirty_two_k = _continuation(
        four_k,
        5,
        "32k",
        32768,
        10,
        version,
        data=_retained_data("32k"),
        max_new_tokens=1024,
    )
    return MochiSmoke(initial_four_k, four_k, thirty_two_k)


def build_context_trial(
    version: str, *, context_cap: int, resume_stage: str, resume_version: str, resume_step: int, updates: int = 2
) -> ArtifactStep[SkyRLRun]:
    """Continue a completed MuonH checkpoint with consecutive near-limit batches."""
    if context_cap < 4096 or context_cap % 1024:
        raise ValueError("Context trials use a cap of at least 4096 in multiples of 1024")
    if updates < 2:
        raise ValueError("A context trial requires at least two consecutive updates")
    stage = f"{context_cap // 1024}k"
    if context_cap >= 32768:
        max_new_tokens = 1024
    elif context_cap >= 16384:
        max_new_tokens = 256
    else:
        max_new_tokens = 64
    previous_name = user_owned_name(f"checkpoints/mochi-muonh/{resume_stage}")
    previous = ArtifactStep.adopt(
        user_owned_name(f"checkpoints/mochi-muonh/resume/{resume_stage}"),
        resume_version,
        source=str(_TRIAL_PATH / previous_name / resume_version),
        kind=SkyRLRun,
    )
    data = apply(
        user_owned_name(f"documents/mochi-one-rack/{stage}"),
        write_context_trial_data,
        version=version,
        retained_data=_retained_data("4k"),
        retained_data_sha256="252aa6ed5ebf99c06ccc078bbea46aa1769307a928e1a4246182f016c3d30eaf",
        tokenizer_source=_MODEL.step,
        output_path=OUT,
        context_cap=context_cap,
        updates=updates,
        max_new_tokens=max_new_tokens,
    )
    step = _continuation(
        previous,
        resume_step,
        stage,
        context_cap,
        resume_step + updates,
        version,
        data=data,
        max_new_tokens=max_new_tokens,
    )
    build_config = step.build_config

    def completed_resume_config(ctx: StepContext) -> SkyRLRunConfig:
        config = build_config(ctx)
        if not ctx.is_fingerprint:
            checkpoint_root = StoragePath(ctx.resolved(previous).checkpoint_root)
            with (checkpoint_root / "latest_ckpt_global_step.txt").open("rt") as stream:
                latest_step = int(stream.read().strip())
            if latest_step != resume_step:
                raise ValueError(f"Requested checkpoint {resume_step}, but the completed artifact records {latest_step}")
        launch = yaml.safe_load(config.launch_config_yaml)
        launch["skyrl"]["trainer"]["max_epochs"] = resume_step + updates + 1
        return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

    return replace(step, build_config=completed_resume_config)


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(("4k-initial", "4k", "32k", "context")), default="32k")
@click.option("--initial-four-k-version", default=None, help="Reuse a completed fresh-MuonH checkpoint-4 artifact.")
@click.option("--four-k-version", default=None, help="Reuse a completed 4K artifact when retrying the continuation.")
@click.option(
    "--context-cap", type=click.IntRange(min=4096), default=None, help="Combined token cap for a context trial."
)
@click.option("--resume-stage", default=None, help="Completed MuonH stage, such as 32k or 48k.")
@click.option("--resume-version", default=None, help="Completed MuonH artifact version.")
@click.option("--resume-step", type=click.IntRange(min=5), default=None, help="Exact saved optimizer step to continue.")
@click.option(
    "--updates", type=click.IntRange(min=2), default=None, help="Consecutive context-trial updates; default 2."
)
@rl_build_options
def main(
    stage: str,
    initial_four_k_version: str | None,
    four_k_version: str | None,
    context_cap: int | None,
    resume_stage: str | None,
    resume_version: str | None,
    resume_step: int | None,
    updates: int | None,
) -> ArtifactStep:
    version = resolve_version("mochi-one-rack", None)
    if stage == "context":
        if context_cap is None or resume_stage is None or resume_version is None or resume_step is None:
            raise click.UsageError(
                "Context trials require --context-cap, --resume-stage, --resume-version and --resume-step"
            )
        if initial_four_k_version is not None or four_k_version is not None:
            raise click.UsageError("Context trials use --resume-version instead of the 4K version options")
        return build_context_trial(
            version,
            context_cap=context_cap,
            resume_stage=resume_stage,
            resume_version=resume_version,
            resume_step=resume_step,
            updates=updates if updates is not None else 2,
        )
    if any(value is not None for value in (context_cap, resume_stage, resume_version, resume_step, updates)):
        raise click.UsageError("Context-trial options require --stage context")
    run = build_smoke(
        version,
        initial_four_k_version=initial_four_k_version,
        four_k_version=four_k_version,
    )
    if stage == "4k-initial":
        return run.initial_four_k
    return run.four_k if stage == "4k" else run.thirty_two_k


if __name__ == "__main__":
    main()
