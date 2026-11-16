# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or run Mochi RL with a named resource and context profile."""

import hashlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

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
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.locality import RoleTopologyMode

RECIPE_PATH = Path(__file__).with_name("recipe.yaml")
DEFAULT_MAX_NEW_TOKENS = 16384
SMOKE_BATCH_GROUPS = 2
SMOKE_BUDGET_SECONDS = 3600
SEED = 20261020
COORDINATOR_QUEUE_ALLOWANCE_HOURS = 48
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=12,
    policy_num_gpus_per_node=4,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=12,
    inference_engine_expert_parallel_size=12,
    train_batch_size=32,
    policy_mini_batch_size=32,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)
MEGATRON = MegatronGeometry(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=12,
    context_parallel_size=4,
    expert_model_parallel_size=4,
    expert_tensor_parallel_size=1,
)
EXECUTION = IrisSkyRLExecution(
    cluster="cw-us-east-08a",
    cluster_config="lib/iris/config/cw-us-east-08a.yaml",
    cpu=120,
    memory="900GB",
    disk="2TB",
    priority="production",
    max_retries=0,
    target_cluster="cw-us-east-08a",
    parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
    coordinator_timeout_hours=96,
    job_timeout_seconds=48 * 60 * 60,
    wandb_entity="marin-community",
    required_env_vars=(
        "DAYTONA_RL_API_KEY",
        "DAYTONA_EVAL_API_KEY",
        "DAYTONA_TARGET",
        "TOGETHER_API_KEY",
        "MARIN_PREFIX",
    ),
    container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    role_topology=RoleTopologyMode.NVLINK_DOMAIN,
)
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3)


@dataclass(frozen=True)
class MochiProfile:
    """Resource and recipe settings, with the CLI's default output artifact name."""

    output_name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry
    execution: IrisSkyRLExecution
    patch: RecipePatch = field(default_factory=RecipePatch)
    data_namespace: str | None = None


PROFILES = MappingProxyType({"one-rack": MochiProfile("checkpoints/mochi-rl-65k", ROLE_PLAN, MEGATRON, EXECUTION)})


def build_run(
    *,
    profile: MochiProfile,
    name: str,
    version: str,
    model: ArtifactHfModel,
    train_data: tuple[ArtifactDataSource, ...],
    max_steps: int,
    validation_data: tuple[ArtifactDataSource, ...] = (),
    smoke: bool = False,
    batch_groups: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    resume: ArtifactStep[Artifact] | None = None,
    recipe_path: Path = RECIPE_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    data_sha256: str | None = None,
    execution: IrisSkyRLExecution | None = None,
    retention: SkyRLRetentionPolicy = RETENTION,
) -> ArtifactStep[SkyRLRun]:
    """Build an RL training artifact from model, data and optional native checkpoint handles.

    A supplied data_sha256 requires one training-data artifact; profiles with
    data_namespace require a digest. When the data changes, the caller must
    supply a new source identity. The file is checked when the launch config is rendered.
    An execution override replaces the profile's placement and timeout settings.
    """
    if not smoke and not validation_data:
        raise ValueError("Normal Mochi RL requires held-out data; select smoke explicitly to omit evaluation")
    if profile.data_namespace is not None and data_sha256 is None:
        raise ValueError(f"{profile.output_name} requires data_sha256")
    if data_sha256 is not None and len(train_data) != 1:
        raise ValueError("data_sha256 requires exactly one training data source")
    groups = (
        batch_groups if batch_groups is not None else SMOKE_BATCH_GROUPS if smoke else profile.role_plan.train_batch_size
    )
    if groups < 2 or groups % 2:
        raise ValueError("Batch groups must be positive and even for equal agentic and nonagentic quotas")
    profile = replace(
        profile, role_plan=replace(profile.role_plan, train_batch_size=groups, policy_mini_batch_size=groups)
    )
    recipe = SkyRLRecipe.from_document(yaml.safe_load(recipe_path.read_text())).merge(profile.patch)
    eval_interval = recipe.trainer.eval_interval if eval_interval is None else eval_interval
    checkpoint_interval = recipe.trainer.ckpt_interval if checkpoint_interval is None else checkpoint_interval
    context = recipe.context_budget.replace(max_new_tokens_per_turn=max_new_tokens)
    callback_updates = {
        "checkpoint": {"save_steps": checkpoint_interval},
        "evaluation": {"eval_steps": eval_interval},
        "best_checkpoint": {"evaluation_interval": eval_interval},
        "wall_time_budget": {"groups_per_step": groups},
    }
    callbacks = []
    for callback in recipe.trainer.callbacks:
        if smoke and callback.type in ("evaluation", "best_checkpoint"):
            continue
        if smoke and callback.type == "wall_time_budget":
            callback = callback.replace(seconds=SMOKE_BUDGET_SECONDS)
        callbacks.append(callback.replace(**callback_updates.get(callback.type, {})))
    recipe = recipe.merge(
        RecipePatch(
            context_budget=context,
            trajectory_runner=sections.TrajectoryRunner(
                rollout_workers=sections.RolloutWorkers(
                    num_workers=SMOKE_BATCH_GROUPS if smoke else recipe.trajectory_runner.rollout_workers.num_workers
                )
            ),
            terminal_bench={
                "installed_agent": {
                    "agent": {
                        "kwargs": {
                            "model_info": {
                                "max_input_tokens": context.request_window_tokens,
                                "max_output_tokens": max_new_tokens,
                            },
                            "compaction_reserve_tokens": max_new_tokens,
                        }
                    }
                }
            },
            trainer=sections.Trainer(
                # Leave enough epochs for small task sets to reach the global-step target.
                epochs=max_steps,
                max_steps=max_steps,
                eval_interval=-1 if smoke else eval_interval,
                ckpt_interval=checkpoint_interval,
                callbacks=tuple(callbacks),
                rollout_buffer=sections.RolloutBuffer(
                    consumed_route_quotas={"agentic": groups // 2, "nonagentic": groups // 2},
                    max_in_flight=SMOKE_BATCH_GROUPS if smoke else recipe.trainer.rollout_buffer.max_in_flight,
                ),
                micro_forward_batch_size_per_gpu=profile.role_plan.micro_train_batch_size_per_gpu,
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**asdict(profile.megatron))),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**asdict(profile.megatron))),
            ),
        )
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(recipe.to_skyrl(), sort_keys=False),
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            topology=SkyRLTopology(
                num_nodes=profile.role_plan.num_nodes,
                gpus_per_node=profile.role_plan.policy_num_gpus_per_node,
                gpu_variant="GB200",
                role_plan=profile.role_plan,
            ),
            retention=retention,
            seed=SEED,
            starting_checkpoint=resume,
        ),
        execution or profile.execution,
        export_hf=False,
    )
    if data_sha256 is None:
        return step
    original_build_config = step.build_config

    def bind_runtime(ctx: StepContext) -> SkyRLRunConfig:
        if not ctx.is_fingerprint:
            source = train_data[0]
            uri = str(StoragePath(ctx.artifact_path(source.step)) / source.relative_path)
            # CoreWeave S3 inputs need the bucket registry's endpoint and credentials.
            fs, path = filesystem_for(uri)
            with fs.open(path, "rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != data_sha256:
                raise ValueError("The mixed-task Parquet input differs from its pinned SHA-256")
        return original_build_config(ctx)

    return replace(step, build_config=bind_runtime)


@click.command(help=__doc__)
@click.option("--profile", type=click.Choice(tuple(PROFILES)), required=True)
@click.option(
    "--smoke",
    is_flag=True,
    help="Use two groups and a one-hour budget after the first update; skip evaluation and selection.",
)
@click.option(
    "--batch-groups",
    type=click.IntRange(min=2),
    help="Even prompt-group count; defaults to the profile, or two for smoke.",
)
@click.option(
    "--eval-interval", type=click.IntRange(min=1), help="Evaluate every N updates; defaults to the recipe cadence."
)
@click.option(
    "--checkpoint-interval", type=click.IntRange(min=1), help="Save every N updates, independently of --max-steps."
)
@click.option("--model-uri", required=True, help="Completed schema-3 BF16 HF export with its portable model manifest.")
@click.option("--tokenizer-uri", help="Pinned tokenizer artifact; defaults to the model export.")
@click.option("--tokenizer-revision", required=True)
@click.option(
    "--data-uri", multiple=True, required=True, help="Mixed-task artifact directory. Repeat for several inputs."
)
@click.option("--data-sha256", help="SHA-256 of one data file; checked before launch and included in its identity.")
@click.option("--data-file", default="train.parquet", show_default=True)
@click.option(
    "--validation-data-uri", multiple=True, help="Held-out artifact directory; required unless --smoke is selected."
)
@click.option("--validation-data-file", default="validation.parquet", show_default=True)
@click.option("--input-version", required=True, help="Immutable version identifying the adopted input set.")
@click.option("--name", help="Output name within your user namespace; defaults to the selected profile's output name.")
@click.option("--max-steps", type=click.IntRange(min=1), required=True, help="Stop at this absolute optimizer step.")
@click.option(
    "--job-timeout-hours",
    type=click.IntRange(min=1),
    default=None,
    help="GPU job wall-clock limit. Resume from a completed checkpoint if reached.",
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_NEW_TOKENS,
    show_default=True,
    help="Response tokens per turn; use 1024 for a short smoke.",
)
@click.option("--resume-uri", help="Starting native RL checkpoint; recovery prefers this run's saved progress.")
@click.option("--checkpoint-count", type=click.IntRange(1, 10), default=RETENTION.resume_checkpoint_count)
@click.option("--ttl-days", type=click.IntRange(min=1), default=RETENTION.temporary_storage_ttl_days)
@click.option(
    "--config-output", type=click.Path(dir_okay=False, path_type=Path), help="Write the source launch YAML locally."
)
@rl_build_options
def main(
    profile: str,
    model_uri: str,
    tokenizer_uri: str | None,
    tokenizer_revision: str,
    data_uri: tuple[str, ...],
    data_file: str,
    data_sha256: str | None,
    input_version: str,
    name: str | None,
    max_steps: int,
    job_timeout_hours: int | None,
    max_new_tokens: int,
    resume_uri: str | None,
    checkpoint_count: int,
    ttl_days: int,
    config_output: Path | None,
    smoke: bool = False,
    batch_groups: int | None = None,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
    validation_data_uri: tuple[str, ...] = (),
    validation_data_file: str = "validation.parquet",
) -> ArtifactStep[SkyRLRun]:
    selected = PROFILES[profile]
    if not smoke and not validation_data_uri:
        raise click.UsageError("--validation-data-uri is required for normal RL; select --smoke to omit evaluation")
    if selected.data_namespace is not None and data_sha256 is None:
        raise click.UsageError(f"--data-sha256 is required for {profile}")
    if data_sha256 is not None and len(data_uri) != 1:
        raise click.UsageError("--data-sha256 requires exactly one --data-uri")
    name = user_owned_name(name or selected.output_name)
    job_timeout_hours = job_timeout_hours or selected.execution.job_timeout_seconds // 3600
    version = resolve_version(name, None)
    model = ArtifactHfModel(
        step=ArtifactStep.adopt(f"{name}/inputs/model", input_version, source=model_uri, kind=LevanterCheckpoint),
        relative_path="",
        tokenizer_uri=tokenizer_uri or model_uri,
        tokenizer_revision=tokenizer_revision,
    )
    train_data = tuple(
        ArtifactDataSource(
            ArtifactStep.adopt(
                (
                    user_owned_name(f"{selected.data_namespace}/{data_sha256}")
                    if selected.data_namespace
                    else f"{name}/inputs/data-{index}" + (f"/{data_sha256}" if data_sha256 else "")
                ),
                input_version,
                source=uri,
            ),
            relative_path=data_file,
        )
        for index, uri in enumerate(data_uri)
    )
    resume = ArtifactStep.adopt(f"{name}/inputs/resume", version, source=resume_uri) if resume_uri is not None else None
    validation_data = tuple(
        ArtifactDataSource(
            ArtifactStep.adopt(f"{name}/inputs/validation-{index}", input_version, source=uri),
            relative_path=validation_data_file,
        )
        for index, uri in enumerate(validation_data_uri)
    )
    step = build_run(
        profile=selected,
        name=name,
        version=version,
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        smoke=smoke,
        batch_groups=batch_groups,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        data_sha256=data_sha256,
        resume=resume,
        execution=replace(
            selected.execution,
            job_timeout_seconds=job_timeout_hours * 3600,
            coordinator_timeout_hours=max(
                selected.execution.coordinator_timeout_hours, job_timeout_hours + COORDINATOR_QUEUE_ALLOWANCE_HOURS
            ),
        ),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=checkpoint_count, temporary_storage_ttl_days=ttl_days),
    )
    if config_output is not None and get_job_info() is None:
        config_output.write_text(materialized_config(step, marin_prefix()).launch_config_yaml)
    return step


if __name__ == "__main__":
    main()
