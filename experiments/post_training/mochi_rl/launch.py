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
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.declared import CheckpointCallback
from skyrlrecipe.locality import RoleTopologyMode

RECIPE_PATH = Path(__file__).with_name("recipe.yaml")
DEFAULT_MAX_NEW_TOKENS = 8192
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
    train_batch_size=2,
    policy_mini_batch_size=2,
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
    coordinator_timeout_hours=72,
    job_timeout_seconds=24 * 60 * 60,
    wandb_entity="marin-community",
    required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY", "MARIN_PREFIX"),
    container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    role_topology=RoleTopologyMode.NVLINK_DOMAIN,
)
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3)


@dataclass(frozen=True)
class MochiProfile:
    name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry
    execution: IrisSkyRLExecution
    patch: RecipePatch = field(default_factory=RecipePatch)
    data_namespace: str | None = None
    checkpoint_at_target: bool = False


PROFILES = MappingProxyType({"one-rack": MochiProfile("checkpoints/mochi-rl-65k", ROLE_PLAN, MEGATRON, EXECUTION)})


def build_run(
    *,
    profile: MochiProfile,
    name: str,
    version: str,
    model: ArtifactHfModel,
    train_data: tuple[ArtifactDataSource, ...],
    max_steps: int,
    resume: ArtifactStep[Artifact] | None = None,
    recipe_path: Path = RECIPE_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    data_sha256: str | None = None,
    execution: IrisSkyRLExecution | None = None,
    retention: SkyRLRetentionPolicy = RETENTION,
) -> ArtifactStep[SkyRLRun]:
    """Bind model, mixed tasks and optional native state to one RL artifact."""
    recipe = SkyRLRecipe.from_document(yaml.safe_load(recipe_path.read_text())).merge(profile.patch)
    if profile.checkpoint_at_target:
        recipe = recipe.merge(
            RecipePatch(
                trainer=sections.Trainer(
                    ckpt_interval=max_steps,
                    callbacks=(CheckpointCallback(type="checkpoint", save_steps=max_steps, save_on_train_end=True),),
                )
            )
        )
    recipe = recipe.merge(
        RecipePatch(
            context_budget=recipe.context_budget.replace(max_new_tokens_per_turn=max_new_tokens),
            trainer=sections.Trainer(
                # Leave enough epochs for small task sets to reach the global-step target.
                epochs=max_steps,
                max_steps=max_steps,
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
            validation_data=(),
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
            with (StoragePath(ctx.artifact_path(source.step)) / source.relative_path).open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != data_sha256:
                raise ValueError("The mixed-task Parquet input differs from its pinned SHA-256")
        return original_build_config(ctx)

    return replace(step, build_config=bind_runtime)


@click.command(help=__doc__)
@click.option("--profile", type=click.Choice(tuple(PROFILES)), required=True)
@click.option("--model-uri", required=True, help="Completed schema-3 BF16 HF export with its portable model manifest.")
@click.option("--tokenizer-uri", help="Pinned tokenizer artifact; defaults to the model export.")
@click.option("--tokenizer-revision", required=True)
@click.option(
    "--data-uri", multiple=True, required=True, help="Mixed-task artifact directory. Repeat for several inputs."
)
@click.option("--data-sha256", help="SHA-256 of one data file; checked before launch and included in its identity.")
@click.option("--data-file", default="train.parquet", show_default=True)
@click.option("--input-version", required=True, help="Immutable version identifying the adopted input set.")
@click.option("--name", help="Output name within your user namespace; defaults to the selected profile name.")
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
) -> ArtifactStep[SkyRLRun]:
    selected = PROFILES[profile]
    if selected.data_namespace is not None and data_sha256 is None:
        raise click.UsageError(f"--data-sha256 is required for {profile}")
    if data_sha256 is not None and len(data_uri) != 1:
        raise click.UsageError("--data-sha256 requires exactly one --data-uri")
    name = user_owned_name(name or selected.name)
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
    step = build_run(
        profile=selected,
        name=name,
        version=version,
        model=model,
        train_data=train_data,
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
