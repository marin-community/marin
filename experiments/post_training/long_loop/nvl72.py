# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVL72 role profiles for the shared long-loop artifact graph."""

import hashlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.namespacing import user_owned_name
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
from skyrlrecipe import ContextBudget, RecipePatch, SkyRLRecipe, sections
from skyrlrecipe.locality import RoleTopologyMode

RECIPE_PATH = Path(__file__).with_name("nvl72-rl.yaml")
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
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=-1, temporary_storage_ttl_days=14)


@dataclass(frozen=True)
class MochiProfile:
    """Resource and recipe settings, with the CLI's default output artifact name."""

    output_name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry
    execution: IrisSkyRLExecution
    patch: RecipePatch = field(default_factory=RecipePatch)
    data_namespace: str | None = None


PROFILES = MappingProxyType(
    {
        "one-rack": MochiProfile("checkpoints/mochi-rl-65k", ROLE_PLAN, MEGATRON, EXECUTION),
        "three-rack": MochiProfile(
            output_name="checkpoints/mochi-three-rack",
            role_plan=replace(
                ROLE_PLAN,
                policy_num_nodes=16,
                num_inference_engines=2,
                inference_engine_data_parallel_size=64,
                inference_engine_expert_parallel_size=64,
            ),
            megatron=replace(MEGATRON, pipeline_model_parallel_size=16),
            execution=EXECUTION,
            data_namespace="documents/mochi-three-rack",
            patch=RecipePatch(
                context_budget=ContextBudget(
                    request_window_tokens=262144, max_new_tokens_per_turn=DEFAULT_MAX_NEW_TOKENS, max_turns=128
                ),
                trainer=sections.Trainer(
                    project_name="mochi-three-rack",
                    policy=sections.Policy(
                        megatron_config=sections.PolicyMegatronConfig(
                            transformer_config_kwargs={
                                "bias_activation_fusion": True,
                                "grug_checkpoint_cpu_offload": True,
                            }
                        ),
                    ),
                    ref=sections.Ref(
                        megatron_config=sections.RefMegatronConfig(
                            transformer_config_kwargs={
                                "bias_activation_fusion": True,
                                "grug_checkpoint_cpu_offload": True,
                            }
                        ),
                    ),
                ),
                generator=sections.Generator(
                    expert_block_sync=sections.ExpertBlockSync(verify=True, timeout_seconds=1800),
                    # The 0.75 KV budget missed the full-context startup requirement by 0.47 GiB.
                    gpu_memory_utilization=0.80,
                    max_num_seqs=64,
                    trajectory_retention=sections.TrajectoryRetention(
                        max_bytes_per_step=256 * 1024**2, max_bytes_per_run=1024**3
                    ),
                ),
                extra_env={
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,NET",
                },
            ),
        ),
    }
)


SNOWBALL_ONE_RACK = replace(
    PROFILES["one-rack"],
    output_name="checkpoints/snowball-nvl72-smoke",
    role_plan=replace(
        ROLE_PLAN,
        policy_num_nodes=8,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    ),
    megatron=replace(MEGATRON, pipeline_model_parallel_size=2),
)
SNOWBALL_THREE_RACK = replace(
    PROFILES["three-rack"],
    output_name="checkpoints/snowball-three-rack-smoke",
    megatron=replace(MEGATRON, pipeline_model_parallel_size=2),
    data_namespace="documents/snowball-three-rack",
)
PROFILES = MappingProxyType(
    {
        **PROFILES,
        "snowball-one-rack": SNOWBALL_ONE_RACK,
        "snowball-three-rack": SNOWBALL_THREE_RACK,
    }
)


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
    if profile.output_name.startswith("checkpoints/snowball") and not smoke:
        raise ValueError("Snowball NVL72 profiles are bounded parity smokes")
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
    if not smoke and checkpoint_interval is not None:
        raise ValueError("Real NVL72 runs use two-hour periodic checkpoints and a final checkpoint")
    checkpoint_interval = (checkpoint_interval or 1) if smoke else -1
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
        if smoke and callback.type == "checkpoint":
            callback = callback.replace(save_interval_seconds=None)
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


def profile_run(
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
            coordinator_name=f"mochi-{profile}-{version}",
            job_timeout_seconds=job_timeout_hours * 3600,
            coordinator_timeout_hours=max(
                selected.execution.coordinator_timeout_hours, job_timeout_hours + COORDINATOR_QUEUE_ALLOWANCE_HOURS
            ),
        ),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=checkpoint_count, temporary_storage_ttl_days=ttl_days),
    )
    if config_output is not None:
        config_output.write_text(materialized_config(step, marin_prefix()).launch_config_yaml)
    return step


@dataclass(frozen=True)
class NVL72RunSpec:
    format: str
    marin_revision: str
    output_prefix: str
    profile: str
    model_uri: str
    tokenizer_revision: str
    data_uri: tuple[str, ...]
    data_sha256: str
    input_version: str
    max_steps: int
    tokenizer_uri: str | None = None
    data_file: str = "train.parquet"
    name: str | None = None
    job_timeout_hours: int = 2
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    resume_uri: str | None = None
    checkpoint_count: int = RETENTION.resume_checkpoint_count
    ttl_days: int = RETENTION.temporary_storage_ttl_days
    batch_groups: int | None = None
    eval_interval: int | None = None
    checkpoint_interval: int | None = None
    validation_data_uri: tuple[str, ...] = ()
    validation_data_file: str = "validation.parquet"


def run_from_spec(spec: NVL72RunSpec, *, smoke: bool, config_output: Path) -> ArtifactStep[SkyRLRun]:
    """Bind pinned input artifacts to the selected NVL72 profile."""
    if spec.format != "nvl72_rl":
        raise ValueError("Expected nvl72_rl run format")
    if spec.profile not in PROFILES:
        raise ValueError(f"Unknown NVL72 profile: {spec.profile}")
    if len(spec.marin_revision) != 40 or any(c not in "0123456789abcdef" for c in spec.marin_revision):
        raise ValueError("marin_revision must be a full Git commit")
    if len(spec.data_sha256) != 64 or any(c not in "0123456789abcdef" for c in spec.data_sha256):
        raise ValueError("data_sha256 must be a SHA-256 digest")
    args = asdict(spec)
    for key in ("format", "marin_revision", "output_prefix"):
        args.pop(key)
    return profile_run(**args, smoke=smoke, config_output=config_output)


def validate_model_geometry(config: dict, profile: MochiProfile) -> None:
    """Reject model dimensions that cannot partition over this role geometry."""
    dimensions = {
        "layers": (config["num_hidden_layers"], profile.megatron.pipeline_model_parallel_size),
        "attention heads": (config["num_attention_heads"], profile.megatron.tensor_model_parallel_size),
        "experts on trainer": (config["num_experts"], profile.megatron.expert_model_parallel_size),
        "experts on inference": (config["num_experts"], profile.role_plan.inference_engine_expert_parallel_size),
    }
    for label, (count, partitions) in dimensions.items():
        if count % partitions:
            raise ValueError(f"Cannot partition {count} {label} across {partitions} ranks")
