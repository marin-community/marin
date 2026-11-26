# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVL72 role profiles for the shared long-loop artifact graph."""

import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.namespacing import user_owned_name
from marin.rl.curriculum import RlCurriculum
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLCheckpointStorage,
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
from skyrlrecipe import ContextBudget, RecipePatch, RLEntrypoint, SkyRLRecipe, sections
from skyrlrecipe.locality import RoleTopologyMode

from experiments.post_training.long_loop.execution_policy import RlStopping, rl_training_callbacks
from experiments.post_training.long_loop.openresearcher_service import (
    OpenResearcherService,
    retrieval_launch,
    run_with_retrieval,
    service_data,
    service_execution,
    service_recipe,
)

RECIPE_PATH = Path(__file__).with_name("nvl72-rl.yaml")
DEFAULT_MAX_NEW_TOKENS = 16384
SMOKE_BATCH_GROUPS = 2
SEED = 20261020
COORDINATOR_QUEUE_ALLOWANCE_HOURS = 48
SNOWBALL_INFINIBAND_PORTS = "=ibp0:1,ibp1:1,ibp2:1,ibp3:1"
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
RETENTION = SkyRLRetentionPolicy(
    resume_checkpoint_count=-1,
    temporary_storage_ttl_days=14,
    native_checkpoint_storage=SkyRLCheckpointStorage.PERMANENT,
)


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
                num_inference_engines=8,
                inference_engine_data_parallel_size=16,
                inference_engine_expert_parallel_size=16,
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
                    rollout_buffer=sections.RolloutBuffer(max_in_flight=128),
                    policy=sections.Policy(
                        nccl_buffer_size_bytes=262144,
                        megatron_config=sections.PolicyMegatronConfig(
                            transformer_config_kwargs={
                                "bias_activation_fusion": True,
                                "grug_checkpoint_cpu_offload": True,
                            }
                        ),
                    ),
                    ref=sections.Ref(
                        nccl_buffer_size_bytes=262144,
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
                    vllm_attention_backend="FLASHINFER",
                    gpu_memory_utilization=0.90,
                    max_num_seqs=128,
                    max_num_batched_tokens=2048,
                    enable_prefix_caching=False,
                    engine_init_kwargs={
                        "compilation_config": {
                            "mode": 3,
                            "cudagraph_mode": "PIECEWISE",
                            "inductor_compile_config": {
                                "enable_auto_functionalized_v2": False,
                                "combo_kernels": True,
                                "benchmark_combo_kernel": True,
                            },
                        },
                    },
                    trajectory_retention=sections.TrajectoryRetention(
                        max_bytes_per_step=256 * 1024**2, max_bytes_per_run=1024**3
                    ),
                ),
                extra_env={
                    "NCCL_NET": "IB",
                    "NCCL_IB_DISABLE": "0",
                    "NCCL_IB_HCA": "ibp",
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
    patch=RecipePatch(
        # Snowball's five KV heads cannot split across CP4 for a2a. All-gather
        # supports its sliding windows without changing the four-group geometry.
        trainer=sections.Trainer(
            policy=sections.Policy(
                megatron_config=sections.PolicyMegatronConfig(transformer_config_kwargs={"cp_comm_type": "all_gather"})
            ),
            ref=sections.Ref(
                megatron_config=sections.RefMegatronConfig(transformer_config_kwargs={"cp_comm_type": "all_gather"})
            ),
        ),
        extra_env={
            "NCCL_BUFFSIZE": str(sections.Policy().nccl_buffer_size_bytes),
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": SNOWBALL_INFINIBAND_PORTS,
        },
    ),
)
SNOWBALL_THREE_RACK = replace(
    PROFILES["three-rack"],
    output_name="checkpoints/snowball-three-rack-smoke",
    megatron=replace(MEGATRON, pipeline_model_parallel_size=2),
    data_namespace="documents/snowball-three-rack",
    # The measured hosts expose both IB and RoCE ports. Keep the smoke on IB;
    # the all-GPU launch gate checks the new allocation before model loading.
    patch=PROFILES["three-rack"].patch.merge(
        RecipePatch(
            trainer=SNOWBALL_ONE_RACK.patch.trainer,
            extra_env={
                "NCCL_BUFFSIZE": str(sections.Policy().nccl_buffer_size_bytes),
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
                "NCCL_IB_HCA": SNOWBALL_INFINIBAND_PORTS,
            },
        )
    ),
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
    recipe_document: dict | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    data_sha256: str | None = None,
    execution: IrisSkyRLExecution | None = None,
    retention: SkyRLRetentionPolicy = RETENTION,
    export_hf: bool = False,
    openresearcher_service: OpenResearcherService | None = None,
    gpu_variant: str = "GB200",
    curriculum: RlCurriculum | None = None,
    stopping: RlStopping = RlStopping.WALL_TIME,
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
    if not smoke and curriculum is None:
        raise ValueError(
            "A long-loop RL round trains only against a fixed curriculum: its holdout is the "
            "validation set and its mixture derives the rollout buffer's route quotas; a launch "
            "without one would evaluate a hand-composed validation parquet"
        )
    if profile.data_namespace is not None and data_sha256 is None:
        raise ValueError(f"{profile.output_name} requires data_sha256")
    if data_sha256 is not None and len(train_data) != 1:
        raise ValueError("data_sha256 requires exactly one training data source")
    if openresearcher_service is not None and data_sha256 != openresearcher_service.parquet_sha256:
        raise ValueError("Service-backed OpenResearcher data must match the pinned variant Parquet digest")
    geometry = profile.megatron
    policy_data_parallel_size = (profile.role_plan.policy_num_nodes * profile.role_plan.policy_num_gpus_per_node) // (
        geometry.tensor_model_parallel_size * geometry.pipeline_model_parallel_size * geometry.context_parallel_size
    )
    smoke_groups = math.lcm(SMOKE_BATCH_GROUPS, policy_data_parallel_size)
    groups = batch_groups if batch_groups is not None else smoke_groups if smoke else profile.role_plan.train_batch_size
    if groups < 2 or groups % 2:
        raise ValueError("Batch groups must be positive and even for equal agentic and nonagentic quotas")
    profile = replace(
        profile, role_plan=replace(profile.role_plan, train_batch_size=groups, policy_mini_batch_size=groups)
    )
    recipe = SkyRLRecipe.from_document(
        recipe_document if recipe_document is not None else yaml.safe_load(recipe_path.read_text())
    ).merge(profile.patch)
    if openresearcher_service is not None:
        recipe = recipe.merge(service_recipe())
        execution = service_execution(execution or profile.execution)
    eval_interval = recipe.trainer.eval_interval if eval_interval is None else eval_interval
    checkpoint_interval = (
        checkpoint_interval if checkpoint_interval is not None else 1 if smoke else recipe.trainer.ckpt_interval
    )
    context = recipe.context_budget.replace(max_new_tokens_per_turn=max_new_tokens)
    callback_updates = {
        "checkpoint": {"save_steps": checkpoint_interval},
        "evaluation": {"eval_steps": eval_interval},
        "best_checkpoint": {"evaluation_interval": eval_interval},
    }
    callbacks = []
    for callback in recipe.trainer.callbacks:
        if smoke and callback.type in ("evaluation", "best_checkpoint"):
            continue
        if smoke and callback.type == "checkpoint":
            callback = callback.replace(save_interval_seconds=None)
        callbacks.append(callback.replace(**callback_updates.get(callback.type, {})))
    budgeted_callbacks = sections.Trainer.model_validate(
        {
            "callbacks": tuple(
                rl_training_callbacks(
                    [callback.to_skyrl() for callback in callbacks], groups_per_step=groups, stopping=stopping
                )
            )
        }
    ).callbacks
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
                callbacks=budgeted_callbacks,
                rollout_buffer=(
                    # A curriculum derives the window's consumed quotas at the spec; a smoke
                    # parity diagnostic has no curriculum and shrinks the equal split.
                    sections.RolloutBuffer(max_in_flight=recipe.trainer.rollout_buffer.max_in_flight)
                    if curriculum is not None
                    else sections.RolloutBuffer(
                        consumed_route_quotas={"agentic": groups // 2, "nonagentic": groups // 2},
                        max_in_flight=SMOKE_BATCH_GROUPS,
                    )
                ),
                micro_forward_batch_size_per_gpu=profile.role_plan.micro_train_batch_size_per_gpu,
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**asdict(profile.megatron))),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**asdict(profile.megatron))),
            ),
        )
    )
    if openresearcher_service is not None:
        buffer = recipe.trainer.rollout_buffer.replace(consumed_route_quotas={}, source_routes={})
        recipe = recipe.replace(trainer=recipe.trainer.replace(rollout_buffer=buffer))
    document = recipe.to_skyrl()
    # Marin derives the entrypoint from the plan; the round YAML's declared value
    # moves onto the role plan instead of staying in the recipe document.
    declared_entrypoint = RLEntrypoint(document.pop("entrypoint", RLEntrypoint.STANDARD.value))
    if declared_entrypoint is RLEntrypoint.GENERATE:
        raise ValueError("RL rounds with a generate entrypoint use the evaluation step instead")
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(document, sort_keys=False),
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            topology=SkyRLTopology(
                num_nodes=profile.role_plan.num_nodes,
                gpus_per_node=profile.role_plan.policy_num_gpus_per_node,
                gpu_variant=gpu_variant,
                role_plan=replace(profile.role_plan, entrypoint=declared_entrypoint.value),
            ),
            retention=retention,
            seed=SEED,
            starting_checkpoint=resume,
            curriculum=curriculum,
        ),
        execution or profile.execution,
        export_hf=export_hf,
    )
    if openresearcher_service is not None:
        launch = retrieval_launch(openresearcher_service, execution or profile.execution)
        return replace(
            step,
            run=partial(
                run_with_retrieval,
                launch=launch,
                parquet_sha256=openresearcher_service.parquet_sha256,
                parquet_relative_path="train.parquet",
            ),
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
    openresearcher_service: OpenResearcherService | None = None,
    validation_data_uri: tuple[str, ...] = (),
    validation_data_file: str = "validation.parquet",
    curriculum: RlCurriculum | None = None,
) -> ArtifactStep[SkyRLRun]:
    selected = PROFILES[profile]
    if not smoke and not validation_data_uri:
        raise click.UsageError("--validation-data-uri is required for normal RL; select --smoke to omit evaluation")
    if not smoke and curriculum is None:
        raise click.UsageError(
            "A long-loop RL round trains only against a fixed curriculum: its holdout is the "
            "validation set and its mixture derives the route quotas; select --smoke to omit evaluation"
        )
    if selected.data_namespace is not None and data_sha256 is None:
        raise click.UsageError(f"--data-sha256 is required for {profile}")
    if data_sha256 is not None and len(data_uri) != 1 and openresearcher_service is None:
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
    if openresearcher_service is not None:
        if data_uri:
            raise ValueError("The OpenResearcher service recipe selects its private HF pin; data_uri must be empty")
        train_data = (service_data(openresearcher_service, version=input_version),)
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
        openresearcher_service=openresearcher_service,
        name=name,
        version=version,
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        smoke=smoke,
        curriculum=curriculum,
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
        retention=replace(RETENTION, resume_checkpoint_count=checkpoint_count, temporary_storage_ttl_days=ttl_days),
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
    curriculum: RlCurriculum | None = None
    checkpoint_interval: int | None = None
    validation_data_uri: tuple[str, ...] = ()
    validation_data_file: str = "validation.parquet"
    openresearcher_service: OpenResearcherService | None = None


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
    curriculum = spec.curriculum
    args = asdict(spec)
    args["openresearcher_service"] = spec.openresearcher_service
    for key in ("format", "marin_revision", "output_prefix"):
        args.pop(key)
    args.pop("curriculum")
    return profile_run(**args, smoke=smoke, config_output=config_output, curriculum=curriculum)


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
