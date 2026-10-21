"""Structured Hydra configuration for one SkyRL Iris launch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import posixpath
import tempfile
from typing import Any, Mapping

from omegaconf import MISSING, DictConfig, OmegaConf

from skyrl_train.config.mismatch_probe import validate_mismatch_probe_config
from skyrl_train.config.callbacks import validate_hf_export_config
from skyrl_train.config.objective_spec import validate_objective

from cloud.iris.ray_storage import RaySpillBackend, resolve_ray_spill_target
from cloud.iris.role_plan import derive_num_nodes, derive_policy_export_role_plan, derive_role_plan
from cloud.iris.rl_config_translation import (
    RL_ENTRYPOINTS,
    RLEntrypoint,
    compose_skyrl_config,
    parse_rl_config,
    registered_rl_entrypoint_module,
    training_type_for_entrypoint,
    validate_tp_divides_heads,
)
from cloud.iris.runtime_environment import RuntimeMode, runtime_profile_for_strategy
from marinskyrl.distillation import validate_generation_logprobs
from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from marinskyrl.task_sources import data_source


DEFAULT_DRAFT_MODEL_CACHE_TTL_DAYS = 14


class RunMode(StrEnum):
    TRAIN = "train"
    CHECKPOINT_EXPORT = "checkpoint_export"


class SubmissionMode(StrEnum):
    PREPARE = "prepare"
    DETACH = "detach"
    WAIT = "wait"


@dataclass
class RunConfig:
    """Experiment identity and reproducibility inputs."""

    id: str = MISSING
    attempt_id: str = MISSING
    seed: int = 42
    mode: str = RunMode.TRAIN.value
    submission: str = SubmissionMode.WAIT.value
    export_hf: bool = True


@dataclass
class RuntimeConfig:
    """MarinSkyRL runtime provenance."""

    source_digest: str = MISSING
    profile: str = MISSING
    entrypoint: str = ""
    training_type: str | None = None
    experiments_dir: str = "/app/experiments"
    task_env: dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchModelLocator:
    """Immutable policy and tokenizer locators carried by the launch config."""

    uri: str = MISSING
    identity: str = MISSING
    local_path: str = MISSING
    tokenizer_uri: str = MISSING
    tokenizer_revision: str = MISSING
    chat_template: str | None = None


@dataclass
class IrisAllocationConfig:
    """Explicit whole-node resources checked against SkyRL placement."""

    num_nodes: int = MISSING
    gpus_per_node: int = MISSING
    gpu_variant: str = MISSING
    cpu: float = MISSING
    memory: str = MISSING
    disk: str = MISSING


@dataclass(frozen=True)
class LaunchTopology:
    """Validated physical topology derived from the SkyRL role plan."""

    num_nodes: int
    gpus_per_node: int
    gpu_variant: str


@dataclass
class IrisConfig:
    """Iris routing, allocation, and retry settings."""

    cluster: str = MISSING
    cluster_config: str = MISSING
    job_name: str = MISSING
    wandb_entity: str | None = None
    allocation: IrisAllocationConfig = field(default_factory=IrisAllocationConfig)
    priority: str = "interactive"
    max_retries: int = 3
    timeout: int = 0
    target_cluster: str | None = None
    parent_cluster_config: str | None = None
    container_profile: str | None = None


@dataclass
class RayConfig:
    """Ray bootstrap settings shared by every task replica."""

    spill_backend: str = RaySpillBackend.LOCAL.value
    spill_dir: str = MISSING
    rendezvous_dir: str = MISSING
    log_dir: str = MISSING
    rendezvous_timeout: int = 1800
    cluster_join_timeout: int = 1800
    driver_liveness_timeout: int = 9000


@dataclass
class ArtifactsConfig:
    """Durable output roots and manifests for the launch."""

    checkpoint_root: str = MISSING
    export_root: str = MISSING
    attempts_root: str = MISSING
    resolved_config_uri: str = MISSING
    terminal_manifest_uri: str = MISSING
    resume_checkpoint_count: int = 2
    draft_model_cache_ttl_days: int = DEFAULT_DRAFT_MODEL_CACHE_TTL_DAYS


@dataclass
class InputsConfig:
    """Immutable model and data locators resolved by Marin."""

    model: LaunchModelLocator = field(default_factory=LaunchModelLocator)
    data_kind: str = "tasks"
    train_data: list[dict[str, Any]] = field(default_factory=list)
    validation_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkyRLLaunchConfig:
    """Source or resolved launch document handed from Marin to MarinSkyRL."""

    schema_version: int = MISSING
    run: RunConfig = field(default_factory=RunConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    iris: IrisConfig = field(default_factory=IrisConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    inputs: InputsConfig = field(default_factory=InputsConfig)
    skyrl: dict[str, Any] = field(default_factory=dict)


def compose_launch_config(raw: Mapping[str, Any] | DictConfig) -> DictConfig:
    """Compose a raw mapping into the strict structured launch schema."""
    schema = OmegaConf.structured(SkyRLLaunchConfig)
    composed = OmegaConf.merge(schema, raw)
    OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    return composed


def _is_source_recipe(skyrl: DictConfig) -> bool:
    return "context_budget" in skyrl or "config_groups" in skyrl or "defaults" in skyrl


def _compose_source_recipe(config: DictConfig) -> DictConfig:
    raw_skyrl = OmegaConf.to_container(config.skyrl, resolve=False)
    if not isinstance(raw_skyrl, dict):
        raise TypeError("skyrl must be a mapping")
    model_uri = str(config.inputs.model.uri)
    model_identity = str(config.inputs.model.identity)
    model_is_cloud = is_cloud_uri(model_uri)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as source_file:
        OmegaConf.save(OmegaConf.create(raw_skyrl), source_file.name, resolve=False)
        parsed = parse_rl_config(source_file.name)
        compiled = compose_skyrl_config(
            parsed,
            {
                "job_name": str(config.iris.job_name),
                "experiments_dir": str(config.runtime.experiments_dir),
                "num_nodes": int(config.iris.allocation.num_nodes),
                "gpus_per_node": int(config.iris.allocation.gpus_per_node),
                "model_path": str(config.inputs.model.local_path),
                "model_source_uri": model_uri if model_is_cloud else None,
                "model_source_identity": model_identity if model_is_cloud else None,
                "model_revision": model_identity,
                "train_data": list(config.inputs.train_data),
                "val_data": list(config.inputs.validation_data),
                "checkpoint_root": str(config.artifacts.checkpoint_root),
                "export_root": str(config.artifacts.export_root),
                "resume_checkpoint_count": int(config.artifacts.resume_checkpoint_count),
                "trajectory_root": join_resource_path(str(config.artifacts.attempts_root), "trajectories"),
                "export_hf_artifact": bool(config.run.export_hf),
                "seed": int(config.run.seed),
            },
            config.iris.allocation,
        )
        OmegaConf.resolve(compiled.config)
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    OmegaConf.set_struct(resolved, False)
    resolved.runtime.entrypoint = compiled.entrypoint
    training_type = training_type_for_entrypoint(
        compiled.entrypoint, max_staleness_steps=compiled.config.trainer.rollout_buffer.max_staleness_steps
    )
    resolved.runtime.training_type = None if training_type is None else training_type.value
    resolved.inputs.data_kind = parsed.data_kind
    resolved.skyrl = compiled.config
    return compose_launch_config(resolved)


def load_launch_config(path: Path) -> DictConfig:
    """Load and validate one complete SkyRL launch document."""
    config = compose_launch_config(OmegaConf.load(path))
    if _is_source_recipe(config.skyrl):
        if config.run.mode != RunMode.TRAIN:
            raise ValueError("checkpoint_export launch configs must already contain a composed SkyRL subtree")
        config = _compose_source_recipe(config)
    probe = config.skyrl.get("trainer", {}).get("mismatch_probe", {})
    if probe.get("enabled") and not probe.get("archive_uri"):
        artifact_root = posixpath.dirname(str(config.artifacts.resolved_config_uri))
        probe.archive_uri = join_resource_path(artifact_root, "mismatch_probe")
    validate_launch_config(config)
    return config


def _resolved_config(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict):
        raise TypeError("SkyRL launch config must be a mapping")
    return value


def _validate_inputs(inputs: dict[str, Any]) -> None:
    model = inputs.get("model")
    if not isinstance(model, dict):
        raise TypeError("inputs.model must be a mapping")
    try:
        LaunchModelLocator(**model)
    except TypeError as error:
        raise ValueError(f"inputs.model is invalid: {error}") from error
    for name in ("train_data", "validation_data"):
        values = inputs.get(name)
        if not isinstance(values, list):
            raise TypeError(f"inputs.{name} must be a list")
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise TypeError(f"inputs.{name}[{index}] must be a mapping")
            data_source(value)


def validate_iris_allocation(config: dict[str, Any]) -> IrisAllocationConfig:
    """Validate explicit Iris resources against the launch mode's active roles."""
    skyrl = config["skyrl"]
    if not isinstance(skyrl, dict):
        raise TypeError("skyrl must be a mapping")
    checkpoint_export = config["run"]["mode"] == RunMode.CHECKPOINT_EXPORT
    plan = derive_policy_export_role_plan(skyrl) if checkpoint_export else derive_role_plan(skyrl)
    allocation = config["iris"]["allocation"]
    policy = plan.claim("policy")
    expected_nodes = derive_num_nodes(plan)
    if allocation["num_nodes"] != expected_nodes:
        raise ValueError(
            f"iris.allocation.num_nodes={allocation['num_nodes']} does not match SkyRL role plan's "
            f"{expected_nodes} physical nodes"
        )
    gpu_mismatch = (
        allocation["gpus_per_node"] < policy.gpus_per_node
        if checkpoint_export
        else allocation["gpus_per_node"] != policy.gpus_per_node
    )
    if gpu_mismatch:
        raise ValueError(
            f"iris.allocation.gpus_per_node={allocation['gpus_per_node']} does not match SkyRL policy "
            f"placement ({policy.gpus_per_node})"
        )
    return IrisAllocationConfig(**allocation)


def validate_launch_config(config: DictConfig) -> LaunchTopology:
    """Validate launch semantics before an Iris job can be submitted."""
    raw = _resolved_config(config)
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported SkyRL launch schema_version: {raw['schema_version']!r}")
    if raw["run"]["mode"] not in set(RunMode):
        raise ValueError(f"unsupported run.mode: {raw['run']['mode']!r}")
    if raw["run"]["submission"] not in set(SubmissionMode):
        raise ValueError(f"unsupported run.submission: {raw['run']['submission']!r}")
    if raw["inputs"]["data_kind"] not in {"tasks", "parquet"}:
        raise ValueError(f"unsupported inputs.data_kind: {raw['inputs']['data_kind']!r}")
    if raw["iris"]["timeout"] < 0:
        raise ValueError("iris.timeout cannot be negative")
    _validate_inputs(raw["inputs"])
    ray = raw["ray"]
    if (
        ray["rendezvous_timeout"] <= 0
        or ray["cluster_join_timeout"] <= 0
        or ray["driver_liveness_timeout"] < 0
    ):
        raise ValueError(
            "ray rendezvous timeout and cluster join timeout must be positive; "
            "driver liveness timeout cannot be negative"
        )
    resolve_ray_spill_target(
        ray["rendezvous_dir"],
        RaySpillBackend(ray["spill_backend"]),
        ray["spill_dir"],
    )
    allocation = validate_iris_allocation(raw)
    skyrl = raw["skyrl"]
    run = raw["run"]
    if run["mode"] == RunMode.TRAIN:
        validate_objective(config.skyrl)
        validate_hf_export_config(config.skyrl)
    runtime = raw["runtime"]
    entrypoint = runtime["entrypoint"]
    registered_rl_entrypoint_module(entrypoint)
    validate_generation_logprobs(config.skyrl)
    validate_mismatch_probe_config(
        skyrl,
        synchronous=entrypoint == RL_ENTRYPOINTS[RLEntrypoint.STANDARD] and run["mode"] == RunMode.TRAIN,
    )
    expected_profile = runtime_profile_for_strategy(
        skyrl.get("trainer", {}).get("strategy"),
        mode=RuntimeMode.CHECKPOINT_EXPORT if run["mode"] == RunMode.CHECKPOINT_EXPORT else RuntimeMode.TRAINING,
    )
    if runtime["profile"] != expected_profile.value:
        raise ValueError(
            f"runtime.profile={runtime['profile']!r} does not match SkyRL trainer strategy ({expected_profile.value!r})"
        )
    generator = skyrl.get("generator", {})
    validate_tp_divides_heads(
        int(generator["inference_engine_tensor_parallel_size"]),
        skyrl.get("model_num_attention_heads"),
    )
    trainer_seed = skyrl.get("trainer", {}).get("seed")
    if trainer_seed != run["seed"]:
        raise ValueError(f"run.seed={run['seed']} does not match skyrl.trainer.seed={trainer_seed!r}")
    return LaunchTopology(
        num_nodes=allocation.num_nodes,
        gpus_per_node=allocation.gpus_per_node,
        gpu_variant=allocation.gpu_variant,
    )
