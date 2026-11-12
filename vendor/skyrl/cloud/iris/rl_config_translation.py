"""Parse Marin recipes and compose the SkyRL subtree of a launch config."""

from __future__ import annotations

import base64
import binascii
import copy
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Protocol

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from skyrl_train.config.objective_spec import validate_objective
from skyrl_train.rollouts.harbor_settings import concurrent_trials
from skyrlrecipe import ContextBudget, RLEntrypoint, SkyRLRecipe
from skyrlrecipe.documents import set_path

from cloud.iris.paths import resolve_paths_in_dict
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT as CHECKPOINT_EXPORT_MODULE
from marinskyrl.distillation import DistillationPlan, compile_distillation_plan, validate_distillation_runtime_support
from marinskyrl.entrypoints import RL_ENTRYPOINTS
from marinskyrl.environment_contract import TrainingType
from marinskyrl.resource_locator import join_resource_path, model_source_for_path
from marinskyrl.rollout_grading import validate_nemotron_ultra_grading
from marinskyrl.speculative_decoding import parse_speculative_decoding_config

RL_CONFIG_TASK_DIR = "/tmp/marin-rl-configs"
RL_CONFIG_PAYLOAD_ENV = "MARIN_RL_CONFIG_B64"
RESOLVED_HARBOR_CONFIG_KEY = "terminal_bench_config"
# Prompt groups generating at once, in training batches, for a recipe that leaves rollout_buffer.max_in_flight null.
IN_FLIGHT_BATCHES = 1.5
TRAINER_NON_PATH_KEYS = frozenset(
    {
        "policy.model.path",
        "callbacks.metric_groups",
        "callbacks.selection_metric",
        "callbacks.additional_evaluations",
        # Harbor resolves task paths against the pinned dataset repository, not the launcher installation.
        "callbacks.cohorts.job.tasks.path",
    }
)


CHECKPOINT_EXPORT_ENTRYPOINT = CHECKPOINT_EXPORT_MODULE


_RL_ENTRYPOINTS_BY_MODULE = MappingProxyType({module: name for name, module in RL_ENTRYPOINTS.items()})


def training_type_for_entrypoint(module: str, *, max_staleness_steps: int) -> TrainingType | None:
    """How an entrypoint module trains, or None for a module that trains nothing.

    Every training entrypoint runs the rollout-buffer loop, which is synchronous at staleness 0.
    """
    entrypoint = _RL_ENTRYPOINTS_BY_MODULE.get(module)
    if entrypoint is None or entrypoint in (RLEntrypoint.GENERATE, RLEntrypoint.TERMINAL_BENCH_GENERATE):
        return None
    return TrainingType.SYNC if max_staleness_steps == 0 else TrainingType.ASYNC


def registered_rl_entrypoint_module(module: str) -> str:
    """Validate and return one registered direct-call module."""
    if module not in (*RL_ENTRYPOINTS.values(), CHECKPOINT_EXPORT_ENTRYPOINT):
        raise ValueError(f"SkyRL entrypoint module is not registered: {module!r}")
    return module


class HPCGeometry(Protocol):
    """Hardware geometry required while translating a launch configuration."""

    gpus_per_node: int


def _materialize_context_budget(raw: Dict[str, Any], budget: ContextBudget) -> Dict[str, Any]:
    """Populate launch-owned token limits from the authored request budget."""
    materialized = copy.deepcopy(raw)
    for path, value in budget.derived_values(materialized.get("terminal_bench")).items():
        set_path(materialized, path, value)
    return materialized


@dataclass
class ParsedRLConfig:
    """Result of parsing an RL configuration YAML file."""

    config_path: Path
    recipe: SkyRLRecipe
    raw: Dict[str, Any]
    entrypoint: str
    distillation_plan: DistillationPlan | None = None
    config_groups: Dict[str, str] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    generator: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    trajectory_runner: Dict[str, Any] = field(default_factory=dict)
    terminal_bench: Optional[Dict[str, Any]] = None


def materialize_launch_config(
    config_path: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Materialize the launcher-forwarded document inside the task container."""
    environment = os.environ if environment is None else environment
    payload = environment.get(RL_CONFIG_PAYLOAD_ENV)
    if payload is None:
        return config_path

    try:
        contents = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"Invalid base64 in {RL_CONFIG_PAYLOAD_ENV}") from error

    destination = Path(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)
    return str(destination)


def parse_rl_config(config_path: str) -> ParsedRLConfig:
    """Validate an authored recipe and prepare its sparse launch sections."""
    path = Path(config_path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: RL config must contain a mapping at the document root")
    if RESOLVED_HARBOR_CONFIG_KEY in raw:
        raise ValueError(
            f"{path}: {RESOLVED_HARBOR_CONFIG_KEY} is a resolved-config key; use terminal_bench in source recipes"
        )

    config_groups = raw.get("config_groups", {})
    if "algorithm_recipe" in config_groups:
        group = _compose_config_groups({"algorithm_recipe": config_groups["algorithm_recipe"]}, config_name=None)
        raw = OmegaConf.to_container(OmegaConf.merge(OmegaConf.to_container(group, resolve=False), raw), resolve=False)
        assert isinstance(raw, dict)
    try:
        recipe = SkyRLRecipe.from_document(raw)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    distillation_plan = compile_distillation_plan(raw)
    entrypoint = RL_ENTRYPOINTS[recipe.entrypoint]
    materialized_raw = _materialize_context_budget(raw, recipe.context_budget)
    trainer = materialized_raw["trainer"]
    generator = materialized_raw["generator"]
    terminal_bench = materialized_raw.get("terminal_bench")
    data = dict(raw.get("data", {}))
    data.pop("kind", None)
    trainer = resolve_paths_in_dict(trainer, skip_keys=TRAINER_NON_PATH_KEYS)
    generator = resolve_paths_in_dict(generator)
    parse_speculative_decoding_config(
        generator.get("speculative_decoding"),
        backend=recipe.generator.backend,
        run_engines_locally=recipe.generator.run_engines_locally,
        entrypoint=entrypoint,
        colocate_all=recipe.trainer.placement.colocate_all,
        num_inference_engines=recipe.generator.num_inference_engines,
        tensor_parallel_size=recipe.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=recipe.generator.inference_engine_pipeline_parallel_size,
        engine_init_kwargs=generator.get("engine_init_kwargs", {}),
        context=f"{path}: generator.speculative_decoding",
    )
    return ParsedRLConfig(
        config_path=path,
        recipe=recipe,
        raw=materialized_raw,
        entrypoint=entrypoint,
        distillation_plan=distillation_plan,
        config_groups=config_groups,
        trainer=trainer,
        generator=generator,
        data=data,
        environment=materialized_raw.get("environment", {}),
        trajectory_runner=raw.get("trajectory_runner", {}),
        terminal_bench=terminal_bench,
    )


def _apply_policy_model_source(trainer: Dict[str, Any], exp_args: Dict[str, Any]) -> str | None:
    """Apply the task-visible policy path and its replayable source identity."""
    model_path = exp_args.get("model_path")
    if not model_path:
        return None
    policy_model = trainer.setdefault("policy", {}).setdefault("model", {})
    policy_model["path"] = model_path
    model_revision = exp_args.get("model_revision")
    if model_revision is not None:
        policy_model["revision"] = model_revision
    model_source = model_source_for_path(
        model_path,
        exp_args.get("model_source_uri"),
        exp_args.get("model_source_identity"),
    )
    if model_source:
        policy_model["source_uri"] = model_source.uri
        policy_model["source_identity"] = model_source.identity
    return model_path


def _role_gpus_per_node(
    placement: Dict[str, Any],
    key: str,
    launch_gpus_per_node: int,
) -> int:
    """Use the allocation width when unset and reject an authored excess."""
    configured = placement.get(key)
    if configured is None:
        return launch_gpus_per_node
    if int(configured) > launch_gpus_per_node:
        raise ValueError(
            f"trainer.placement.{key}={configured} exceeds the available {launch_gpus_per_node} GPUs per node"
        )
    return int(configured)


@dataclass(frozen=True)
class CompiledSkyRLConfig:
    """A registered entrypoint and its fully composed SkyRL configuration."""

    entrypoint: str
    config: DictConfig


@dataclass(frozen=True)
class TaskLocalSkyRLValues:
    """Values resolved only after an Iris task has staged its inputs."""

    train_data: tuple[str, ...]
    validation_data: tuple[str, ...]
    terminal_bench_data: tuple[str, ...]
    policy_model_path: str | None = None
    draft_model_uri: str | None = None


TASK_LOCAL_SKYRL_PATHS = MappingProxyType(
    {
        "train_data": ("data.train_data",),
        "validation_data": ("data.val_data",),
        "terminal_bench_data": ("data.terminal_bench_data",),
        "policy_model_path": ("trainer.policy.model.path", "trainer.ref.model.path"),
        "draft_model_uri": ("generator.speculative_decoding.model.source_uri",),
    }
)


def _data_override(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith("["):
        return value
    return json.loads(value)


def _apply_trajectory_retention_path(generator: Dict[str, Any], experiments_dir: str, job_name: str) -> None:
    retention = dict(generator.get("trajectory_retention", {}))
    if not retention.get("output_path") and experiments_dir and job_name:
        retention["output_path"] = join_resource_path(
            experiments_dir,
            job_name,
            "trace_jobs",
            "training_trajectories",
        )
    if retention:
        generator["trajectory_retention"] = retention


def _skyrl_config_sections(
    parsed: ParsedRLConfig,
    exp_args: Mapping[str, Any],
    hpc: HPCGeometry,
) -> Dict[str, Dict[str, Any]]:
    """Return SkyRL sections after applying launch-time values as data."""
    validate_distillation_runtime_support(parsed.distillation_plan)

    trainer = copy.deepcopy(parsed.trainer)
    generator = copy.deepcopy(parsed.generator)
    data = copy.deepcopy(parsed.data)
    environment = copy.deepcopy(parsed.environment)
    trajectory_runner = copy.deepcopy(parsed.trajectory_runner)
    experiments_dir = str(exp_args.get("experiments_dir", ""))
    job_name = str(exp_args.get("job_name", ""))

    if not trainer.get("run_name") and job_name:
        trainer["run_name"] = job_name
    if exp_args.get("export_root"):
        trainer["export_path"] = exp_args["export_root"]
    elif not trainer.get("export_path") and experiments_dir and job_name:
        trainer["export_path"] = join_resource_path(experiments_dir, job_name, "exports")
    if exp_args.get("checkpoint_root"):
        trainer["ckpt_path"] = exp_args["checkpoint_root"]
    elif not trainer.get("ckpt_path") and experiments_dir and job_name:
        trainer["ckpt_path"] = join_resource_path(experiments_dir, job_name, "checkpoints")
    _apply_trajectory_retention_path(generator, experiments_dir, job_name)
    if exp_args.get("trajectory_root"):
        generator.setdefault("trajectory_retention", {})["output_path"] = exp_args["trajectory_root"]
    if exp_args.get("resume_checkpoint_count") is not None:
        trainer["max_ckpts_to_keep"] = int(exp_args["resume_checkpoint_count"])
    if exp_args.get("seed") is not None:
        trainer["seed"] = int(exp_args["seed"])

    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", hpc.gpus_per_node))
    placement = copy.deepcopy(trainer.get("placement", {}))
    policy_num_nodes = exp_args.get("policy_num_nodes")
    if placement.get("policy_num_nodes") is None:
        placement["policy_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes
    if placement.get("ref_num_nodes") is None:
        placement["ref_num_nodes"] = policy_num_nodes if policy_num_nodes is not None else num_nodes
    placement["policy_num_gpus_per_node"] = _role_gpus_per_node(placement, "policy_num_gpus_per_node", gpus_per_node)
    placement["ref_num_gpus_per_node"] = _role_gpus_per_node(placement, "ref_num_gpus_per_node", gpus_per_node)
    trainer["placement"] = placement

    if generator.get("num_inference_engines") is None:
        generator["num_inference_engines"] = (
            num_nodes * gpus_per_node
        ) // parsed.recipe.generator.inference_engine_tensor_parallel_size
    for key in ("train_data", "val_data"):
        supplied = _data_override(exp_args.get(key))
        if not supplied:
            continue
        authored = data.get(key)
        if authored and authored != supplied:
            raise ValueError(f"data.{key} conflicts with the nonempty launch inputs")
        data[key] = supplied

    model_path = _apply_policy_model_source(trainer, dict(exp_args))
    if model_path:
        generator.setdefault("engine_init_kwargs", {})["served_model_name"] = model_path.rsplit("/", 1)[-1]

    hf_hub_repo_id = exp_args.get("hf_hub_repo_id")
    if exp_args.get("export_hf_artifact") is not None:
        trainer["export_hf_artifact"] = bool(exp_args["export_hf_artifact"])
    if hf_hub_repo_id:
        trainer["hf_hub_repo_id"] = hf_hub_repo_id
    terminal_bench = copy.deepcopy(parsed.terminal_bench)
    if terminal_bench is not None and terminal_bench.get("installed_agent") is not None:
        installed_agent = terminal_bench["installed_agent"]
        if not installed_agent.get("output_path"):
            installed_agent["output_path"] = join_resource_path(
                generator["trajectory_retention"]["output_path"], "installed-agents"
            )

    sections = {
        "trainer": trainer,
        "generator": generator,
        "data": data,
        "environment": environment,
        "trajectory_runner": trajectory_runner,
    }
    for section in ("teachers", "teacher_routing"):
        if parsed.raw.get(section):
            sections[section] = copy.deepcopy(parsed.raw[section])
    if terminal_bench is not None:
        sections[RESOLVED_HARBOR_CONFIG_KEY] = terminal_bench
    return sections


def _merge_config_mapping(config: DictConfig, values: Mapping[str, Any], prefix: str = "") -> None:
    """Merge validated recipe values and launcher settings into the composed config."""
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            if not isinstance(OmegaConf.select(config, path), DictConfig):
                OmegaConf.update(config, path, {}, merge=False, force_add=True)
            _merge_config_mapping(config, value, path)
        else:
            OmegaConf.update(config, path, copy.deepcopy(value), merge=False, force_add=True)


def _compose_config_groups(config_groups: Mapping[str, str], *, config_name: str | None) -> DictConfig:
    config_dir = Path(str(files("skyrl_train.config"))).resolve()
    group_overrides = [f"+{group_name}={config_name}" for group_name, config_name in config_groups.items()]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=group_overrides)
    OmegaConf.set_struct(config, True)
    return config


def _resolve_max_in_flight(config: DictConfig) -> None:
    """Default the groups generating at once to IN_FLIGHT_BATCHES batches, within the run's Harbor sandboxes."""
    rollout_buffer = config.trainer.rollout_buffer
    if rollout_buffer.max_in_flight is not None:
        return
    groups = math.ceil(IN_FLIGHT_BATCHES * config.trainer.train_batch_size)
    if RESOLVED_HARBOR_CONFIG_KEY in config:
        sandboxes = concurrent_trials(config[RESOLVED_HARBOR_CONFIG_KEY].get("harbor", {}))
        groups = min(groups, max(1, sandboxes // config.generator.n_samples_per_prompt))
    rollout_buffer.max_in_flight = groups


def compose_skyrl_config(
    parsed: ParsedRLConfig,
    exp_args: Mapping[str, Any],
    hpc: HPCGeometry,
) -> CompiledSkyRLConfig:
    """Compose the final SkyRL subtree from its config groups and launch values."""
    config = _compose_config_groups(parsed.config_groups, config_name="ppo_base_config")
    _merge_config_mapping(config, _skyrl_config_sections(parsed, exp_args, hpc))
    _resolve_max_in_flight(config)
    validate_nemotron_ultra_grading(config, parsed.distillation_plan)
    validate_objective(config)
    return CompiledSkyRLConfig(
        entrypoint=registered_rl_entrypoint_module(parsed.entrypoint),
        config=config,
    )


def apply_task_local_values(config: DictConfig, values: TaskLocalSkyRLValues) -> DictConfig:
    """Apply the complete allowlisted task-local patch to a SkyRL config."""
    updated = copy.deepcopy(config)
    for field_name, paths in TASK_LOCAL_SKYRL_PATHS.items():
        value = getattr(values, field_name)
        if value is None or isinstance(value, tuple) and not value:
            continue
        for path in paths:
            OmegaConf.update(updated, path, list(value) if isinstance(value, tuple) else value, force_add=False)
    OmegaConf.resolve(updated)
    return updated
