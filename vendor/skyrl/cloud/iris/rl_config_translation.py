"""Parse Marin recipes and compose the SkyRL subtree of a launch config."""

from __future__ import annotations

import base64
import binascii
import copy
from importlib.resources import files
import json
import math
import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from skyrl_train.config.objective_spec import validate_objective
from skyrl_train.config.ftpo import FTPOConfig

from cloud.iris.paths import resolve_paths_in_dict
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT as CHECKPOINT_EXPORT_MODULE
from marinskyrl.environment_contract import TrainingType
from marinskyrl.distillation import DistillationPlan, compile_distillation_plan, validate_distillation_runtime_support
from marinskyrl.resource_locator import join_resource_path, model_source_for_path
from marinskyrl.speculative_decoding import STANDARD_TRAINING_ENTRYPOINT, parse_speculative_decoding_config
from marinskyrl.remote_io import filesystem_and_path, open_output_stream
from marinskyrl.rollout_grading import validate_nemotron_ultra_grading

# Directory containing the bundled example RL config YAML files.
SKYRL_CONFIG_DIR = Path(__file__).parent / "configs"
RL_CONFIG_TASK_DIR = "/tmp/marin-rl-configs"
RL_CONFIG_PAYLOAD_ENV = "MARIN_RL_CONFIG_B64"
RESOLVED_HARBOR_CONFIG_KEY = "terminal_bench_config"
TRAINER_NON_PATH_KEYS = frozenset(
    {
        "policy.model.path",
        "callbacks.metric_groups",
        "callbacks.additional_evaluations",
        # Harbor resolves task paths against the pinned dataset repository, not the launcher installation.
        "callbacks.cohorts.job.tasks.path",
    }
)


class RLEntrypoint(StrEnum):
    """Execution modes supported by Iris RL configurations."""

    GENERATE = "generate"
    STANDARD = "standard"
    TASKCOMPENDIUM = "taskcompendium"
    TERMINAL_BENCH = "terminal_bench"
    TERMINAL_BENCH_GENERATE = "terminal_bench_generate"


RL_ENTRYPOINTS = MappingProxyType(
    {
        RLEntrypoint.GENERATE: "skyrl_train.entrypoints.main_generate",
        RLEntrypoint.STANDARD: STANDARD_TRAINING_ENTRYPOINT,
        RLEntrypoint.TASKCOMPENDIUM: "skyrl_train.entrypoints.taskcompendium",
        RLEntrypoint.TERMINAL_BENCH: "skyrl_train.entrypoints.terminal_bench",
        RLEntrypoint.TERMINAL_BENCH_GENERATE: "skyrl_train.entrypoints.terminal_bench_generate",
    }
)
CHECKPOINT_EXPORT_ENTRYPOINT = CHECKPOINT_EXPORT_MODULE


def resolve_rl_entrypoint(value: str | None, *, config_path: Path) -> str:
    """Resolve one supported RL execution mode to its packaged module."""
    name = RLEntrypoint.STANDARD if value is None else value
    try:
        entrypoint = RLEntrypoint(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in RLEntrypoint)
        raise ValueError(
            f"{config_path}: entrypoint must be a registered name ({choices}); got {name!r}. "
            "Python module paths are not accepted in RL configs."
        ) from error

    return RL_ENTRYPOINTS[entrypoint]


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


_REQUIRED_CONTEXT_BUDGET_FIELDS = frozenset(
    {
        "request_window_tokens",
        "max_new_tokens_per_turn",
        "max_turns",
    }
)
_CONTEXT_BUDGET_FRACTION_FIELDS = frozenset({"generated_budget_fraction", "overlong_cache_fraction"})
_CONTEXT_BUDGET_FIELDS = _REQUIRED_CONTEXT_BUDGET_FIELDS | _CONTEXT_BUDGET_FRACTION_FIELDS | {"serving_window_tokens"}
_DEFAULT_GENERATED_BUDGET_FRACTION = 0.5
_DEFAULT_OVERLONG_CACHE_FRACTION = 0.25

_DERIVED_CONTEXT_FIELDS = (
    ("trainer", "max_prompt_length"),
    ("generator", "max_input_length"),
    ("generator", "max_turns"),
    ("generator", "sampling_params", "max_generate_length"),
    ("generator", "engine_init_kwargs", "max_model_len"),
    ("terminal_bench", "harbor", "max_turns"),
    ("generator", "trajectory_reward_shaping", "overlong", "l_max"),
    ("generator", "trajectory_reward_shaping", "overlong", "l_cache"),
)


@dataclass(frozen=True)
class ContextBudget:
    """One coherent token budget for an Iris RL rollout request."""

    request_window_tokens: int
    max_new_tokens_per_turn: int
    max_turns: int
    serving_window_tokens: int
    generated_budget_fraction: float = _DEFAULT_GENERATED_BUDGET_FRACTION
    overlong_cache_fraction: float = _DEFAULT_OVERLONG_CACHE_FRACTION

    @property
    def max_input_tokens(self) -> int:
        """Return the input allowance after reserving one complete response."""
        return self.request_window_tokens - self.max_new_tokens_per_turn

    @property
    def generated_tokens_per_trajectory(self) -> int:
        """Return the generated-token allowance used by trajectory-level shaping."""
        if self.max_turns == 1:
            return self.max_new_tokens_per_turn
        return max(1, int(self.request_window_tokens * self.generated_budget_fraction))

    @property
    def overlong_cache_tokens(self) -> int:
        """Return the soft-overlong transition width."""
        return int(self.generated_tokens_per_trajectory * self.overlong_cache_fraction)

    def as_dict(self) -> Dict[str, int | float]:
        """Return the persisted representation, including derived client input."""
        return {
            "request_window_tokens": self.request_window_tokens,
            "serving_window_tokens": self.serving_window_tokens,
            "max_new_tokens_per_turn": self.max_new_tokens_per_turn,
            "max_turns": self.max_turns,
            "generated_budget_fraction": self.generated_budget_fraction,
            "overlong_cache_fraction": self.overlong_cache_fraction,
            "max_input_tokens": self.max_input_tokens,
            "generated_tokens_per_trajectory": self.generated_tokens_per_trajectory,
            "overlong_cache_tokens": self.overlong_cache_tokens,
        }


def _path_is_declared(mapping: Dict[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _validate_no_derived_context_fields(raw: Dict[str, Any], config_path: Path) -> None:
    declared = [".".join(path) for path in _DERIVED_CONTEXT_FIELDS if _path_is_declared(raw, path)]
    if declared:
        raise ValueError(
            f"{config_path} declares derived context fields: {', '.join(declared)}. "
            "Declare only context_budget instead."
        )


def _require_positive_integer(value: Any, field_name: str, config_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{config_path}: context_budget.{field_name} must be a positive integer, got {value!r}")
    return value


def _require_fraction(value: Any, field_name: str, config_path: Path, *, allow_zero: bool) -> float:
    valid = False
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        lower_bound_satisfied = value >= 0 if allow_zero else value > 0
        valid = math.isfinite(value) and lower_bound_satisfied and value <= 1
    if not valid:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{config_path}: context_budget.{field_name} must be in {interval}, got {value!r}")
    return float(value)


def resolve_context_budget(raw: Dict[str, Any], config_path: Path) -> ContextBudget:
    """Validate and resolve the single public context budget declaration.

    The request window includes the prompt and the current response. The derived
    client input limit therefore reserves the complete per-turn output allowance.
    """
    _validate_no_derived_context_fields(raw, config_path)
    config = raw.get("context_budget")
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: context_budget must be a mapping")

    unknown = set(config) - _CONTEXT_BUDGET_FIELDS
    if unknown:
        raise ValueError(f"{config_path}: unknown context_budget fields: {', '.join(sorted(unknown))}")
    missing = _REQUIRED_CONTEXT_BUDGET_FIELDS - set(config)
    if missing:
        raise ValueError(f"{config_path}: missing context_budget fields: {', '.join(sorted(missing))}")

    budget = ContextBudget(
        request_window_tokens=_require_positive_integer(
            config["request_window_tokens"], "request_window_tokens", config_path
        ),
        max_new_tokens_per_turn=_require_positive_integer(
            config["max_new_tokens_per_turn"], "max_new_tokens_per_turn", config_path
        ),
        max_turns=_require_positive_integer(config["max_turns"], "max_turns", config_path),
        serving_window_tokens=_require_positive_integer(
            config.get("serving_window_tokens", config["request_window_tokens"]), "serving_window_tokens", config_path
        ),
        generated_budget_fraction=_require_fraction(
            config.get("generated_budget_fraction", _DEFAULT_GENERATED_BUDGET_FRACTION),
            "generated_budget_fraction",
            config_path,
            allow_zero=False,
        ),
        overlong_cache_fraction=_require_fraction(
            config.get("overlong_cache_fraction", _DEFAULT_OVERLONG_CACHE_FRACTION),
            "overlong_cache_fraction",
            config_path,
            allow_zero=True,
        ),
    )
    if budget.max_input_tokens <= 0:
        raise ValueError(
            f"{config_path}: request_window_tokens ({budget.request_window_tokens}) must exceed "
            f"max_new_tokens_per_turn ({budget.max_new_tokens_per_turn})"
        )
    if budget.serving_window_tokens < budget.request_window_tokens:
        raise ValueError(f"{config_path}: serving_window_tokens must cover the complete request_window_tokens")
    return budget


def _materialize_context_budget(
    raw: Dict[str, Any], budget: ContextBudget
) -> tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return SkyRL sections populated from one resolved context budget."""
    trainer = copy.deepcopy(raw.get("trainer", {}))
    generator = copy.deepcopy(raw.get("generator", {}))
    terminal_bench = copy.deepcopy(raw.get("terminal_bench"))
    materialized_raw = copy.deepcopy(raw)

    trainer["max_prompt_length"] = budget.max_input_tokens
    generator["max_input_length"] = budget.max_input_tokens
    generator["max_turns"] = budget.max_turns
    generator.setdefault("sampling_params", {})["max_generate_length"] = budget.max_new_tokens_per_turn
    generator.setdefault("engine_init_kwargs", {})["max_model_len"] = budget.serving_window_tokens
    generator.setdefault("trajectory_reward_shaping", {})["overlong"] = {
        "l_max": budget.generated_tokens_per_trajectory,
        "l_cache": budget.overlong_cache_tokens,
    }

    if terminal_bench is not None:
        harbor = terminal_bench.setdefault("harbor", {})
        harbor["max_turns"] = budget.max_turns

    materialized_raw["context_budget"] = budget.as_dict()
    materialized_raw["trainer"] = copy.deepcopy(trainer)
    materialized_raw["generator"] = copy.deepcopy(generator)
    if terminal_bench is not None:
        materialized_raw["terminal_bench"] = copy.deepcopy(terminal_bench)
    return trainer, generator, terminal_bench, materialized_raw


def write_resolved_context_budget(budget: ContextBudget, destination: Path | str, config_path: Path) -> Path | str:
    """Persist the resolved context contract for an Iris RL launch."""
    payload = (
        json.dumps(
            {
                "config_path": str(config_path),
                "context_budget": budget.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if isinstance(destination, Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
        return destination
    filesystem, path = filesystem_and_path(destination)
    with open_output_stream(filesystem, path) as artifact:
        artifact.write(payload.encode())
    return destination


# =============================================================================
# SkyRL Internal Engine Kwargs - DO NOT SET IN YAML CONFIGS
# =============================================================================
# These kwargs are set internally by SkyRL and will cause "duplicate keyword
# argument" errors if also specified in engine_init_kwargs.
#
# Source: skyrl_train/inference_engines/ray_wrapped_inference_engine.py
# =============================================================================

SKYRL_INTERNAL_ENGINE_KWARGS = frozenset(
    {
        # Hardcoded values
        "trust_remote_code",  # Always True
        "worker_extension_cls",  # vLLM SkyRL extension path
        "data_parallel_backend",  # Hardcoded "mp"
        "max_logprobs",  # Hardcoded 1
        # Calculated from config/environment
        "distributed_executor_backend",  # Calculated from TP size ("uni" or "ray")
        "enforce_eager",  # Set from generator.enforce_eager config
        "tensor_parallel_size",  # Set from generator config
        "data_parallel_size",  # Set from generator config
        "seed",  # Set from config
        "enable_prefix_caching",  # Set from generator config
        "dtype",  # Set from generator.model_dtype
        "gpu_memory_utilization",  # Set from generator config
        "max_num_batched_tokens",  # Set from generator config
        "max_num_seqs",  # Set from generator config
        "enable_sleep_mode",  # Set from trainer.placement.colocate_all
        "vllm_v1_disable_multiproc",  # Set from generator config
        # Ray internal management
        "bundle_indices",  # Calculated from parallelism config
        "num_gpus",  # Ray resource allocation
        "noset_visible_devices",  # Ray CUDA_VISIBLE_DEVICES handling
        # SGLang-specific (if using SGLang backend)
        "model_path",  # Set from trainer.policy.model.path
        "tp_size",  # Alias for tensor_parallel_size
        "mem_fraction_static",  # Alias for gpu_memory_utilization
        "random_seed",  # Alias for seed
        "disable_radix_cache",  # Inverse of enable_prefix_caching
        "max_prefill_tokens",  # Alias for max_num_batched_tokens
        "max_running_requests",  # Alias for max_num_seqs
        "mm_attention_backend",  # Hardcoded "fa3"
        "attention_backend",  # Hardcoded "fa3"
        "enable_memory_saver",  # Set from inference_engine_enable_sleep
        "tokenizer",  # Passed from external tokenizer
        "custom_weight_loader",  # Hardcoded SkyRL path
        "skip_tokenizer_init",  # Hardcoded True for SGLang
        "speculative_config",  # Derived from generator.speculative_decoding
    }
)


def validate_engine_init_kwargs(
    engine_init_kwargs: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> None:
    """Fail fast if ``engine_init_kwargs`` contains SkyRL-internal keys.

    SkyRL sets certain vLLM/SGLang engine kwargs internally; specifying them in
    the YAML config causes "duplicate keyword argument" errors at runtime.

    Raises:
        ValueError: If any forbidden keys are found in engine_init_kwargs.
    """
    if not engine_init_kwargs:
        return

    forbidden_found = set(engine_init_kwargs.keys()) & SKYRL_INTERNAL_ENGINE_KWARGS

    if forbidden_found:
        config_context = f" in {config_path}" if config_path else ""
        forbidden_list = "\n".join(f"  - {k}" for k in sorted(forbidden_found))
        all_forbidden = "\n".join(f"  - {k}" for k in sorted(SKYRL_INTERNAL_ENGINE_KWARGS))

        raise ValueError(
            f"engine_init_kwargs{config_context} contains keys that SkyRL sets internally.\n"
            f"These will cause 'duplicate keyword argument' errors at runtime.\n\n"
            f"FORBIDDEN KEYS FOUND:\n{forbidden_list}\n\n"
            f"Remove these from your config. SkyRL handles them automatically.\n\n"
            f"FULL LIST OF SKYRL-INTERNAL KWARGS (never set these):\n{all_forbidden}\n\n"
            f"SAFE TO SET: kv_cache_dtype, quantization, cpu_offload_gb, etc."
        )


@dataclass
class ParsedRLConfig:
    """Result of parsing an RL configuration YAML file."""

    config_path: Path
    raw: Dict[str, Any]
    context_budget: ContextBudget
    entrypoint: str
    distillation_plan: DistillationPlan | None = None
    config_groups: Dict[str, str] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    generator: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    trajectory_runner: Dict[str, Any] = field(default_factory=dict)
    terminal_bench: Optional[Dict[str, Any]] = None
    tensor_parallel_size: int = 1
    # "tasks" (default; terminal_bench task-dir extraction) or "parquet" (single-turn
    # RLVR: an HF id / .parquet is passed through to PromptDataset, NOT task-extracted).
    # Launcher-only (popped out of the `data` section so it never reaches Hydra).
    data_kind: str = "tasks"


@dataclass(frozen=True)
class ParsedCheckpointExportConfig:
    """Policy configuration needed to reconstruct a checkpoint for conversion."""

    config_path: Path
    config_groups: Dict[str, str]
    trainer: Dict[str, Any]


def validate_tp_divides_heads(
    tensor_parallel_size: int,
    num_attention_heads: Optional[int],
    config_path: Optional[Path] = None,
) -> None:
    """Fail fast if the inference TP size does not divide the model's attention-head count.

    vLLM requires ``num_attention_heads % tensor_parallel_size == 0``; a bad value (e.g.
    TP=8 against the dense delphi arch's 42 heads) wedges vLLM at engine init with no
    launcher-side signal. Skipped when ``num_attention_heads`` is unset (existing configs).

    Raises:
        ValueError: If ``num_attention_heads`` is set and not divisible by TP.
    """
    if not num_attention_heads:
        return
    if num_attention_heads % tensor_parallel_size != 0:
        valid = [t for t in range(1, num_attention_heads + 1) if num_attention_heads % t == 0]
        config_context = f" in {config_path}" if config_path else ""
        raise ValueError(
            f"generator.inference_engine_tensor_parallel_size={tensor_parallel_size} does not "
            f"divide model_num_attention_heads={num_attention_heads}{config_context}.\n"
            f"vLLM requires num_attention_heads % tensor_parallel_size == 0, or the engine "
            f"wedges at init with no launcher signal.\n"
            f"Valid TP values for {num_attention_heads} heads: {valid} (NEVER 8 for delphi's 42)."
        )


def resolve_rl_config_path(raw_path: str) -> Path:
    """Resolve an RL config path, checking the bundled ``configs/`` fallback.

    Resolution order: the path as-is, then ``SKYRL_CONFIG_DIR / raw_path``, then
    ``SKYRL_CONFIG_DIR / raw_path.yaml``.

    Raises:
        FileNotFoundError: If the config file cannot be found in any location.
    """
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()

    fallback = SKYRL_CONFIG_DIR / raw_path
    if fallback.exists():
        return fallback.resolve()

    fallback_yaml = SKYRL_CONFIG_DIR / f"{raw_path}.yaml"
    if fallback_yaml.exists():
        return fallback_yaml.resolve()

    raise FileNotFoundError(
        f"RL config not found: {raw_path}\nSearched: {path}, {SKYRL_CONFIG_DIR / raw_path}, {fallback_yaml}"
    )


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


def load_rl_recipe(config_path: str) -> DictConfig:
    """Compose a source recipe with Hydra defaults before deriving launch settings."""
    path = resolve_rl_config_path(config_path)
    raw = OmegaConf.load(path)
    if "defaults" not in raw:
        return raw
    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        return compose(
            config_name=path.name,
            overrides=[f"hydra.searchpath=[file://{SKYRL_CONFIG_DIR.resolve()}]"],
        )


def parse_rl_config(
    config_path: str,
    model_override: Optional[str] = None,
) -> ParsedRLConfig:
    """Parse an RL config YAML and extract all settings."""
    path = resolve_rl_config_path(config_path)
    raw = OmegaConf.to_container(load_rl_recipe(str(path)), resolve=False) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: RL config must contain a mapping at the document root")
    if RESOLVED_HARBOR_CONFIG_KEY in raw:
        raise ValueError(
            f"{path}: {RESOLVED_HARBOR_CONFIG_KEY} is a resolved-config key; use terminal_bench in source recipes"
        )

    config_groups = raw.get("config_groups", {})
    if "algorithm_recipe" in config_groups:
        recipe = _compose_config_groups({"algorithm_recipe": config_groups["algorithm_recipe"]}, config_name=None)
        raw = OmegaConf.to_container(OmegaConf.merge(OmegaConf.to_container(recipe, resolve=False), raw), resolve=False)
        assert isinstance(raw, dict)
    distillation_plan = compile_distillation_plan(raw)
    context_budget = resolve_context_budget(raw, path)

    entrypoint = resolve_rl_entrypoint(raw.get("entrypoint"), config_path=path)
    trainer, generator, terminal_bench, materialized_raw = _materialize_context_budget(raw, context_budget)
    data = dict(raw.get("data", {}))
    environment = raw.get("environment", {})
    trajectory_runner = raw.get("trajectory_runner", {})
    # data.kind selects launcher staging and is not part of SkyRL's data config.
    data_kind = data.pop("kind", "tasks")

    # Validate engine_init_kwargs doesn't contain SkyRL-internal keys.
    engine_init_kwargs = generator.get("engine_init_kwargs", {})
    validate_engine_init_kwargs(engine_init_kwargs, config_path=path)

    # Resolve relative paths in config sections to absolute paths so they work
    # regardless of the working directory at runtime. Skip data.train_data /
    # data.val_data as they may be HF repo IDs.
    trainer = resolve_paths_in_dict(trainer, skip_keys=TRAINER_NON_PATH_KEYS)
    generator = resolve_paths_in_dict(generator)

    parse_speculative_decoding_config(
        generator.get("speculative_decoding"),
        backend=generator.get("backend", "vllm"),
        run_engines_locally=generator.get("run_engines_locally", True),
        entrypoint=entrypoint,
        colocate_all=trainer.get("placement", {}).get("colocate_all", True),
        num_inference_engines=generator.get("num_inference_engines", 1),
        tensor_parallel_size=generator.get("inference_engine_tensor_parallel_size", 4),
        pipeline_parallel_size=generator.get("inference_engine_pipeline_parallel_size", 1),
        engine_init_kwargs=generator.get("engine_init_kwargs", {}),
        context=f"{path}: generator.speculative_decoding",
    )

    if model_override:
        trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_override

    tensor_parallel_size = generator.get("inference_engine_tensor_parallel_size", 1)

    # TP-divides-heads guard (delphi's 42-head arch forbids TP=8). No-op unless the config
    # declares model_num_attention_heads.
    validate_tp_divides_heads(tensor_parallel_size, raw.get("model_num_attention_heads"), config_path=path)

    return ParsedRLConfig(
        config_path=path,
        raw=materialized_raw,
        context_budget=context_budget,
        entrypoint=entrypoint,
        distillation_plan=distillation_plan,
        config_groups=config_groups,
        trainer=trainer,
        generator=generator,
        data=data,
        environment=environment,
        trajectory_runner=trajectory_runner,
        terminal_bench=terminal_bench,
        tensor_parallel_size=tensor_parallel_size,
        data_kind=data_kind,
    )


def parse_checkpoint_export_config(
    config_path: str,
    model_override: str,
) -> ParsedCheckpointExportConfig:
    """Read policy configuration without validating or materializing rollout settings."""
    path = resolve_rl_config_path(config_path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: RL config must contain a mapping at the document root")

    trainer = resolve_paths_in_dict(copy.deepcopy(raw.get("trainer", {})), skip_keys=TRAINER_NON_PATH_KEYS)
    trainer.setdefault("policy", {}).setdefault("model", {})["path"] = model_override
    return ParsedCheckpointExportConfig(
        config_path=path,
        config_groups=dict(raw.get("config_groups", {})),
        trainer=trainer,
    )


_OPTIONAL_HYDRA_PATTERNS = {
    ".distillation",
    ".domain_weights",
    ".engine_init_kwargs",
    ".chat_template_kwargs",
    ".speculative_decoding",
    ".hf_hub_",
    ".enable_db_registration",
    ".optimizer_kwargs",
    ".rope_scaling",
    ".wrap_policy",
    ".transformer_config_kwargs",
}


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
    configured = placement.get(key)
    if configured is not None and int(configured) <= launch_gpus_per_node:
        return int(configured)
    return launch_gpus_per_node


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


def _checkpoint_export_trainer(
    parsed: ParsedCheckpointExportConfig,
    exp_args: Mapping[str, Any],
    hpc: HPCGeometry,
) -> Dict[str, Any]:
    trainer = copy.deepcopy(parsed.trainer)
    placement = trainer.setdefault("placement", {})
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", hpc.gpus_per_node))
    placement["policy_num_nodes"] = num_nodes
    placement["policy_num_gpus_per_node"] = _role_gpus_per_node(
        placement,
        "policy_num_gpus_per_node",
        gpus_per_node,
    )
    _apply_policy_model_source(trainer, dict(exp_args))
    return trainer


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
        generator["num_inference_engines"] = (num_nodes * gpus_per_node) // parsed.tensor_parallel_size
    if exp_args.get("train_data"):
        data["train_data"] = _data_override(exp_args["train_data"])
    if exp_args.get("val_data"):
        data["val_data"] = _data_override(exp_args["val_data"])

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


_OPEN_CONFIG_ROOTS = frozenset({"teachers", "teacher_routing", RESOLVED_HARBOR_CONFIG_KEY})
_FTPO_CONFIG_PATHS = frozenset(f"trainer.algorithm.ftpo.{item.name}" for item in fields(FTPOConfig))


def _path_allows_new_keys(path: str) -> bool:
    return (
        path in _FTPO_CONFIG_PATHS
        or path.startswith("environment.task_sessions.")
        or path.startswith("generator.reasoning_spans.agent_policies.")
        or path.split(".", 1)[0] in _OPEN_CONFIG_ROOTS
        or any(pattern in f".{path}" for pattern in _OPTIONAL_HYDRA_PATTERNS)
    )


def _merge_config_mapping(config: DictConfig, values: Mapping[str, Any], prefix: str = "") -> None:
    """Merge launch values into declared SkyRL config paths."""
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        current = OmegaConf.select(config, path, default=...)
        key_exists = current is not ...
        if isinstance(value, Mapping):
            if not isinstance(current, DictConfig):
                if key_exists and current is not None and not _path_allows_new_keys(path):
                    raise ValueError(f"SkyRL config key {path!r} is not a mapping in ppo_base_config")
                if not key_exists and not _path_allows_new_keys(path):
                    raise ValueError(f"SkyRL config contains unknown key {path!r}")
                OmegaConf.update(config, path, {}, merge=False, force_add=_path_allows_new_keys(path))
            _merge_config_mapping(config, value, path)
            continue
        if not key_exists and not _path_allows_new_keys(path):
            raise ValueError(f"SkyRL config contains unknown key {path!r}")
        OmegaConf.update(config, path, copy.deepcopy(value), merge=False, force_add=_path_allows_new_keys(path))


def _compose_config_groups(config_groups: Mapping[str, str], *, config_name: str | None) -> DictConfig:
    config_dir = Path(str(files("skyrl_train.config"))).resolve()
    group_overrides = [f"+{group_name}={config_name}" for group_name, config_name in config_groups.items()]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=group_overrides)
    OmegaConf.set_struct(config, True)
    return config


def compose_skyrl_config(
    parsed: ParsedRLConfig,
    exp_args: Mapping[str, Any],
    hpc: HPCGeometry,
) -> CompiledSkyRLConfig:
    """Compose the final SkyRL subtree from its config groups and launch values."""
    config = _compose_config_groups(parsed.config_groups, config_name="ppo_base_config")
    _merge_config_mapping(config, _skyrl_config_sections(parsed, exp_args, hpc))
    validate_nemotron_ultra_grading(config, parsed.distillation_plan)
    validate_objective(config)
    return CompiledSkyRLConfig(
        entrypoint=registered_rl_entrypoint_module(parsed.entrypoint),
        config=config,
    )


def compose_checkpoint_export_config(
    parsed: ParsedCheckpointExportConfig,
    exp_args: Mapping[str, Any],
    hpc: HPCGeometry,
) -> CompiledSkyRLConfig:
    """Compose the policy-only checkpoint-export SkyRL subtree."""
    config = _compose_config_groups(parsed.config_groups, config_name="ppo_base_config")
    _merge_config_mapping(config, {"trainer": _checkpoint_export_trainer(parsed, exp_args, hpc)})
    return CompiledSkyRLConfig(
        entrypoint=CHECKPOINT_EXPORT_ENTRYPOINT,
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
