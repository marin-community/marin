"""Resolve Harbor task settings for the shared rollout worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set

from omegaconf import DictConfig, OmegaConf
from skyrl_train.trajectory_runners.harbor.identity_aware_reward import IDENTITY_AWARE_SHAPER
from skyrl_train.utils.harbor_errors import (
    DEFAULT_ERROR_HANDLING_CONFIG,
    ErrorHandlingConfig,
    retry_excluded_exception_types,
)

from skyrl_train.rollouts.harbor_settings import EnvironmentConfig, EnvironmentType, RetryConfig, VerifierConfig


@dataclass
class FieldMapping:
    """Defines how a YAML field maps to Harbor config."""

    harbor_field: str  # Field name in Harbor's Pydantic model
    field_type: str = "direct"  # "direct" or "kwargs"
    default: Any = None  # Default value if not specified


@dataclass
class SectionSchema:
    """Schema for a Harbor config section (agent, environment, etc.)."""

    fields: Dict[str, FieldMapping] = field(default_factory=dict)

    def get_all_field_names(self) -> Set[str]:
        return set(self.fields.keys())


AGENT_SCHEMA = SectionSchema(
    fields={
        "override_timeout_sec": FieldMapping("override_timeout_sec"),
        "max_timeout_sec": FieldMapping("max_timeout_sec"),
        "max_turns": FieldMapping("max_turns", field_type="kwargs"),
    }
)

EVAL_SCHEMA = SectionSchema(
    fields={"eval_timeout_override_sec": FieldMapping("eval_timeout_override_sec", default=900)}
)

ENVIRONMENT_SCHEMA = SectionSchema(
    fields={
        "override_cpus": FieldMapping("override_cpus"),
        "override_memory_mb": FieldMapping("override_memory_mb"),
        "override_storage_mb": FieldMapping("override_storage_mb"),
        "override_gpus": FieldMapping("override_gpus"),
        "environment_type": FieldMapping("type", default=EnvironmentType.DAYTONA.value),
        "env_network_policy": FieldMapping("network_policy", field_type="kwargs"),
        "ttl_minutes": FieldMapping("ttl_minutes", field_type="kwargs"),
        "daytona_api_key_env": FieldMapping("api_key_env", field_type="kwargs"),
        "eval_daytona_api_key_env": FieldMapping("eval_api_key_env", field_type="kwargs"),
    }
)

# Verifier config fields
VERIFIER_SCHEMA = SectionSchema(
    fields={
        "verifier_override_timeout_sec": FieldMapping("override_timeout_sec"),
        "verifier_max_timeout_sec": FieldMapping("max_timeout_sec"),
        "verifier_disable": FieldMapping("disable"),
    }
)

# Trial-level config fields
TRIAL_SCHEMA = SectionSchema(
    fields={
        "timeout_multiplier": FieldMapping("timeout_multiplier", default=1.0),
        "trial_attempt_timeout_sec": FieldMapping("trial_attempt_timeout_sec"),
    }
)

# Retry settings for the task worker.
RETRY_SCHEMA = SectionSchema(
    fields={
        "max_retries": FieldMapping("max_retries", default=2),
        "min_wait_sec": FieldMapping("min_wait_sec", default=1.0),
        "max_wait_sec": FieldMapping("max_wait_sec", default=60.0),
        "wait_multiplier": FieldMapping("wait_multiplier", default=2.0),
        # Exception filtering - comma-separated strings in YAML, converted to sets
        "include_exceptions": FieldMapping("include_exceptions"),
        "exclude_exceptions": FieldMapping("exclude_exceptions"),
    }
)

# Orchestrator config fields
ORCHESTRATOR_SCHEMA = SectionSchema(
    fields={
        "n_concurrent_trials": FieldMapping("n_concurrent_trials"),
    }
)

# Reward shaping config fields
REWARD_SHAPING_SCHEMA = SectionSchema(
    fields={
        # Parser for test output (pytest, unittest, generic, or None for auto-detect)
        "reward_parser": FieldMapping("reward_parser", default=None),
        # Shaper strategy:
        #   Group-based: identity_aware_pass_ratio (default)
        #   Verifier-based: pass_ratio, effective_pass_ratio, weighted, threshold, binary_partial, original
        #   Trajectory-based: thinking_length, format_quality
        #   Composite: composite (weighted combination of verifier + trajectory shapers)
        "reward_shaper": FieldMapping("reward_shaper", default=IDENTITY_AWARE_SHAPER),
        # Optional weights keyed by the deterministic test-and-trial record ID.
        "identity_aware_test_weights": FieldMapping("identity_aware_test_weights", default=None),
        # Whether to enable reward shaping (if False, uses original binary reward)
        "enable_reward_shaping": FieldMapping("enable_reward_shaping", default=False),
        # Fallback to original reward if parsing fails
        "reward_shaping_fallback": FieldMapping("reward_shaping_fallback", default=True),
        # Threshold shaper params
        "reward_threshold": FieldMapping("reward_threshold", default=1.0),
        "below_threshold_scale": FieldMapping("below_threshold_scale", default=0.5),
        # Binary partial shaper params
        "partial_threshold": FieldMapping("partial_threshold", default=0.9),
        "partial_credit": FieldMapping("partial_credit", default=0.5),
        # Thinking length shaper params
        "thinking_target_tokens": FieldMapping("thinking_target_tokens", default=750),
        "thinking_sigma_tokens": FieldMapping("thinking_sigma_tokens", default=250),
        "thinking_min_turns_ratio": FieldMapping("thinking_min_turns_ratio", default=0.5),
        # Format quality shaper params
        "format_required_fields": FieldMapping("format_required_fields", default=None),
        "format_penalize_truncated": FieldMapping("format_penalize_truncated", default=True),
        # Command quality shaper params
        "command_quality_error_penalty_weight": FieldMapping("command_quality_error_penalty_weight", default=1.0),
        "command_quality_min_turns": FieldMapping("command_quality_min_turns", default=2),
        # Composite shaper params
        "composite_components": FieldMapping("composite_components", default=None),
        "composite_verifier_shaper": FieldMapping("composite_verifier_shaper", default="pass_ratio"),
        # composite_loop (loop-behavior reward shaping) params.
        # `loop_shaping` is the nested config block (all components default-off);
        # see skyrl_train.utils.reward_shaping.DEFAULT_LOOP_SHAPING_CONFIG.
        # `loop_outcome_shaper` selects which verifier-based shaper computes the
        # outcome term (default pass_ratio -> byte-identical to today when the
        # components are off). Both default to None so the composite_loop shaper
        # falls back to its own defaults when unset.
        "loop_shaping": FieldMapping("loop_shaping", default=None),
        "loop_outcome_shaper": FieldMapping("loop_outcome_shaper", default=None),
        # Loop-behavior reward shaping (Stage B / F5 + F4): the master gate for the
        # per-token shaping channel + span tagger. Default False -> the runner
        # emits NEITHER token_level_shaping NOR response_span_tags, so the
        # TrajectoryBatch is byte-identical to today. When True, Stage B emits the
        # channel as ZEROS (no-op) plus the span tags; Stages C/D fill the channel.
        "enable_token_reward_channel": FieldMapping("enable_token_reward_channel", default=False),
        # When True (and the channel is enabled) also emit F4 span tags. Separable
        # so the tagger cost can be disabled independently. No-op when the channel
        # is off.
        "enable_span_tagging": FieldMapping("enable_span_tagging", default=True),
        # Loop-behavior reward shaping (Stage C / F2 + F6): potential-based shaping
        # of the EDIT-token span from the in-trajectory test-delta. Default False ->
        # the channel ships Stage-B ZEROS (no-op, byte-identical). Requires
        # enable_token_reward_channel + enable_span_tagging. PBS is policy-invariant
        # (Ng 1999); the total shaping is bounded to ±pbs_max_total_shaping so the
        # hidden-test outcome reward stays dominant.
        "enable_pbs_shaping": FieldMapping("enable_pbs_shaping", default=False),
        "pbs_gamma": FieldMapping("pbs_gamma", default=1.0),
        "pbs_max_total_shaping": FieldMapping("pbs_max_total_shaping", default=0.3),
        "pbs_potential_shape": FieldMapping("pbs_potential_shape", default="linear"),
        # Truncation penalty: penalize generations that terminate at
        # max_generate_length rather than emitting a stop token. 0.0 == no-op
        # (byte-identical). A cap-truncated trial is otherwise scored
        # identically to an honest wrong answer, so nothing opposes the policy
        # drifting longer until every generation hits the wall. When >0, a
        # truncated trial with zero original reward is scored at
        # -truncation_penalty (below the zero floor) so it is distinguishable
        # from an honest failure in the advantage signal.
        "truncation_penalty": FieldMapping("truncation_penalty", default=0.0),
    }
)

# Error handling config fields (for RLOO-N advantage estimator)
# Controls how different failure types are treated:
# - "mask" exceptions: Excluded from baseline (neutral - infrastructure failures)
# - "zero" exceptions: Included in baseline with reward=0 (agent failures)
#
# Default classification comes from SkyRL's Harbor error taxonomy. Lists
# here are explicit campaign overrides and therefore default empty.
ERROR_HANDLING_SCHEMA = SectionSchema(
    fields={
        # Enable RLOO-N style error handling (exclude infrastructure failures from baseline)
        "enable_error_classification": FieldMapping(
            "enable_error_classification",
            default=DEFAULT_ERROR_HANDLING_CONFIG.enable_error_classification,
        ),
        # Exceptions to pass through (ignore exception, use verifier reward normally).
        # The verifier still runs after these errors in Harbor, so we get a real reward.
        # Use for soft limits like timeout/context-length where partial work is evaluated.
        "passthrough_exceptions": FieldMapping(
            "passthrough_exceptions",
            default=list(DEFAULT_ERROR_HANDLING_CONFIG.passthrough_exceptions),
        ),
        # Exceptions to mask (exclude from baseline, no gradient contribution)
        # These are treated as "neutral" - infrastructure issues, not agent failures
        "mask_exceptions": FieldMapping(
            "mask_exceptions",
            default=list(DEFAULT_ERROR_HANDLING_CONFIG.mask_exceptions),
        ),
        # Exceptions to zero (include in baseline with reward=0)
        # These are treated as agent failures - the model should learn to avoid them
        "zero_exceptions": FieldMapping(
            "zero_exceptions",
            default=list(DEFAULT_ERROR_HANDLING_CONFIG.zero_exceptions),
        ),
        # Default treatment for unclassified exceptions ("mask", "zero", or "passthrough")
        "default_error_treatment": FieldMapping(
            "default_error_treatment",
            default=DEFAULT_ERROR_HANDLING_CONFIG.default_error_treatment.value,
        ),
        "preserve_logprobs_on_timeout": FieldMapping(
            "preserve_logprobs_on_timeout",
            default=DEFAULT_ERROR_HANDLING_CONFIG.preserve_logprobs_on_timeout,
        ),
    }
)

# Complete schema registry
HARBOR_SCHEMA = {
    "agent": AGENT_SCHEMA,
    "environment": ENVIRONMENT_SCHEMA,
    "verifier": VERIFIER_SCHEMA,
    "trial": TRIAL_SCHEMA,
    "retry": RETRY_SCHEMA,
    "orchestrator": ORCHESTRATOR_SCHEMA,
    "reward_shaping": REWARD_SHAPING_SCHEMA,
    "error_handling": ERROR_HANDLING_SCHEMA,
    "eval": EVAL_SCHEMA,
}


def _get_all_exposed_fields() -> Set[str]:
    """Get all field names exposed in our schema."""
    exposed = set()
    for schema in HARBOR_SCHEMA.values():
        exposed.update(schema.get_all_field_names())
    return exposed


# =============================================================================
# Harbor task settings
# =============================================================================


class HarborConfigBuilder:
    """Read task resources, deadlines, retries, and reward settings from SkyRL config."""

    def __init__(self, terminal_bench_cfg: DictConfig):
        unknown = set(terminal_bench_cfg) - {"harbor", "installed_agent"}
        if unknown:
            raise ValueError(f"Unknown terminal task settings: {sorted(unknown)}")
        self._harbor_cfg = OmegaConf.to_container(terminal_bench_cfg.get("harbor", OmegaConf.create({})), resolve=True)
        if not isinstance(self._harbor_cfg, dict):
            raise ValueError("terminal_bench_config.harbor must be a mapping")
        unknown = set(self._harbor_cfg) - _get_all_exposed_fields()
        if unknown:
            raise ValueError(f"Unknown Harbor task settings: {sorted(unknown)}")

    def _get_field_value(self, yaml_key: str, mapping: FieldMapping) -> Any:
        return self._harbor_cfg.get(yaml_key, mapping.default)

    def agent_fields(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build agent direct fields and kwargs from config."""
        direct_fields = {}
        kwargs_fields = {}

        for yaml_key, mapping in AGENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                if mapping.field_type == "kwargs":
                    kwargs_fields[mapping.harbor_field] = value
                else:
                    direct_fields[mapping.harbor_field] = value

        return direct_fields, kwargs_fields

    def environment_config(self) -> EnvironmentConfig:
        """Build EnvironmentConfig from config."""
        env_fields = {}
        env_kwargs = {}

        for yaml_key, mapping in ENVIRONMENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                if mapping.field_type == "kwargs":
                    # Pass through to environment kwargs
                    env_kwargs[mapping.harbor_field] = value
                elif mapping.harbor_field == "type":
                    # Special handling for environment type
                    if isinstance(value, str):
                        value = EnvironmentType(value)
                    env_fields[mapping.harbor_field] = value
                else:
                    env_fields[mapping.harbor_field] = value

        # Add kwargs if any were collected
        if env_kwargs:
            env_fields["kwargs"] = env_kwargs

        return EnvironmentConfig(**env_fields)

    def verifier_config(self) -> VerifierConfig:
        """Build VerifierConfig from config."""
        verifier_fields = {}

        for yaml_key, mapping in VERIFIER_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                verifier_fields[mapping.harbor_field] = value

        return VerifierConfig(**verifier_fields)

    def trial_fields(self) -> Dict[str, Any]:
        """Get trial-level fields from config."""
        trial_fields = {}

        for yaml_key, mapping in TRIAL_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                trial_fields[mapping.harbor_field] = value

        return trial_fields

    def build_retry_config(self) -> RetryConfig:
        """Build the shared worker retry policy.

        Explicit and Harbor-default exclusions are combined with every exception
        type that the shared taxonomy and campaign overrides classify as pass-through.
        Pass-through failures are terminal results that may retain verifier output;
        retrying would discard that result.

        Returns:
            RetryConfig with exponential backoff and resolved terminal exceptions.
        """
        retry_fields = {}

        for yaml_key, mapping in RETRY_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                # Handle exception sets (YAML lists -> Python sets)
                if yaml_key in ("include_exceptions", "exclude_exceptions"):
                    if isinstance(value, (list, tuple)):
                        value = set(value)
                    elif isinstance(value, str):
                        # Support comma-separated string
                        value = {s.strip() for s in value.split(",") if s.strip()}
                retry_fields[mapping.harbor_field] = value

        retry_config = RetryConfig(**retry_fields)
        excluded = retry_excluded_exception_types(
            retry_config.exclude_exceptions,
            self.get_error_handling_config(),
        )
        return retry_config.model_copy(update={"exclude_exceptions": set(excluded)})

    def get_n_concurrent_trials(self, default: int = 16) -> int:
        """
        Get the number of concurrent Harbor tasks.

        Args:
            default: Default concurrency if not specified in config.

        Returns:
            Number of concurrent trials to run.
        """
        mapping = ORCHESTRATOR_SCHEMA.fields.get("n_concurrent_trials")
        if mapping:
            value = self._get_field_value("n_concurrent_trials", mapping)
            if value is not None:
                return int(value)
        return default

    def get_reward_shaping_config(self) -> Dict[str, Any]:
        """
        Get reward shaping configuration for the task worker.

        Returns:
            Dict with keys:
                - enable_reward_shaping: bool
                - reward_parser: str | None (pytest, unittest, generic, or None for auto)
                - reward_shaper: str (identity_aware_pass_ratio, pass_ratio, weighted, etc.)
                - reward_shaping_fallback: bool
                - shaper_kwargs: dict with shaper-specific params
        """
        config = {}

        for yaml_key, mapping in REWARD_SHAPING_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                config[yaml_key] = value

        # Build shaper kwargs from shaper-specific params
        shaper_kwargs = {}

        if "identity_aware_test_weights" in config:
            val = config.pop("identity_aware_test_weights")
            if OmegaConf.is_config(val):
                val = OmegaConf.to_container(val, resolve=True)
            shaper_kwargs["test_weights"] = val

        # Threshold shaper params
        if "reward_threshold" in config:
            shaper_kwargs["threshold"] = config.pop("reward_threshold")
        if "below_threshold_scale" in config:
            shaper_kwargs["below_threshold_scale"] = config.pop("below_threshold_scale")

        # Binary partial shaper params
        if "partial_threshold" in config:
            shaper_kwargs["partial_threshold"] = config.pop("partial_threshold")
        if "partial_credit" in config:
            shaper_kwargs["partial_credit"] = config.pop("partial_credit")

        # Thinking length shaper params
        if "thinking_target_tokens" in config:
            shaper_kwargs["target_tokens"] = config.pop("thinking_target_tokens")
        if "thinking_sigma_tokens" in config:
            shaper_kwargs["sigma_tokens"] = config.pop("thinking_sigma_tokens")
        if "thinking_min_turns_ratio" in config:
            shaper_kwargs["min_thinking_turns_ratio"] = config.pop("thinking_min_turns_ratio")

        # Format quality shaper params
        if "format_required_fields" in config:
            val = config.pop("format_required_fields")
            if val is not None:
                shaper_kwargs["required_fields"] = val
        if "format_penalize_truncated" in config:
            shaper_kwargs["penalize_truncated_json"] = config.pop("format_penalize_truncated")

        # Command quality shaper params
        if "command_quality_error_penalty_weight" in config:
            shaper_kwargs["error_penalty_weight"] = config.pop("command_quality_error_penalty_weight")
        if "command_quality_min_turns" in config:
            shaper_kwargs["min_turns"] = config.pop("command_quality_min_turns")

        # Composite shaper params
        if "composite_components" in config:
            val = config.pop("composite_components")
            if val is not None:
                shaper_kwargs["components"] = val
        if "composite_verifier_shaper" in config:
            shaper_kwargs["verifier_shaper"] = config.pop("composite_verifier_shaper")

        # composite_loop shaper params. `loop_shaping` is the nested block
        # (deep-merged onto defaults inside CompositeLoopShaper); `outcome_shaper`
        # selects the outcome term's verifier shaper.
        if "loop_shaping" in config:
            val = config.pop("loop_shaping")
            if val is not None:
                # OmegaConf DictConfig -> plain dict so the shaper can mutate it.
                if OmegaConf.is_config(val):
                    val = OmegaConf.to_container(val, resolve=True)
                shaper_kwargs["loop_shaping"] = val
        if "loop_outcome_shaper" in config:
            val = config.pop("loop_outcome_shaper")
            if val is not None:
                shaper_kwargs["outcome_shaper"] = val

        # Pass trajectory shaper kwargs through for composite mode
        trajectory_shaper_kwargs = {}
        if "target_tokens" in shaper_kwargs or "sigma_tokens" in shaper_kwargs:
            trajectory_shaper_kwargs["thinking_length"] = {
                k: shaper_kwargs[k]
                for k in ["target_tokens", "sigma_tokens", "min_thinking_turns_ratio"]
                if k in shaper_kwargs
            }
        if "required_fields" in shaper_kwargs or "penalize_truncated_json" in shaper_kwargs:
            trajectory_shaper_kwargs["format_quality"] = {
                k: shaper_kwargs[k] for k in ["required_fields", "penalize_truncated_json"] if k in shaper_kwargs
            }
        if "error_penalty_weight" in shaper_kwargs or "min_turns" in shaper_kwargs:
            trajectory_shaper_kwargs["command_quality"] = {
                k: shaper_kwargs[k] for k in ["error_penalty_weight", "min_turns"] if k in shaper_kwargs
            }
        if trajectory_shaper_kwargs:
            shaper_kwargs["trajectory_shaper_kwargs"] = trajectory_shaper_kwargs

        config["shaper_kwargs"] = shaper_kwargs

        return config

    def get_error_handling_config(self) -> ErrorHandlingConfig:
        """Build the validated RLOO-N treatment configuration."""
        config = {}

        for yaml_key, mapping in ERROR_HANDLING_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping)
            if value is not None:
                config[yaml_key] = value

        return ErrorHandlingConfig.from_mapping(config)

    def get_eval_timeout_override_sec(self) -> int | None:
        """Return the eval override, preserving explicit null as no override."""
        value = self._get_field_value("eval_timeout_override_sec", EVAL_SCHEMA.fields["eval_timeout_override_sec"])
        return None if value is None else int(value)
