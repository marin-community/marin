# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration and native-schema adaptation for the isolated Harbor driver."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_MODEL_INFO = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
_OPENCODE_AGENT = "opencode"
_HOSTED_VLLM_PROVIDER = "hosted_vllm"
_HOSTED_VLLM_DISPLAY_NAME = "Hosted vLLM"
_OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible"


@dataclass(frozen=True)
class HarborRetryConfig:
    """Retry failed trials unless their exception type is explicitly excluded."""

    max_retries: int = 0
    exclude_exceptions: tuple[str, ...] = ()
    wait_multiplier: float = 1.0
    min_wait: float = 1.0
    max_wait: float = 60.0


@dataclass(frozen=True)
class HarborEnvironmentConfig:
    """Sandbox type, lifecycle, and resources for each trial."""

    environment_type: str
    force_build: bool = False
    delete: bool = True
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarborAgentConfig:
    """Agent identity, timeouts, token budget, and implementation arguments."""

    name: str
    max_output_tokens: int = 8192
    max_timeout: float | None = None
    setup_timeout: float | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarborVerifierConfig:
    """Verifier execution policy."""

    max_timeout: float | None = None


@dataclass(frozen=True)
class HarborRunConfig:
    """One Harbor evaluation of one served model."""

    dataset: str
    revision: str
    agent: HarborAgentConfig
    environment: HarborEnvironmentConfig
    n_concurrent: int = 4
    task_limit: int | None = None
    attempts: int = 1
    timeout_multiplier: float = 1.0
    retry: HarborRetryConfig = field(default_factory=HarborRetryConfig)
    verifier: HarborVerifierConfig = field(default_factory=HarborVerifierConfig)


@dataclass(frozen=True)
class HarborDriverConfig:
    """JSON-safe input shared by the Marin parent and isolated Harbor process."""

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    run: HarborRunConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarborDriverConfig":
        """Decode a JSON object while rejecting unknown or missing config fields."""
        values = dict(data)
        run_values = dict(values.pop("run"))
        run_values["agent"] = HarborAgentConfig(**run_values["agent"])
        run_values["environment"] = HarborEnvironmentConfig(**run_values["environment"])
        run_values["retry"] = HarborRetryConfig(**run_values["retry"])
        run_values["verifier"] = HarborVerifierConfig(**run_values["verifier"])
        return cls(run=HarborRunConfig(**run_values), **values)


def _opencode_config_for_endpoint(config: object, endpoint_url: str) -> dict[str, Any]:
    opencode_config = config
    if not isinstance(opencode_config, Mapping):
        raise ValueError("Harbor agent opencode_config must be a mapping")
    providers = opencode_config.get("provider", {})
    if not isinstance(providers, Mapping):
        raise ValueError("Harbor OpenCode provider config must be a mapping")
    hosted_vllm = providers.get(_HOSTED_VLLM_PROVIDER, {})
    if not isinstance(hosted_vllm, Mapping):
        raise ValueError("Harbor OpenCode hosted_vllm provider config must be a mapping")
    options = hosted_vllm.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Harbor OpenCode hosted_vllm provider options must be a mapping")

    return {
        **opencode_config,
        "provider": {
            **providers,
            _HOSTED_VLLM_PROVIDER: {
                **hosted_vllm,
                "npm": _OPENAI_COMPATIBLE_PACKAGE,
                "name": _HOSTED_VLLM_DISPLAY_NAME,
                "options": {**options, "baseURL": endpoint_url},
            },
        },
    }


def native_job_config(config: HarborDriverConfig) -> dict[str, Any]:
    """Translate the stable Marin config into the pinned Harbor ``JobConfig`` schema."""
    run = config.run
    agent_kwargs = dict(run.agent.kwargs)
    configured_model_info = agent_kwargs.get("model_info") or {}
    if not isinstance(configured_model_info, Mapping):
        raise ValueError("Harbor agent model_info must be a mapping")
    model_info = {
        **_DEFAULT_MODEL_INFO,
        **configured_model_info,
        "max_output_tokens": run.agent.max_output_tokens,
    }
    agent_kwargs.update({"api_base": config.endpoint_url, "model_info": model_info})
    if run.agent.name == _OPENCODE_AGENT:
        agent_kwargs["opencode_config"] = _opencode_config_for_endpoint(
            agent_kwargs.get("opencode_config", {}),
            config.endpoint_url,
        )

    if config.dataset_path is not None:
        dataset = {"path": config.dataset_path, "n_tasks": run.task_limit}
    else:
        selector = {"ref": run.revision} if "/" in run.dataset else {"version": run.revision}
        dataset = {"name": run.dataset, **selector, "n_tasks": run.task_limit}

    return {
        "job_name": config.job_name,
        "jobs_dir": config.jobs_dir,
        "n_attempts": run.attempts,
        "timeout_multiplier": run.timeout_multiplier,
        "n_concurrent_trials": run.n_concurrent,
        "retry": {
            "max_retries": run.retry.max_retries,
            "exclude_exceptions": list(run.retry.exclude_exceptions),
            "wait_multiplier": run.retry.wait_multiplier,
            "min_wait_sec": run.retry.min_wait,
            "max_wait_sec": run.retry.max_wait,
        },
        "environment": {
            "type": run.environment.environment_type,
            "force_build": run.environment.force_build,
            "delete": run.environment.delete,
            "override_cpus": run.environment.cpus,
            "override_memory_mb": run.environment.memory_mb,
            "override_storage_mb": run.environment.storage_mb,
            "kwargs": dict(run.environment.kwargs),
        },
        "verifier": {"max_timeout_sec": run.verifier.max_timeout},
        "agents": [
            {
                "name": run.agent.name,
                "model_name": f"hosted_vllm/{config.served_model}",
                "max_timeout_sec": run.agent.max_timeout,
                "override_setup_timeout_sec": run.agent.setup_timeout,
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [dataset],
    }
