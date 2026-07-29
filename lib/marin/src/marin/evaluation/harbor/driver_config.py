# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration and native-schema adaptation for the isolated Harbor driver."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marin.external_dependencies import HARBOR

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
_CONFIG_VALIDATOR = Path(__file__).with_name("config_validator.py")


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
class MarinHarborConfig:
    """Stable Marin policy plus runtime values lowered into a native Harbor job."""

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    run: HarborRunConfig


@dataclass(frozen=True)
class NativeHarborConfig:
    """A validated, normalized native Harbor ``JobConfig`` and its durable identity."""

    normalized: Mapping[str, Any]
    digest: str
    dataset: str
    revision: str
    agent: str
    environment: str


def _single_config_entry(config: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    values = config.get(field_name)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
        raise ValueError(f"Harbor config must declare exactly one {field_name.removesuffix('s')}")
    return values[0]


def _native_config_from_normalized(config: Mapping[str, Any]) -> NativeHarborConfig:
    if config.get("tasks"):
        raise ValueError("Harbor config tasks are incompatible with the shared launcher; declare exactly one dataset")
    agent = _single_config_entry(config, "agents")
    dataset = _single_config_entry(config, "datasets")
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("Harbor config must declare one environment")

    dataset_name = dataset.get("name") or dataset.get("path")
    agent_name = agent.get("name") or agent.get("import_path")
    environment_name = environment.get("type") or environment.get("import_path")
    if not isinstance(dataset_name, str):
        raise ValueError("Harbor config dataset must have a name or path")
    if not isinstance(agent_name, str):
        raise ValueError("Harbor config agent must have a name or import_path")
    if not isinstance(environment_name, str):
        raise ValueError("Harbor config environment must have a type or import_path")

    canonical = json.dumps(config, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return NativeHarborConfig(
        normalized=config,
        digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        dataset=dataset_name,
        revision=str(dataset.get("ref") or dataset.get("version") or "unversioned"),
        agent=agent_name,
        environment=environment_name,
    )


def load_native_harbor_config(path: Path) -> NativeHarborConfig:
    """Validate and normalize a native Harbor YAML or JSON file with the pinned Harbor schema."""
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--prerelease=allow",
        "--with",
        HARBOR.requirement(),
        "python",
        str(_CONFIG_VALIDATOR),
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"validator exited with status {exc.returncode}"
        raise ValueError(f"invalid Harbor config {path}: {detail}") from exc
    try:
        normalized = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Harbor validator returned invalid JSON for {path}") from exc
    if not isinstance(normalized, Mapping):
        raise ValueError(f"Harbor config {path} must contain a mapping")
    return _native_config_from_normalized(normalized)


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


def native_job_config(config: MarinHarborConfig) -> dict[str, Any]:
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
                "model_name": f"{_HOSTED_VLLM_PROVIDER}/{config.served_model}",
                "max_timeout_sec": run.agent.max_timeout,
                "override_setup_timeout_sec": run.agent.setup_timeout,
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [dataset],
    }


def adapt_native_job_config(
    config: NativeHarborConfig,
    *,
    job_name: str,
    jobs_dir: str,
    endpoint_url: str,
    served_model: str,
    task_limit: int | None,
    model_agent_kwargs: Mapping[str, str],
) -> dict[str, Any]:
    """Overlay Marin-owned runtime values on a normalized Harbor ``JobConfig``."""
    resolved = json.loads(json.dumps(config.normalized))
    resolved["job_name"] = job_name
    resolved["jobs_dir"] = jobs_dir

    dataset = resolved["datasets"][0]
    if task_limit is not None:
        dataset["n_tasks"] = task_limit

    agent = resolved["agents"][0]
    agent["model_name"] = f"{_HOSTED_VLLM_PROVIDER}/{served_model}"
    agent_kwargs = {**model_agent_kwargs, **agent.get("kwargs", {})}
    agent_kwargs["api_base"] = endpoint_url
    if agent.get("name") == _OPENCODE_AGENT:
        agent_kwargs["opencode_config"] = _opencode_config_for_endpoint(
            agent_kwargs.get("opencode_config", {}),
            endpoint_url,
        )
    agent["kwargs"] = agent_kwargs
    return resolved
