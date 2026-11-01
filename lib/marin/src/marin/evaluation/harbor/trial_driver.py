# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Harbor policies inside its isolated runtime environment."""

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import fsspec.config
import yaml
from harbor.agents.factory import AgentFactory  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor.agents.installed.acp_registry import (  # pyrefly: ignore[missing-import]
    is_acp_registry_shorthand,
    parse_registry_spec,
)
from harbor.agents.installed.base import BaseInstalledAgent  # pyrefly: ignore[missing-import]
from harbor.environments.factory import _load_environment_class  # pyrefly: ignore[missing-import]
from harbor.environments.model_relay import (  # pyrefly: ignore[missing-import]
    CANDIDATE_MODEL_RELAY_PORT,
    MODEL_RELAY_PORT,
    host_model_authentication,
)
from harbor.environments.prepared_compose import (  # pyrefly: ignore[missing-import]
    COMPOSE_FILENAMES,
    validate_prepared_task,
)
from harbor.environments.resource_policies import validate_local_resources  # pyrefly: ignore[missing-import]
from harbor.environments.runtime import DAYTONA_API_URL, host_backend_credentials  # pyrefly: ignore[missing-import]
from harbor.job import Job  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor.models.task.config import TaskOS  # pyrefly: ignore[missing-import]
from harbor.models.task.task import Task  # pyrefly: ignore[missing-import]
from harbor.utils.env import (  # pyrefly: ignore[missing-import]
    resolve_env_vars,
    task_environment,
    validate_task_environment,
)
from harbor_config import JobConfig  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor_config.env import get_required_host_vars  # pyrefly: ignore[missing-import]
from harbor_config.env import resolve_env_vars as resolve_host_env_vars  # pyrefly: ignore[missing-import]
from harbor_config.errors import ErrorCategory, errors_by_category, known_error_types  # pyrefly: ignore[missing-import]
from harbor_config.models.agent.name import AgentName  # pyrefly: ignore[missing-import]
from harbor_config.models.environment_type import EnvironmentType  # pyrefly: ignore[missing-import]
from harbor_config.models.job.config import ArchiveConfig, DatasetConfig  # pyrefly: ignore[missing-import]
from harbor_config.models.trial.config import AgentConfig  # pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, ValidationError
from rigging.filesystem.s3_compat import s3_credential_options
from rigging.filesystem.storage_path import StoragePath

from marin.evaluation.harbor.agent_context import (
    MAX_INPUT_TOKENS_KEY,
    MAX_OUTPUT_TOKENS_KEY,
    MODEL_INFO_KEY,
    reconciled_model_info,
)

_HOSTED_VLLM_PROVIDER = "hosted_vllm"
_HOSTED_VLLM_DISPLAY_NAME = "Hosted vLLM"
_OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible"
_OPENCODE_AGENT = "opencode"
_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
_PI_ACP_REGISTRY_ID = "pi-acp"
_TAU3_USER_AGENT = "harbor.agents.tau3_acp_relay.harbor_agent:Tau3HarborACPRelay"
_PI_VERSION = "0.87.0"
_TERMINUS_2_AGENT = "terminus-2"
_LLM_CALL_KWARGS_KEY = "llm_call_kwargs"
_MAX_TOKENS_KEY = "max_tokens"
_STABLE_JOB_NAME = "__marin_job__"
_STABLE_JOBS_DIR = "/__marin_jobs__"
_STABLE_MODEL = "__marin_model__"
_STABLE_ENDPOINT = "http://marin.invalid/v1"
_HF_DATASET_PREFIX = "hf://"
_MODEL_RELAY_ENVIRONMENT_KWARGS = frozenset(
    {
        "model_endpoint",
        "model_name",
        "verifier_model_endpoint",
        "candidate_model_endpoint",
        "candidate_model_name",
    }
)
_LOCAL_ENVIRONMENT_KWARGS = _MODEL_RELAY_ENVIRONMENT_KWARGS | {
    "bwrap",
    "umoci",
    "image_cache",
    "registry_authfile",
    "signature_policy",
    "cgroup_parent",
    "ttl",
}
_MODEL_RELAY_ENVIRONMENTS = frozenset({EnvironmentType.LOCAL, EnvironmentType.DAYTONA})
_TASK_MODEL_ENVIRONMENT = {
    "OPENAI_BASE_URL": f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1",
    "OPENAI_API_KEY": "EMPTY",
}
_VERIFIER_MODEL_ROLES = ("MODEL_NAME", "JUDGE_MODEL", "REFUSAL_JUDGE_MODEL", "EXPANSION_MODEL")


class RuntimeOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    model_identity: str
    task_limit: int | None
    model_agent_kwargs: dict[str, Any]
    verifier_env: dict[str, str]
    archive_root: str
    archive_dataset: str


class _DatasetKind(StrEnum):
    HARBOR_REGISTRY = "harbor_registry"
    HUGGING_FACE = "hugging_face"
    LOCAL = "local"


@dataclass(frozen=True)
class _DatasetMetadata:
    kind: _DatasetKind
    selector: str
    revision: str | None


def _document(path: Path) -> Mapping[str, object]:
    if path.suffix in {".yaml", ".yml"}:
        document = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        document = json.loads(path.read_text())
    else:
        raise ValueError(f"unsupported Harbor config file format: {path.suffix}")
    if not isinstance(document, Mapping):
        raise ValueError("Harbor config must contain a mapping")
    return document


def _single_entry(config: JobConfig, field_name: str) -> object:
    values = getattr(config, field_name)
    if len(values) != 1:
        raise ValueError(f"Harbor config must declare exactly one {field_name.removesuffix('s')}")
    return values[0]


def _resolve_import_path(import_path: str, label: str) -> object:
    if ":" not in import_path:
        raise ValueError(f"Harbor {label} import path must use module.path:ClassName")
    module_path, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Harbor {label} import path could not be resolved") from exc


_AGENT_CALLBACK_KEYWORDS = (
    ("setup", ("environment",)),
    ("run", ("instruction", "environment", "context")),
)


def _validate_agent_callbacks(agent_class: object, import_path: str) -> None:
    """Validate the keyword callbacks used by upstream Harbor's trial runner."""
    for method_name, keywords in _AGENT_CALLBACK_KEYWORDS:
        method = getattr(agent_class, method_name, None)
        if not callable(method):
            raise ValueError(f"Harbor agent {import_path} does not define a callable {method_name}()")
        try:
            inspect.signature(method).bind(agent_class, **dict.fromkeys(keywords))
        except TypeError as exc:
            raise ValueError(
                f"Harbor agent {import_path}.{method_name}() does not accept the keyword arguments "
                f"upstream Harbor calls it with ({', '.join(keywords)}): {exc}"
            ) from exc


def _validate_agent(agent: AgentConfig) -> str:
    if agent.import_path is not None:
        if agent.mode == "local":
            raise ValueError("Harbor local-mode agents must use a supported agent name")
        agent_class = _resolve_import_path(agent.import_path, "agent")
        _validate_agent_callbacks(agent_class, agent.import_path)
        return agent.import_path
    agent_name = agent.name
    if agent_name is None:
        raise ValueError("Harbor config agent name is not supported by the pinned runtime")
    if is_acp_registry_shorthand(agent_name):
        if agent.mode == "local" or AgentName.ACP not in AgentFactory._AGENT_MAP:
            raise ValueError("Harbor ACP registry agent is not available in the pinned runtime")
        agent_id, _ = parse_registry_spec(agent_name)
        if agent_id != _PI_ACP_REGISTRY_ID:
            raise ValueError("Marin's hosted model settings support only the pi-acp registry agent")
        return agent_name
    if agent_name not in AgentName.values():
        raise ValueError("Harbor config agent name is not supported by the pinned runtime")
    agent_name = AgentName(agent_name)
    agent_registry = AgentFactory._LOCAL_AGENT_MAP if agent.mode == "local" else AgentFactory._AGENT_MAP
    if agent_name not in agent_registry:
        raise ValueError("Harbor config agent is not available in the pinned runtime")
    return agent_name.value


def _validate_environment(config: JobConfig) -> str:
    environment = config.environment
    _load_environment_class(environment.type)
    if environment.type == EnvironmentType.LOCAL:
        _validate_prepared_policy(config)
    elif environment.type in {EnvironmentType.QEMU, EnvironmentType.SHELLSIM}:
        resolve_env_vars(config.verifier.env)
    return environment.type.value


def _validate_prepared_policy(config: JobConfig) -> None:
    """Keep model transport and credentials in the trusted harness for offline tasks."""
    environment = config.environment
    label = f"Shellbox {environment.type.value}"
    if environment.force_build or environment.extra_docker_compose or environment.mounts or environment.env:
        raise ValueError(f"{label} does not accept builds, Compose overlays, mounts, or forwarded environment variables")
    unknown_kwargs = environment.kwargs.keys() - _LOCAL_ENVIRONMENT_KWARGS
    if unknown_kwargs:
        raise ValueError(f"Unsupported {label} environment kwargs: {sorted(unknown_kwargs)}")
    if (user_agent := config.user_agent) is not None:
        if (
            user_agent.name is not None
            or user_agent.import_path != _TAU3_USER_AGENT
            or user_agent.mode != "container"
            or user_agent.env
            or user_agent.skills
            or user_agent.user_persona_path is not None
            or user_agent.user_prompt_template_path is not None
            or user_agent.bridge.prompt_path is not None
            or user_agent.bridge.kwargs
            or len(user_agent.mcp_servers) != 1
        ):
            raise ValueError(f"{label} supports only the sandboxed Tau3 ACP simulated user")
        server = user_agent.mcp_servers[0]
        if (
            server.name != "tau3-runtime"
            or server.transport != "streamable-http"
            or server.url != "http://tau3-runtime:8000/mcp"
            or server.command is not None
            or server.args
        ):
            raise ValueError(f"{label} requires the prepared tau3-runtime MCP service")
        if {"extra_env", "mcp_servers", "skills_dir", "turn_callback"} & user_agent.kwargs.keys():
            raise ValueError(f"{label} does not accept alternate simulated-user inputs")
    for agent in config.agents:
        pi_acp = (
            agent.name is not None
            and is_acp_registry_shorthand(agent.name)
            and parse_registry_spec(agent.name)[0] == _PI_ACP_REGISTRY_ID
        ) or (
            agent.name == "acp"
            and isinstance(agent.kwargs.get("registry_entry"), dict)
            and agent.kwargs["registry_entry"].get("id") == _PI_ACP_REGISTRY_ID
            and not {"registry_entry_path", "registry_spec"} & agent.kwargs.keys()
        )
        installed = agent.name == "pi" or pi_acp
        if (
            (agent.name not in {"terminus-2", "oracle"} and not installed)
            or agent.import_path is not None
            or agent.mode != "container"
        ):
            raise ValueError(f"{label} requires terminus-2, oracle, Pi, or pi-acp")
        relay_env = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{CANDIDATE_MODEL_RELAY_PORT}/v1",
            "OPENAI_API_BASE": f"http://127.0.0.1:{CANDIDATE_MODEL_RELAY_PORT}/v1",
            "OPENAI_API_KEY": "EMPTY",
        }
        if (agent.env and (not installed or agent.env != relay_env)) or agent.mcp_servers or agent.skills:
            raise ValueError(f"{label} does not accept agent environment forwarding, MCP servers, or skills")
        if {"extra_env", "mcp_servers", "skills_dir", "turn_callback"} & agent.kwargs.keys():
            raise ValueError(f"{label} does not accept alternate agent environment, MCP, skill, or callback inputs")
    if config.verifier.import_path is not None:
        raise ValueError(f"{label} requires a sandboxed verifier without a custom import")
    validate_task_environment(config.verifier.env)


def _validate_job_tasks(job: Job) -> None:
    # Harbor has no public iterator over its resolved task selection. Inspect the
    # already-downloaded tasks so preflight and execution enforce the same policy.
    for task_config in job._task_configs:
        task_path = (
            job._task_download_results[task_config.get_task_id()].path
            if task_config.is_git_task() or task_config.is_package_task()
            else task_config.path
        )
        task = Task(task_path, disable_verification=job.config.verifier.disable)
        task_environments = (
            task.config.environment.env,
            {**task.config.verifier.env, **job.config.verifier.env},
            task.config.solution.env,
        )
        if not job.config.environment.kwargs.get("model_endpoint") and any(
            {"OPENAI_API_KEY", "OPENAI_BASE_URL", "TOGETHER_API_KEY", "CUSTOM_API_KEY"} & values.keys()
            for values in task_environments
        ):
            raise ValueError("Task model inputs require an explicit host judge capability before allocation")
        if task.config.environment.os != TaskOS.LINUX:
            raise ValueError("Shellbox Harbor adapters currently require Linux tasks")
        if job.config.environment.type == EnvironmentType.LOCAL:
            environment = job.config.environment
            task_environment_config = task.config.environment.model_copy(
                update={
                    name: value
                    for name in ("cpus", "memory_mb", "storage_mb", "gpus", "tpu")
                    if (value := getattr(environment, f"override_{name}")) is not None
                }
            )
            validate_prepared_task(task_environment_config, task.paths.environment_dir)
            validate_local_resources(
                task_environment_config,
                cpu_enforcement_policy=environment.cpu_enforcement_policy,
                memory_enforcement_policy=environment.memory_enforcement_policy,
                cgroup_parent=(
                    Path(environment.kwargs["cgroup_parent"])
                    if environment.kwargs.get("cgroup_parent") is not None
                    else None
                ),
            )
            if task.config.steps:
                raise ValueError("Prepared Shellbox environments do not yet support multi-step tasks")
        elif any((task.paths.environment_dir / name).exists() for name in COMPOSE_FILENAMES):
            raise ValueError("Compose service groups require the Shellbox local backend")
        resolve_env_vars(task.config.environment.env)
        resolve_env_vars({**task.config.verifier.env, **job.config.verifier.env})
        resolve_env_vars(task.config.solution.env)
        if task.config.verifier.environment is not None or task.config.verifier.environment_mode == "separate":
            raise ValueError("Shellbox model capabilities currently require a shared verifier environment")


def _raw_dataset_path(document: Mapping[str, object]) -> object:
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], Mapping):
        return None
    return datasets[0].get("path")


def _dataset_metadata(
    dataset: DatasetConfig,
    raw_path: object,
) -> _DatasetMetadata:
    if isinstance(raw_path, str) and raw_path.startswith(_HF_DATASET_PREFIX):
        raise ValueError("Harbor hf:// sources must use datasets[].name, not datasets[].path")
    if dataset.path is not None:
        if dataset.path.is_absolute():
            raise ValueError("Harbor local dataset paths must be relative to the config file")
        return _DatasetMetadata(_DatasetKind.LOCAL, str(dataset.path), None)

    assert dataset.name is not None
    revision = dataset.ref or dataset.version
    if dataset.name.startswith(_HF_DATASET_PREFIX):
        selector = dataset.name.removeprefix(_HF_DATASET_PREFIX)
        repository_parts = selector.split("/")
        if len(repository_parts) != 2 or any(not part for part in repository_parts):
            raise ValueError("Harbor hf:// dataset names must identify an org/repository")
        if dataset.version is not None:
            raise ValueError("Harbor hf:// datasets must use ref for their revision")
        return _DatasetMetadata(_DatasetKind.HUGGING_FACE, selector, revision)
    return _DatasetMetadata(_DatasetKind.HARBOR_REGISTRY, dataset.name, revision)


def _opencode_config(config: object, endpoint_url: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("Harbor agent opencode_config must be a mapping")
    providers = config.get("provider", {})
    if not isinstance(providers, Mapping):
        raise ValueError("Harbor OpenCode provider config must be a mapping")
    hosted_vllm = providers.get(_HOSTED_VLLM_PROVIDER, {})
    if not isinstance(hosted_vllm, Mapping):
        raise ValueError("Harbor OpenCode hosted_vllm provider config must be a mapping")
    options = hosted_vllm.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Harbor OpenCode hosted_vllm provider options must be a mapping")
    return {
        **config,
        "provider": {
            **providers,
            _HOSTED_VLLM_PROVIDER: {
                **hosted_vllm,
                "npm": _OPENAI_COMPATIBLE_PACKAGE,
                "name": _HOSTED_VLLM_DISPLAY_NAME,
                "options": {**options, "baseURL": endpoint_url, "apiKey": "EMPTY"},
            },
        },
    }


def _terminus_llm_call_kwargs(
    config: Mapping[str, Any] | None,
    max_output_tokens: int,
) -> dict[str, Any]:
    if config is None:
        call_kwargs: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        call_kwargs = config
    else:
        raise ValueError("Harbor agent llm_call_kwargs must be a mapping")

    if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise ValueError(f"Harbor agent model_info.max_output_tokens must be positive, got {max_output_tokens!r}")
    max_tokens = call_kwargs.get(_MAX_TOKENS_KEY, max_output_tokens)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"Harbor agent llm_call_kwargs.max_tokens must be positive, got {max_tokens!r}")
    if max_tokens > max_output_tokens:
        raise ValueError(
            f"Harbor agent llm_call_kwargs.max_tokens is {max_tokens} but model_info.max_output_tokens is only "
            f"{max_output_tokens}; lower the request limit or raise generation.max_gen_toks"
        )
    return {**call_kwargs, _MAX_TOKENS_KEY: max_tokens}


def _pinned_pi_agent[T: AgentConfig](agent: T) -> T:
    if agent.name != AgentName.PI:
        return agent
    return agent.model_copy(update={"kwargs": {**agent.kwargs, "version": _PI_VERSION}})


def _agent_config(
    agent: AgentConfig,
    *,
    endpoint_url: str,
    served_model: str,
    model_identity: str,
    kwargs: Mapping[str, object],
) -> AgentConfig:
    runtime_kwargs = {**kwargs, "api_base": endpoint_url}
    if agent.name == _OPENCODE_AGENT:
        runtime_kwargs["opencode_config"] = _opencode_config(kwargs.get("opencode_config", {}), endpoint_url)
    runtime_env = agent.env
    if agent.name is not None and is_acp_registry_shorthand(agent.name):
        runtime_env = {**agent.env, _OPENAI_BASE_URL_ENV: endpoint_url}
    configured = AgentConfig.model_validate(
        {
            **agent.model_dump(mode="python"),
            "model_name": f"{_HOSTED_VLLM_PROVIDER}/{model_identity}",
            "model_alias": f"{_HOSTED_VLLM_PROVIDER}/{served_model}",
            "kwargs": runtime_kwargs,
            "env": runtime_env,
        },
        extra="forbid",
    )

    return _pinned_pi_agent(configured)


def _installed_agent(agent: AgentConfig) -> bool:
    if agent.mode != "container":
        return False
    if agent.name is not None and is_acp_registry_shorthand(agent.name):
        agent_class = AgentFactory.get_agent_class(AgentName.ACP)
    elif agent.name is not None and agent.name in AgentName.values():
        agent_class = AgentFactory.get_agent_class(AgentName(agent.name))
    else:
        assert agent.import_path is not None
        agent_class = _resolve_import_path(agent.import_path, "agent")
    return isinstance(agent_class, type) and issubclass(agent_class, BaseInstalledAgent)


def _effective_agent(agent: AgentConfig, overlay: RuntimeOverlay) -> AgentConfig:
    kwargs = {**overlay.model_agent_kwargs, **agent.kwargs}
    model_info = reconciled_model_info(
        overlay.model_agent_kwargs.get(MODEL_INFO_KEY),
        agent.kwargs.get(MODEL_INFO_KEY),
    )
    kwargs[MODEL_INFO_KEY] = model_info
    if agent.name == _TERMINUS_2_AGENT:
        kwargs[_LLM_CALL_KWARGS_KEY] = _terminus_llm_call_kwargs(
            kwargs.get(_LLM_CALL_KWARGS_KEY),
            model_info[MAX_OUTPUT_TOKENS_KEY],
        )
    installed = _installed_agent(agent)
    if installed:
        kwargs["api_key"] = "EMPTY"
    configured = _agent_config(
        agent,
        endpoint_url=(f"http://127.0.0.1:{CANDIDATE_MODEL_RELAY_PORT}/v1" if installed else overlay.endpoint_url),
        served_model=overlay.served_model,
        model_identity=overlay.model_identity,
        kwargs=kwargs,
    )
    if installed:
        configured = configured.model_copy(
            update={
                "env": {
                    **_dummy_model_keys(configured.env),
                    "OPENAI_BASE_URL": configured.kwargs["api_base"],
                    "OPENAI_API_BASE": configured.kwargs["api_base"],
                    "OPENAI_API_KEY": "EMPTY",
                }
            }
        )
    return configured


def _stable_agent(agent: AgentConfig) -> AgentConfig:
    reconciled_model_info(None, agent.kwargs.get(MODEL_INFO_KEY))
    return _agent_config(
        agent,
        endpoint_url=_STABLE_ENDPOINT,
        served_model=_STABLE_MODEL,
        model_identity=_STABLE_MODEL,
        kwargs=agent.kwargs,
    )


def _dataset_with_runtime(
    dataset: DatasetConfig,
    *,
    dataset_path: str | None,
    task_limit: int | None,
) -> DatasetConfig:
    document = dataset.model_dump(mode="python")
    if dataset_path is not None:
        document.pop("name", None)
        document.pop("ref", None)
        document.pop("version", None)
        document["path"] = dataset_path
    if task_limit is not None:
        document["n_tasks"] = task_limit
    return DatasetConfig.model_validate(document, extra="forbid")


def _effective_config(config: JobConfig, overlay: RuntimeOverlay) -> JobConfig:
    validate_task_environment(
        {
            name: "EMPTY"
            for declaration in _environment_declarations(config)
            for name, _ in get_required_host_vars(declaration)
        }
    )
    agent = _effective_agent(config.agents[0], overlay)
    dataset = _dataset_with_runtime(
        config.datasets[0],
        dataset_path=overlay.dataset_path,
        task_limit=overlay.task_limit,
    )
    environment = config.environment
    if _installed_agent(config.agents[0]):
        if environment.type not in _MODEL_RELAY_ENVIRONMENTS:
            raise ValueError("Installed candidate model relays require the local or Daytona Shellbox backend")
        environment = environment.model_copy(
            update={
                "kwargs": {
                    **environment.kwargs,
                    "candidate_model_endpoint": overlay.endpoint_url,
                    "candidate_model_name": overlay.served_model,
                }
            }
        )
    judge = _judge_settings(config, overlay)
    verifier_env = {**config.verifier.env, **overlay.verifier_env}
    if judge is not None:
        endpoint, model_name = judge
        if environment.type not in _MODEL_RELAY_ENVIRONMENTS:
            raise ValueError("Host model relays require the local or Daytona Shellbox backend")
        environment = environment.model_copy(
            update={
                "kwargs": {
                    **environment.kwargs,
                    "model_endpoint": endpoint,
                    "model_name": model_name,
                    "verifier_model_endpoint": endpoint,
                }
            }
        )
        verifier_env = _guest_model_environment(verifier_env, model_name)
        if agent.name == "sotopia":
            agent = agent.model_copy(
                update={
                    "kwargs": {
                        **agent.kwargs,
                        "partner_model": "openai/" + model_name,
                        "evaluator_model": "openai/" + model_name,
                        "partner_api_base": _TASK_MODEL_ENVIRONMENT["OPENAI_BASE_URL"],
                        "evaluator_api_base": _TASK_MODEL_ENVIRONMENT["OPENAI_BASE_URL"],
                    }
                }
            )
    agent = agent.model_copy(update={"env": _dummy_model_keys(agent.env)})
    effective = config.model_copy(
        update={
            "job_name": overlay.job_name,
            "jobs_dir": JobConfig(jobs_dir=str(StoragePath(overlay.jobs_dir))).jobs_dir,
            "agents": [agent],
            "user_agent": _pinned_pi_agent(config.user_agent) if config.user_agent is not None else None,
            "datasets": [dataset],
            "environment": environment,
            "verifier": config.verifier.model_copy(update={"env": verifier_env}),
            "archive": ArchiveConfig(
                root=overlay.archive_root,
                dataset=overlay.archive_dataset,
            ),
        }
    )
    with task_environment(_model_environment(effective)):
        _validate_environment(effective)
    return effective


def _environment_declarations(config: JobConfig) -> tuple[dict[str, str], ...]:
    agents = [*config.agents, *([config.user_agent] if config.user_agent is not None else [])]
    return (*(agent.env for agent in agents), config.verifier.env)


def _dummy_model_keys(values: dict[str, str]) -> dict[str, str]:
    return {
        name: "EMPTY" if name.endswith("API_KEY") or name in {"ANTHROPIC_AUTH_TOKEN", "API_KEY"} else value
        for name, value in values.items()
    }


def _guest_model_environment(values: dict[str, str], model: str) -> dict[str, str]:
    result = {**_dummy_model_keys(values), **_TASK_MODEL_ENVIRONMENT}
    if any(name != "MODEL_NAME" and name in values for name in _VERIFIER_MODEL_ROLES):
        result["OPENAI_API_BASE"] = _TASK_MODEL_ENVIRONMENT["OPENAI_BASE_URL"]
    for name in _VERIFIER_MODEL_ROLES:
        if name in result:
            result[name] = "openai/" + model if name != "MODEL_NAME" else model
    return result


def _judge_settings(config: JobConfig, overlay: RuntimeOverlay) -> tuple[str, str] | None:
    values = {**config.verifier.env, **overlay.verifier_env}
    endpoint = values.get("OPENAI_BASE_URL")
    if endpoint:
        configured_model = _single_judge_model(config.verifier.env)
        model = _single_judge_model(overlay.verifier_env) or configured_model
        if not model:
            raise ValueError("A judge endpoint requires an explicit verifier model role")
        return endpoint, model
    agent = config.agents[0]
    if agent.name == "sotopia":
        partner_endpoint = agent.kwargs.get("partner_api_base")
        evaluator_endpoint = agent.kwargs.get("evaluator_api_base")
        partner_model = agent.kwargs.get("partner_model")
        evaluator_model = agent.kwargs.get("evaluator_model")
        if (
            not partner_endpoint
            or not partner_model
            or (partner_endpoint, partner_model) != (evaluator_endpoint, evaluator_model)
        ):
            raise ValueError("Sotopia requires one explicit shared host model capability for partner and evaluator")
        return str(partner_endpoint), str(partner_model).removeprefix("openai/")
    return None


def _single_judge_model(values: dict[str, str]) -> str | None:
    models = {
        (
            model
            if name == "MODEL_NAME"
            else (
                model.removeprefix("together_ai/") if model.startswith("together_ai/") else model.removeprefix("openai/")
            )
        )
        for name in _VERIFIER_MODEL_ROLES
        if (model := values.get(name))
    }
    if len(models) > 1:
        raise ValueError("Distinct verifier model roles require separate host capabilities")
    return next(iter(models), None)


def _judge_authentication(config: JobConfig, overlay: RuntimeOverlay) -> dict[str, str]:
    judge = _judge_settings(config, overlay)
    if judge is None:
        return {}
    declarations = {**config.agents[0].env, **config.verifier.env, **overlay.verifier_env}
    auth = {
        name: value
        for name, value in declarations.items()
        if name in {"OPENAI_API_KEY", "TOGETHER_API_KEY", "CUSTOM_API_KEY"}
    }
    sources = {name for name, _ in get_required_host_vars(auth)}
    if sources - {"OPENAI_API_KEY", "TOGETHER_API_KEY", "CUSTOM_API_KEY"}:
        raise ValueError("Judge credentials must name an explicit model-provider source")
    resolved = resolve_host_env_vars(auth, source={name: os.environ[name] for name in sources if name in os.environ})
    key = resolved.get("OPENAI_API_KEY") or resolved.get("TOGETHER_API_KEY") or resolved.get("CUSTOM_API_KEY") or "EMPTY"
    return {judge[0]: key}


def _model_authentication(config: JobConfig, overlay: RuntimeOverlay, *, candidate_api_key: str) -> dict[str, str]:
    authentication = _judge_authentication(config, overlay)
    if _installed_agent(config.agents[0]):
        if overlay.endpoint_url in authentication and authentication[overlay.endpoint_url] != candidate_api_key:
            raise ValueError("Candidate and judge credentials conflict for the same host endpoint")
        authentication[overlay.endpoint_url] = candidate_api_key
    return authentication


def _model_environment(config: JobConfig) -> dict[str, str]:
    if config.environment.type not in _MODEL_RELAY_ENVIRONMENTS or not config.environment.kwargs.get("model_endpoint"):
        return {}
    model_name = str(config.environment.kwargs.get("model_name", _STABLE_MODEL))
    return {
        **_TASK_MODEL_ENVIRONMENT,
        "TOGETHER_API_KEY": "EMPTY",
        "CUSTOM_API_KEY": "EMPTY",
        "TAU2_USER_MODEL": "openai/" + model_name,
        "TAU2_NL_ASSERTIONS_MODEL": "openai/" + model_name,
    }


def _stable_config(config: JobConfig) -> JobConfig:
    agent = _stable_agent(config.agents[0])
    stable = config.model_copy(
        update={
            "job_name": _STABLE_JOB_NAME,
            "jobs_dir": _STABLE_JOBS_DIR,
            "agents": [agent],
            "user_agent": _pinned_pi_agent(config.user_agent) if config.user_agent is not None else None,
        }
    )
    return JobConfig.model_validate(stable.model_dump(mode="python"), extra="forbid")


def _normalized(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalized(value[key]) for key in sorted(value)}
    if isinstance(value, set | frozenset):
        members = [_normalized(member) for member in value]
        return sorted(members, key=_stable_json)
    if isinstance(value, list | tuple):
        return [_normalized(member) for member in value]
    if isinstance(value, Enum):
        return _normalized(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_policy_json(config: JobConfig) -> str:
    return _stable_json(_normalized(config.model_dump(mode="python")))


def _preflight_one(path: Path, model_agent_kwargs: Mapping[str, object], judge_model: str | None) -> dict[str, object]:
    document = _document(path)
    config = JobConfig.model_validate(document, extra="forbid")
    if config.tasks:
        raise ValueError("Harbor config tasks are incompatible with the shared launcher; declare one dataset")
    agent = _single_entry(config, "agents")
    dataset = _single_entry(config, "datasets")
    assert isinstance(agent, AgentConfig)
    assert isinstance(dataset, DatasetConfig)
    agent_name = _validate_agent(agent)
    with task_environment(_model_environment(config)):
        environment_name = _validate_environment(config)
    dataset_metadata = _dataset_metadata(dataset, _raw_dataset_path(document))

    stable_config = _stable_config(config)
    stable_policy_json = _stable_policy_json(stable_config)
    with tempfile.TemporaryDirectory(prefix="marin-harbor-preflight-job-") as jobs_dir:
        dataset_path = (
            str((path.parent / dataset_metadata.selector).resolve())
            if dataset_metadata.kind == _DatasetKind.LOCAL
            else None
        )
        effective = _effective_config(
            stable_config,
            RuntimeOverlay(
                job_name=_STABLE_JOB_NAME,
                jobs_dir=jobs_dir,
                dataset_path=dataset_path,
                endpoint_url=_STABLE_ENDPOINT,
                served_model=_STABLE_MODEL,
                model_identity=_STABLE_MODEL,
                task_limit=None,
                model_agent_kwargs=dict(model_agent_kwargs),
                # Validate hosted judge capabilities before serving allocates an endpoint.
                verifier_env=(
                    {"OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": _STABLE_ENDPOINT, "MODEL_NAME": judge_model}
                    if judge_model is not None
                    else {}
                ),
                archive_root=jobs_dir,
                archive_dataset=dataset_metadata.selector,
            ),
        )
        effective_agent = effective.agents[0]
        if effective_agent.name == AgentName.PI:
            try:
                AgentFactory.create_agent_from_name(
                    AgentName.PI,
                    logs_dir=Path(jobs_dir) / "agent",
                    model_name=effective_agent.model_name,
                    model_alias=effective_agent.model_alias,
                    **effective_agent.kwargs,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid hosted Pi configuration: {exc}. "
                    "Check the model YAML's agent.agent_kwargs or the Harbor policy's agents[0].kwargs; "
                    "hosted Pi requires an explicit thinking_format."
                ) from exc
        with task_environment(_model_environment(effective)):
            job = asyncio.run(Job.create(effective))
            _validate_job_tasks(job)
    if len(job.benchmark_metadata) != 1:
        raise ValueError("Harbor shared launcher requires exactly one benchmark descriptor")
    infrastructure_errors = errors_by_category(ErrorCategory.INFRASTRUCTURE)
    agent_errors = errors_by_category(ErrorCategory.AGENT)
    passthrough_errors = errors_by_category(ErrorCategory.PASSTHROUGH)
    undecided_errors = known_error_types() - infrastructure_errors - agent_errors - passthrough_errors
    model_info = effective.agents[0].kwargs[MODEL_INFO_KEY]
    return {
        "stable_policy_json": stable_policy_json,
        "digest": f"sha256:{hashlib.sha256(stable_policy_json.encode()).hexdigest()}",
        "dataset_kind": dataset_metadata.kind,
        "dataset_selector": dataset_metadata.selector,
        "dataset_revision": dataset_metadata.revision,
        "agent": agent_name,
        "environment": environment_name,
        "verifier_env_keys": sorted(
            {
                name
                for declaration in _environment_declarations(config)
                for name, default in get_required_host_vars(declaration)
                if default is None
            }
        ),
        "error_taxonomy": {
            "infrastructure": sorted(infrastructure_errors),
            "agent": sorted(agent_errors),
            "passthrough": sorted(passthrough_errors),
            "undecided": sorted(undecided_errors),
        },
        "max_input_tokens": model_info[MAX_INPUT_TOKENS_KEY],
        "max_output_tokens": model_info[MAX_OUTPUT_TOKENS_KEY],
        "benchmark_metadata": job.benchmark_metadata[0].model_dump(mode="json"),
        "trials_per_task": config.n_attempts * len(config.agents),
    }


def _preflight(request_path: Path) -> None:
    requests = json.loads(request_path.read_text())
    if not isinstance(requests, list):
        raise ValueError("Harbor preflight request must be a list")
    results: list[dict[str, object]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("Harbor preflight request entries must be objects")
        path = request.get("path")
        model_agent_kwargs = request.get("model_agent_kwargs", {})
        judge_model = request["judge_model"]
        if not isinstance(path, str):
            raise ValueError("Harbor preflight request path must be a string")
        if not isinstance(model_agent_kwargs, Mapping):
            raise ValueError("Harbor preflight model agent kwargs must be a mapping")
        if judge_model is not None and not isinstance(judge_model, str):
            raise ValueError("Harbor preflight judge model must be a string or null")
        results.append(_preflight_one(Path(path), model_agent_kwargs, judge_model))
    sys.stdout.write(json.dumps(results, ensure_ascii=False, separators=(",", ":")))


async def _run(config: JobConfig) -> None:
    with task_environment(_model_environment(config)):
        job = await Job.create(config)
        _validate_job_tasks(job)
        await job.run()


def effective_job_config(policy_path: Path, overlay_path: Path) -> JobConfig:
    """Parse an opaque policy, apply the Marin overlay, and validate the full job."""
    policy = JobConfig.model_validate_json(policy_path.read_text(), extra="forbid")
    if policy.tasks or len(policy.agents) != 1 or len(policy.datasets) != 1:
        raise ValueError("Harbor stable policy violates the shared launcher contract")
    overlay = RuntimeOverlay.model_validate_json(overlay_path.read_text())
    return _effective_config(policy, overlay)


def _diagnostic(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.json(include_url=False, include_input=False)
    return str(exc)


def main() -> None:
    try:
        # Harbor uses UPath directly, so configure its filesystem credentials
        # before parsing remote job paths or starting artifact writers.
        fsspec.config.conf.setdefault("s3", {}).update(s3_credential_options())
        command = sys.argv[1]
        if command == "preflight":
            _preflight(Path(sys.argv[2]))
            return
        if command != "run":
            raise ValueError(f"unknown command {command!r}")
        policy = JobConfig.model_validate_json(Path(sys.argv[2]).read_text(), extra="forbid")
        config = effective_job_config(Path(sys.argv[2]), Path(sys.argv[3]))
        overlay = RuntimeOverlay.model_validate_json(Path(sys.argv[3]).read_text())
    except (IndexError, json.JSONDecodeError, OSError, TypeError, ValueError, ValidationError) as exc:
        print(_diagnostic(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    with (
        host_backend_credentials(
            daytona_api_key=os.environ.get("DAYTONA_API_KEY"),
            daytona_api_url=os.environ.get("DAYTONA_API_URL", DAYTONA_API_URL),
            daytona_target=os.environ.get("DAYTONA_TARGET"),
        ),
        host_model_authentication(
            _model_authentication(policy, overlay, candidate_api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        ),
    ):
        asyncio.run(_run(config))


if __name__ == "__main__":
    main()
