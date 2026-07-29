# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Harbor policies inside the pinned external environment."""

import asyncio
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from harbor.agents.factory import AgentFactory  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor.environments.factory import _load_environment_class  # pyrefly: ignore[missing-import]
from harbor.job import Job  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor_config import JobConfig  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor_config.models.agent.name import AgentName  # pyrefly: ignore[missing-import]
from harbor_config.models.job.config import DatasetConfig  # pyrefly: ignore[missing-import]
from harbor_config.models.trial.config import AgentConfig  # pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, ValidationError

_HOSTED_VLLM_PROVIDER = "hosted_vllm"
_HOSTED_VLLM_DISPLAY_NAME = "Hosted vLLM"
_OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible"
_OPENCODE_AGENT = "opencode"
_STABLE_JOB_NAME = "__marin_job__"
_STABLE_JOBS_DIR = "/__marin_jobs__"
_STABLE_MODEL = "__marin_model__"
_STABLE_ENDPOINT = "http://marin.invalid/v1"
_PLACEHOLDER_DATASET_PATH = "/__marin_dataset__"
_DEFAULT_MODEL_INFO = MappingProxyType(
    {
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
)


class RuntimeOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    task_limit: int | None
    model_agent_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _DatasetMetadata:
    kind: str
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


def _job_config_from_document(document: Mapping[str, object]) -> JobConfig:
    return JobConfig.model_validate(document, extra="forbid")


def _single_entry(config: JobConfig, field_name: str) -> object:
    values = getattr(config, field_name)
    if len(values) != 1:
        raise ValueError(f"Harbor config must declare exactly one {field_name.removesuffix('s')}")
    return values[0]


def _resolve_import_path(import_path: str, label: str) -> None:
    if ":" not in import_path:
        raise ValueError(f"Harbor {label} import path must use module.path:ClassName")
    module_path, class_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_path)
        getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Harbor {label} import path could not be resolved") from exc


def _validate_agent(agent: AgentConfig) -> str:
    if agent.import_path is not None:
        if agent.mode == "local":
            raise ValueError("Harbor local-mode agents must use a supported agent name")
        _resolve_import_path(agent.import_path, "agent")
        return agent.import_path
    if agent.name is None or agent.name not in AgentName.values():
        raise ValueError("Harbor config agent name is not supported by the pinned runtime")
    agent_name = AgentName(agent.name)
    agent_registry = AgentFactory._LOCAL_AGENT_MAP if agent.mode == "local" else AgentFactory._AGENT_MAP
    if agent_name not in agent_registry:
        raise ValueError("Harbor config agent is not available in the pinned runtime")
    return agent.name


def _validate_environment(config: JobConfig) -> str:
    environment = config.environment
    if environment.import_path is not None:
        _resolve_import_path(environment.import_path, "environment")
        return environment.import_path
    if environment.type is None:
        raise ValueError("Harbor config environment must have a type or import_path")
    _load_environment_class(environment.type)
    return environment.type.value


def _raw_dataset_path(document: Mapping[str, object]) -> object:
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], Mapping):
        return None
    return datasets[0].get("path")


def _dataset_metadata(
    dataset: DatasetConfig,
    raw_path: object,
) -> _DatasetMetadata:
    if isinstance(raw_path, str) and raw_path.startswith("hf://"):
        raise ValueError("Harbor hf:// sources must use datasets[].name, not datasets[].path")
    if dataset.path is not None:
        if dataset.path.is_absolute():
            raise ValueError("Harbor local dataset paths must be relative to the config file")
        return _DatasetMetadata("local", str(dataset.path), None)

    assert dataset.name is not None
    revision = dataset.ref or dataset.version
    if dataset.name.startswith("hf://"):
        selector = dataset.name.removeprefix("hf://")
        repository_parts = selector.split("/")
        if len(repository_parts) != 2 or any(not part for part in repository_parts):
            raise ValueError("Harbor hf:// dataset names must identify an org/repository")
        if dataset.version is not None:
            raise ValueError("Harbor hf:// datasets must use ref for their revision")
        return _DatasetMetadata("hugging_face", selector, revision)
    return _DatasetMetadata("harbor_registry", dataset.name, revision)


def _model_info(value: object) -> dict[str, Any]:
    configured = {} if value is None else value
    if not isinstance(configured, Mapping):
        raise ValueError("Harbor agent model_info must be a mapping")
    return {**_DEFAULT_MODEL_INFO, **configured}


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
                "options": {**options, "baseURL": endpoint_url},
            },
        },
    }


def _agent_with_runtime(
    agent: AgentConfig,
    *,
    endpoint_url: str,
    served_model: str,
    model_agent_kwargs: Mapping[str, object],
    include_model_defaults: bool,
) -> AgentConfig:
    policy_kwargs = agent.kwargs
    kwargs = {**model_agent_kwargs, **policy_kwargs}
    if include_model_defaults:
        kwargs["model_info"] = _model_info(kwargs.get("model_info"))
    elif "model_info" in kwargs:
        _model_info(kwargs["model_info"])
    kwargs["api_base"] = endpoint_url
    if agent.name == _OPENCODE_AGENT:
        kwargs["opencode_config"] = _opencode_config(kwargs.get("opencode_config", {}), endpoint_url)
    return AgentConfig.model_validate(
        {
            **agent.model_dump(mode="python"),
            "model_name": f"{_HOSTED_VLLM_PROVIDER}/{served_model}",
            "kwargs": kwargs,
        },
        extra="forbid",
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
    agent = _agent_with_runtime(
        config.agents[0],
        endpoint_url=overlay.endpoint_url,
        served_model=overlay.served_model,
        model_agent_kwargs=overlay.model_agent_kwargs,
        include_model_defaults=True,
    )
    dataset = _dataset_with_runtime(
        config.datasets[0],
        dataset_path=overlay.dataset_path,
        task_limit=overlay.task_limit,
    )
    effective = config.model_copy(
        update={
            "job_name": overlay.job_name,
            "jobs_dir": overlay.jobs_dir,
            "agents": [agent],
            "datasets": [dataset],
        }
    )
    return JobConfig.model_validate(effective.model_dump(mode="json"), extra="forbid")


def _stable_config(config: JobConfig) -> JobConfig:
    agent = _agent_with_runtime(
        config.agents[0],
        endpoint_url=_STABLE_ENDPOINT,
        served_model=_STABLE_MODEL,
        model_agent_kwargs={},
        include_model_defaults=False,
    )
    stable = config.model_copy(
        update={
            "job_name": _STABLE_JOB_NAME,
            "jobs_dir": _STABLE_JOBS_DIR,
            "agents": [agent],
        }
    )
    return JobConfig.model_validate(stable.model_dump(mode="json"), extra="forbid")


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


def _preflight_one(path: Path, model_agent_kwargs: Mapping[str, object]) -> dict[str, object]:
    document = _document(path)
    config = _job_config_from_document(document)
    if config.tasks:
        raise ValueError("Harbor config tasks are incompatible with the shared launcher; declare one dataset")
    agent = _single_entry(config, "agents")
    dataset = _single_entry(config, "datasets")
    assert isinstance(agent, AgentConfig)
    assert isinstance(dataset, DatasetConfig)
    agent_name = _validate_agent(agent)
    environment_name = _validate_environment(config)
    dataset_metadata = _dataset_metadata(dataset, _raw_dataset_path(document))

    stable_config = _stable_config(config)
    stable_policy_json = _stable_policy_json(stable_config)
    placeholder_dataset_path = None if dataset_metadata.kind == "harbor_registry" else _PLACEHOLDER_DATASET_PATH
    _effective_config(
        stable_config,
        RuntimeOverlay(
            job_name=_STABLE_JOB_NAME,
            jobs_dir=_STABLE_JOBS_DIR,
            dataset_path=placeholder_dataset_path,
            endpoint_url=_STABLE_ENDPOINT,
            served_model=_STABLE_MODEL,
            task_limit=None,
            model_agent_kwargs=dict(model_agent_kwargs),
        ),
    )
    return {
        "stable_policy_json": stable_policy_json,
        "digest": f"sha256:{hashlib.sha256(stable_policy_json.encode()).hexdigest()}",
        "dataset_kind": dataset_metadata.kind,
        "dataset_selector": dataset_metadata.selector,
        "dataset_revision": dataset_metadata.revision,
        "agent": agent_name,
        "environment": environment_name,
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
        if not isinstance(path, str):
            raise ValueError("Harbor preflight request path must be a string")
        if not isinstance(model_agent_kwargs, Mapping):
            raise ValueError("Harbor preflight model agent kwargs must be a mapping")
        results.append(_preflight_one(Path(path), model_agent_kwargs))
    sys.stdout.write(json.dumps(results, ensure_ascii=False, separators=(",", ":")))


async def _run(config: JobConfig) -> None:
    job = await Job.create(config)
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
        command = sys.argv[1]
        if command == "preflight":
            _preflight(Path(sys.argv[2]))
            return
        if command != "run":
            raise ValueError(f"unknown command {command!r}")
        config = effective_job_config(Path(sys.argv[2]), Path(sys.argv[3]))
    except (IndexError, json.JSONDecodeError, OSError, TypeError, ValueError, ValidationError) as exc:
        print(_diagnostic(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
