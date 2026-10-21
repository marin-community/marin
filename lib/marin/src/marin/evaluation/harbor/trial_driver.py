# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Harbor policies inside the pinned external environment."""

import asyncio
import base64
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.factory import AgentFactory  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor.agents.installed.acp_registry import (  # pyrefly: ignore[missing-import]
    is_acp_registry_shorthand,
    parse_registry_spec,
)
from harbor.agents.installed.mini_swe_agent import MiniSweAgent  # pyrefly: ignore[missing-import]
from harbor.agents.terminus_2.tmux_session import (  # pyrefly: ignore[missing-import]
    TmuxCommandError,
    TmuxSession,
    TmuxSessionEndedError,
)
from harbor.environments.daytona.environment import DaytonaEnvironment  # pyrefly: ignore[missing-import]
from harbor.environments.factory import _load_environment_class  # pyrefly: ignore[missing-import]
from harbor.job import Job  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor.models.job.lock import TrialLock  # pyrefly: ignore[missing-import]
from harbor.verifier.verifier import Verifier  # pyrefly: ignore[missing-import]
from harbor_config import JobConfig  # pyrefly: ignore[missing-import]  # installed by external driver
from harbor_config.env import get_required_host_vars  # pyrefly: ignore[missing-import]
from harbor_config.errors import ErrorCategory, errors_by_category, known_error_types  # pyrefly: ignore[missing-import]
from harbor_config.models.agent.name import AgentName  # pyrefly: ignore[missing-import]
from harbor_config.models.job.config import ArchiveConfig, DatasetConfig  # pyrefly: ignore[missing-import]
from harbor_config.models.trial.config import AgentConfig, TrialConfig  # pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, ValidationError
from rigging.filesystem.storage_path import StoragePath

from marin.evaluation.harbor.agent_context import (
    MAX_INPUT_TOKENS_KEY,
    MAX_OUTPUT_TOKENS_KEY,
    MODEL_INFO_KEY,
    reconciled_model_info,
)
from marin.evaluation.harbor.driver_protocol import FULL_GIT_COMMIT_LENGTH

_HOSTED_VLLM_PROVIDER = "hosted_vllm"
_HOSTED_VLLM_DISPLAY_NAME = "Hosted vLLM"
_OPENAI_COMPATIBLE_PACKAGE = "@ai-sdk/openai-compatible"
_OPENCODE_AGENT = "opencode"
_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
_PI_ACP_REGISTRY_ID = "pi-acp"
_TERMINUS_2_AGENT = "terminus-2"
_LLM_CALL_KWARGS_KEY = "llm_call_kwargs"
_MAX_TOKENS_KEY = "max_tokens"
_MINI_SWE_LITELLM_VERSION = "1.105.1"
_MINI_SWE_CONSTRAINT_PATH = "/tmp/marin-mini-swe.constraints"
_STABLE_JOB_NAME = "__marin_job__"
_STABLE_JOBS_DIR = "/__marin_jobs__"
_STABLE_MODEL = "__marin_model__"
_STABLE_ENDPOINT = "http://marin.invalid/v1"
_HF_DATASET_PREFIX = "hf://"


class RuntimeOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    task_limit: int | None
    model_agent_kwargs: dict[str, Any]
    verifier_env: dict[str, str]
    archive_root: str
    archive_dataset: str
    hosted_judge_config: dict[str, Any] | None = None


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
                "options": {**options, "baseURL": endpoint_url},
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


def _agent_config(
    agent: AgentConfig,
    *,
    endpoint_url: str,
    served_model: str,
    kwargs: Mapping[str, object],
) -> AgentConfig:
    runtime_kwargs = {**kwargs, "api_base": endpoint_url}
    if agent.name == _OPENCODE_AGENT:
        runtime_kwargs["opencode_config"] = _opencode_config(kwargs.get("opencode_config", {}), endpoint_url)
    runtime_env = agent.env
    if agent.name is not None and is_acp_registry_shorthand(agent.name):
        runtime_env = {**agent.env, _OPENAI_BASE_URL_ENV: endpoint_url}
    return AgentConfig.model_validate(
        {
            **agent.model_dump(mode="python"),
            "model_name": f"{_HOSTED_VLLM_PROVIDER}/{served_model}",
            "kwargs": runtime_kwargs,
            "env": runtime_env,
        },
        extra="forbid",
    )


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
    return _agent_config(
        agent,
        endpoint_url=overlay.endpoint_url,
        served_model=overlay.served_model,
        kwargs=kwargs,
    )


def _stable_agent(agent: AgentConfig) -> AgentConfig:
    reconciled_model_info(None, agent.kwargs.get(MODEL_INFO_KEY))
    return _agent_config(
        agent,
        endpoint_url=_STABLE_ENDPOINT,
        served_model=_STABLE_MODEL,
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
    agent = _effective_agent(config.agents[0], overlay)
    dataset = _dataset_with_runtime(
        config.datasets[0],
        dataset_path=overlay.dataset_path,
        task_limit=overlay.task_limit,
    )
    effective = config.model_copy(
        update={
            "job_name": overlay.job_name,
            "jobs_dir": JobConfig(jobs_dir=str(StoragePath(overlay.jobs_dir))).jobs_dir,
            "agents": [agent],
            "datasets": [dataset],
            "verifier": config.verifier.model_copy(update={"env": {**config.verifier.env, **overlay.verifier_env}}),
            "archive": ArchiveConfig(
                root=overlay.archive_root,
                dataset=overlay.archive_dataset,
            ),
        }
    )
    return effective


def _stable_config(config: JobConfig) -> JobConfig:
    agent = _stable_agent(config.agents[0])
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


def _harbor_config_commit() -> str:
    distribution_name = importlib.metadata.packages_distributions()["harbor_config"][0]
    direct_url = json.loads(importlib.metadata.distribution(distribution_name).read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not isinstance(commit, str) or len(commit) != FULL_GIT_COMMIT_LENGTH:
        raise ValueError("Harbor distribution does not identify its pinned commit")
    return commit


def _preflight_one(path: Path, model_agent_kwargs: Mapping[str, object]) -> dict[str, object]:
    document = _document(path)
    config = JobConfig.model_validate(document, extra="forbid")
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
                task_limit=None,
                model_agent_kwargs=dict(model_agent_kwargs),
                verifier_env={},
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
                    **effective_agent.kwargs,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid hosted Pi configuration: {exc}. "
                    "Check the model YAML's agent.agent_kwargs or the Harbor policy's agents[0].kwargs; "
                    "hosted Pi requires an explicit thinking_format."
                ) from exc
        job = asyncio.run(Job.create(effective))
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
            {name for name, default in get_required_host_vars(config.verifier.env) if default is None}
        ),
        "error_taxonomy": {
            "infrastructure": sorted(infrastructure_errors),
            "agent": sorted(agent_errors),
            "passthrough": sorted(passthrough_errors),
            "undecided": sorted(undecided_errors),
            "commit": _harbor_config_commit(),
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
        if not isinstance(path, str):
            raise ValueError("Harbor preflight request path must be a string")
        if not isinstance(model_agent_kwargs, Mapping):
            raise ValueError("Harbor preflight model agent kwargs must be a mapping")
        results.append(_preflight_one(Path(path), model_agent_kwargs))
    sys.stdout.write(json.dumps(results, ensure_ascii=False, separators=(",", ":")))


@contextmanager
def _hosted_judge_transport_identity(overlay: RuntimeOverlay):
    """Permit URL rotation only for the fully bound, identical hosted judge.

    The pinned Harbor fingerprint already ignores candidate serving transport, but
    includes the judge URL. Keep stored configs and grades intact; normalize only
    the comparison key after checking the complete model-bound resume identity.
    """
    expected = overlay.hosted_judge_config
    if expected is None:
        yield
        return
    identity = json.loads((StoragePath.parse(overlay.archive_root) / "harbor_resume_identity.json").read_text())
    judge_name = expected.get("name")
    if (
        identity.get("schema_version") != 2
        or identity.get("judge") != expected
        or not isinstance(judge_name, str)
        or overlay.verifier_env.get("MODEL_NAME") != judge_name
    ):
        raise ValueError("Hosted judge transport rotation requires an identical bound judge configuration")
    original = TrialConfig.identity_fingerprint
    original_lock_key = TrialLock._equality_key

    def lock_key(trial):
        # Harbor separately compares every trial in lock.json before running.
        # Normalize a copy so the saved lock and runtime endpoint stay intact.
        env = trial.verifier.env
        if env.get("MODEL_NAME") == judge_name and "OPENAI_BASE_URL" in env:
            trial = trial.model_copy(deep=True)
            trial.verifier.env["OPENAI_BASE_URL"] = "__marin_identical_hosted_judge_transport__"
        return original_lock_key(trial)

    def fingerprint(self: TrialConfig) -> dict[str, Any]:
        value = original(self)
        verifier = value.get("verifier") or {}
        env = verifier.get("env") or {}
        if env.get("MODEL_NAME") == judge_name and "OPENAI_BASE_URL" in env:
            env["OPENAI_BASE_URL"] = "__marin_identical_hosted_judge_transport__"
        return value

    TrialConfig.identity_fingerprint = fingerprint
    TrialLock._equality_key = lock_key
    try:
        yield
    finally:
        TrialConfig.identity_fingerprint = original
        TrialLock._equality_key = original_lock_key


def _opaque_terminal_batch_command(command: str) -> str:
    if not command.startswith("if ! command -v tmux >/dev/null 2>&1; then\n"):
        return command
    if "harbor_tmux_has_session() {" not in command:
        return command
    # A process-name cleanup inside the pane must not match the transport's
    # bash/timeout argv merely because that argv contains the agent's script.
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"eval \"$(printf '%s' '{encoded}' | base64 -d)\""


@dataclass(frozen=True)
class _VerifierCpuAllocation:
    environment: Any
    cpus: int


_VERIFIER_CPU_ALLOCATION: ContextVar[_VerifierCpuAllocation | None] = ContextVar(
    "marin_verifier_cpu_allocation", default=None
)
_VERIFIER_THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _daytona_verifier_cpu_limits():
    original_verify = Verifier.verify
    original_exec = DaytonaEnvironment._sandbox_exec
    exec_signature = inspect.signature(original_exec)

    async def verify(verifier: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(verifier.environment, DaytonaEnvironment):
            return await original_verify(verifier, *args, **kwargs)
        cpus = verifier.environment._sandbox.cpu
        if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1:
            raise ValueError("Daytona verifier requires its allocated CPU count")
        token = _VERIFIER_CPU_ALLOCATION.set(_VerifierCpuAllocation(verifier.environment, cpus))
        try:
            return await original_verify(verifier, *args, **kwargs)
        finally:
            _VERIFIER_CPU_ALLOCATION.reset(token)

    async def sandbox_exec(environment: Any, command: str, *args: Any, **kwargs: Any) -> Any:
        allocation = _VERIFIER_CPU_ALLOCATION.get()
        if allocation is None or allocation.environment is not environment:
            return await original_exec(environment, command, *args, **kwargs)
        bound = exec_signature.bind(environment, command, *args, **kwargs)
        env = dict(bound.arguments.get("env") or {})
        # Daytona exposes host CPUs even when its cgroup grants one CPU. Native
        # test libraries otherwise create host-sized pools and exhaust the
        # original verifier timeout. Explicit task thread settings take priority.
        for key in _VERIFIER_THREAD_ENV_KEYS:
            env.setdefault(key, str(allocation.cpus))
        bound.arguments["env"] = env
        return await original_exec(*bound.args, **bound.kwargs)

    Verifier.verify = verify
    DaytonaEnvironment._sandbox_exec = sandbox_exec
    try:
        yield
    finally:
        Verifier.verify = original_verify
        DaytonaEnvironment._sandbox_exec = original_exec


@contextmanager
def _daytona_ended_terminal_capture():
    original = TmuxSession.send_keys_and_capture

    async def capture(session: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(session, *args, **kwargs)
        except (TmuxCommandError, TmuxSessionEndedError) as exc:
            if not isinstance(session.environment, DaytonaEnvironment):
                raise
            capture_failed = isinstance(exc, TmuxCommandError) and str(exc).startswith(
                f"{session.environment.session_id}: tmux capture failed:"
            )
            session_ended = isinstance(exc, TmuxSessionEndedError) and str(exc) == (
                f"{session.environment.session_id}: tmux session ended before "
                "Harbor could complete the requested terminal action"
            )
            if not (capture_failed or session_ended):
                raise
            # A terminal can close after Harbor's liveness check while the model
            # answers, or between commands and capture. Confirm absence before
            # following the existing session-ended path to the verifier.
            try:
                result = await session.environment.exec(
                    command=f"tmux has-session -t {shlex.quote(session._session_name)} 2>&1",
                    timeout_sec=5,
                    user=session._user,
                )
            except Exception:
                raise exc from None
            status = (result.stdout or "") + (result.stderr or "")
            if result.return_code != 1 or not any(
                marker in status for marker in ("can't find session:", "no server running on", "no sessions")
            ):
                raise
            output = await session._incremental_output_from_captures("", "")
            return output, False

    TmuxSession.send_keys_and_capture = capture
    try:
        yield
    finally:
        TmuxSession.send_keys_and_capture = original


@contextmanager
def _opaque_daytona_terminal_transport():
    original = DaytonaEnvironment._sandbox_exec

    async def sandbox_exec(environment: Any, command: str, *args: Any, **kwargs: Any) -> Any:
        return await original(environment, _opaque_terminal_batch_command(command), *args, **kwargs)

    DaytonaEnvironment._sandbox_exec = sandbox_exec
    try:
        yield
    finally:
        DaytonaEnvironment._sandbox_exec = original


@contextmanager
def _mini_swe_dependency_constraints():
    """Keep sandbox tool calls on the dependency version used before the MCP import regression."""
    original_install = MiniSweAgent.install

    async def install(agent: MiniSweAgent, environment: Any) -> None:
        original_env = agent._extra_env
        if "UV_CONSTRAINT" in original_env:
            raise ValueError("Mini-SWE dependency pin conflicts with an existing UV_CONSTRAINT")
        with tempfile.TemporaryDirectory(prefix="marin-mini-swe-constraints-") as directory:
            constraint = Path(directory) / "constraints.txt"
            constraint.write_text(f"litellm=={_MINI_SWE_LITELLM_VERSION}\n")
            await environment.upload_file(constraint, _MINI_SWE_CONSTRAINT_PATH)
        agent._extra_env = {**original_env, "UV_CONSTRAINT": _MINI_SWE_CONSTRAINT_PATH}
        try:
            await original_install(agent, environment)
            check = (
                "import importlib.metadata; version = importlib.metadata.version('litellm'); "
                f"assert version == {_MINI_SWE_LITELLM_VERSION!r}, version; print(version)"
            )
            result = await agent.exec_as_agent(
                environment,
                command='. "$HOME/.local/bin/env"; "$(uv tool dir)/mini-swe-agent/bin/python" -c ' + shlex.quote(check),
            )
            agent.logger.info("Mini-SWE LiteLLM dependency verified: %s", result.stdout.strip())
        finally:
            agent._extra_env = original_env

    MiniSweAgent.install = install
    try:
        yield
    finally:
        MiniSweAgent.install = original_install


async def _run(config: JobConfig, overlay: RuntimeOverlay) -> None:
    with (
        _hosted_judge_transport_identity(overlay),
        _daytona_verifier_cpu_limits(),
        _opaque_daytona_terminal_transport(),
        _daytona_ended_terminal_capture(),
        _mini_swe_dependency_constraints(),
    ):
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
    overlay = RuntimeOverlay.model_validate_json(Path(sys.argv[3]).read_text())
    asyncio.run(_run(config, overlay))


if __name__ == "__main__":
    main()
