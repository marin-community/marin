# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opaque Harbor policies and the isolated-driver process protocol."""

import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.evaluation.eval_env import env_vars_from_keys
from marin.external_dependencies import HARBOR

_DEFAULT_MODEL_INFO = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}
_HOSTED_VLLM_PROVIDER = "hosted_vllm"
_TRIAL_DRIVER = Path(__file__).with_name("trial_driver.py")
_DRIVER_PYTHONPATH = str(Path(__file__).parents[3])
_DRIVER_SYSTEM_ENV_KEYS = (
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "PATH",
    "PYTHONHASHSEED",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

HARBOR_PACKAGES = (HARBOR.requirement(), *HARBOR.runtime_requirements)
HARBOR_RUNTIME = "; ".join(HARBOR_PACKAGES)

logger = logging.getLogger(__name__)


class HarborDatasetKind(StrEnum):
    """How Marin obtains the dataset before the isolated driver runs."""

    HARBOR_REGISTRY = "harbor_registry"
    HUGGING_FACE = "hugging_face"
    LOCAL = "local"


@dataclass(frozen=True)
class ValidatedHarborConfig:
    """An opaque validated policy plus Marin-owned launch metadata."""

    stable_policy_json: str
    digest: str
    dataset_kind: HarborDatasetKind
    dataset_selector: str
    dataset_revision: str | None
    config_dir: Path
    agent: str
    environment: str

    @property
    def record_dataset(self) -> str:
        if self.dataset_kind == HarborDatasetKind.HUGGING_FACE:
            return f"hf://{self.dataset_selector}"
        return self.dataset_selector

    @property
    def record_revision(self) -> str:
        return self.dataset_revision or "unversioned"


@dataclass(frozen=True)
class HarborRuntimeOverlay:
    """Marin-owned values applied after the isolated driver reparses the policy."""

    job_name: str
    jobs_dir: str
    dataset_path: str | None
    endpoint_url: str
    served_model: str
    task_limit: int | None
    model_agent_kwargs: Mapping[str, object]


# These authoring dataclasses remain only for the registry policies not converted
# in the first YAML migration. They are removed with that mechanical follow-up.
@dataclass(frozen=True)
class HarborRetryConfig:
    max_retries: int = 0
    exclude_exceptions: tuple[str, ...] = ()
    wait_multiplier: float = 1.0
    min_wait: float = 1.0
    max_wait: float = 60.0


@dataclass(frozen=True)
class HarborEnvironmentConfig:
    environment_type: str
    force_build: bool = False
    delete: bool = True
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarborAgentConfig:
    name: str
    max_output_tokens: int = 8192
    max_timeout: float | None = None
    setup_timeout: float | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarborVerifierConfig:
    max_timeout: float | None = None


@dataclass(frozen=True)
class HarborRunConfig:
    """Temporary Python authoring form for Harbor policies not yet moved to YAML."""

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


def legacy_harbor_policy_document(run: HarborRunConfig) -> dict[str, object]:
    """Lower a not-yet-migrated Python policy for isolated preflight."""
    agent_kwargs = dict(run.agent.kwargs)
    configured_model_info = agent_kwargs.get("model_info")
    if configured_model_info is not None and not isinstance(configured_model_info, Mapping):
        raise ValueError("Harbor agent model_info must be a mapping")
    if configured_model_info is not None or run.agent.max_output_tokens != _DEFAULT_MODEL_INFO["max_output_tokens"]:
        agent_kwargs["model_info"] = {
            **_DEFAULT_MODEL_INFO,
            **(configured_model_info or {}),
            "max_output_tokens": run.agent.max_output_tokens,
        }

    if run.dataset.startswith("hf://"):
        dataset = {"name": run.dataset, "ref": run.revision, "n_tasks": run.task_limit}
    else:
        selector = {"ref": run.revision} if "/" in run.dataset else {"version": run.revision}
        dataset = {"name": run.dataset, **selector, "n_tasks": run.task_limit}

    return {
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
                "max_timeout_sec": run.agent.max_timeout,
                "override_setup_timeout_sec": run.agent.setup_timeout,
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [dataset],
    }


def _driver_command(command: str, *paths: Path) -> list[str]:
    args = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--prerelease=allow",
    ]
    for package in HARBOR_PACKAGES:
        args.extend(("--with", package))
    args.extend(("python", str(_TRIAL_DRIVER), command, *(str(path) for path in paths)))
    return args


def _driver_environment(driver_env: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = env_vars_from_keys(_DRIVER_SYSTEM_ENV_KEYS)
    environment.update(driver_env or {})
    environment["PYTHONPATH"] = _DRIVER_PYTHONPATH
    return environment


def _driver_failure(exc: subprocess.CalledProcessError) -> ValueError:
    stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
    stdout = exc.stdout.strip() if isinstance(exc.stdout, str) else ""
    detail = stderr or stdout or f"driver exited with status {exc.returncode}"
    return ValueError(detail)


def _capture_driver(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_driver_environment(),
        )
    except subprocess.CalledProcessError as exc:
        raise _driver_failure(exc) from exc


def _stream_driver(command: list[str], driver_env: Mapping[str, str]) -> None:
    try:
        subprocess.run(command, check=True, env=_driver_environment(driver_env))
    except subprocess.CalledProcessError as exc:
        raise _driver_failure(exc) from exc


def _validated_config(payload: object, path: Path) -> ValidatedHarborConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Harbor preflight returned a non-object result for {path}")

    def required_string(name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Harbor preflight returned invalid {name!r} metadata for {path}")
        return value

    revision = payload.get("dataset_revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError(f"Harbor preflight returned invalid dataset revision metadata for {path}")
    try:
        dataset_kind = HarborDatasetKind(required_string("dataset_kind"))
    except ValueError as exc:
        raise ValueError(f"Harbor preflight returned an unknown dataset kind for {path}") from exc

    return ValidatedHarborConfig(
        stable_policy_json=required_string("stable_policy_json"),
        digest=required_string("digest"),
        dataset_kind=dataset_kind,
        dataset_selector=required_string("dataset_selector"),
        dataset_revision=revision,
        config_dir=path.resolve().parent,
        agent=required_string("agent"),
        environment=required_string("environment"),
    )


def preflight_harbor_configs(
    requests: Sequence[tuple[Path, Mapping[str, object]]],
) -> tuple[ValidatedHarborConfig, ...]:
    """Validate Harbor policies and placeholder effective jobs in one isolated process."""
    if not requests:
        return ()

    request_payload = [
        {
            "path": str(path.resolve()),
            "model_agent_kwargs": dict(model_agent_kwargs),
        }
        for path, model_agent_kwargs in requests
    ]
    with tempfile.TemporaryDirectory(prefix="marin-harbor-preflight-") as temp_dir:
        request_path = Path(temp_dir) / "requests.json"
        try:
            request_path.write_text(json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError) as exc:
            raise ValueError("Harbor model agent kwargs must be JSON-serializable") from exc
        request_path.chmod(0o600)
        try:
            completed = _capture_driver(_driver_command("preflight", request_path))
        except ValueError as exc:
            paths = ", ".join(str(path) for path, _ in requests)
            raise ValueError(f"invalid Harbor config in [{paths}]: {exc}") from exc

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Harbor preflight returned invalid JSON") from exc
    if not isinstance(response, list) or len(response) != len(requests):
        raise ValueError(
            f"Harbor preflight returned {len(response) if isinstance(response, list) else 'invalid'} "
            f"result(s) for {len(requests)} request(s)"
        )
    return tuple(_validated_config(payload, path) for payload, (path, _) in zip(response, requests, strict=True))


def preflight_harbor_config(
    path: Path,
    model_agent_kwargs: Mapping[str, object],
) -> ValidatedHarborConfig:
    """Validate one Harbor policy and a placeholder effective job."""
    return preflight_harbor_configs(((path, model_agent_kwargs),))[0]


def run_harbor_driver(
    config: ValidatedHarborConfig,
    overlay: HarborRuntimeOverlay,
    driver_env: Mapping[str, str],
) -> None:
    """Apply a runtime overlay and run one Harbor job in the isolated environment."""
    runtime = {"version": HARBOR.version, "commit": HARBOR.commit}
    logger.info("Harbor runtime: %s", json.dumps(runtime, sort_keys=True), extra={"harbor_runtime": runtime})
    with tempfile.TemporaryDirectory(prefix="marin-harbor-run-") as temp_dir:
        policy_path = Path(temp_dir) / "policy.json"
        overlay_path = Path(temp_dir) / "overlay.json"
        policy_path.write_text(config.stable_policy_json)
        try:
            overlay_path.write_text(
                json.dumps(
                    {
                        **asdict(overlay),
                        "model_agent_kwargs": dict(overlay.model_agent_kwargs),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Harbor runtime overlay must be JSON-serializable") from exc
        policy_path.chmod(0o600)
        overlay_path.chmod(0o600)
        command = _driver_command("run", policy_path, overlay_path)
        logger.info("running Harbor driver: %s", " ".join(command))
        _stream_driver(command, driver_env)
