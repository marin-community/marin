# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opaque Harbor policies and the isolated-driver process protocol."""

import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rigging.config_discovery import find_project_root

from marin.evaluation.eval_env import env_vars_from_keys
from marin.external_dependencies import HARBOR

_TRIAL_DRIVER = Path(__file__).with_name("trial_driver.py")
_DRIVER_PYTHONPATH = str(Path(__file__).parents[3])
_OWNER_ONLY_MODE = 0o600
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
    workspace_dataset_path: Path | None
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
    dataset_selector = required_string("dataset_selector")
    workspace_dataset_path = None
    if dataset_kind == HarborDatasetKind.LOCAL:
        workspace_root = find_project_root(path)
        if workspace_root is None:
            raise ValueError(f"Harbor local dataset source must be inside a Marin workspace: {path}")
        local_dataset_path = (path.resolve().parent / dataset_selector).resolve()
        try:
            workspace_dataset_path = local_dataset_path.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"Harbor local dataset source must be inside the Marin workspace: {local_dataset_path}"
            ) from exc

    return ValidatedHarborConfig(
        stable_policy_json=required_string("stable_policy_json"),
        digest=required_string("digest"),
        dataset_kind=dataset_kind,
        dataset_selector=dataset_selector,
        dataset_revision=revision,
        workspace_dataset_path=workspace_dataset_path,
        agent=required_string("agent"),
        environment=required_string("environment"),
    )


def preflight_harbor_configs(
    requests: Sequence[tuple[Path, Mapping[str, object]]],
) -> tuple[ValidatedHarborConfig, ...]:
    """Validate Harbor policies and their placeholder effective jobs before launch."""
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
        request_path.chmod(_OWNER_ONLY_MODE)
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
        policy_path.chmod(_OWNER_ONLY_MODE)
        overlay_path.chmod(_OWNER_ONLY_MODE)
        command = _driver_command("run", policy_path, overlay_path)
        logger.info("running Harbor driver: %s", " ".join(command))
        _stream_driver(command, driver_env)
