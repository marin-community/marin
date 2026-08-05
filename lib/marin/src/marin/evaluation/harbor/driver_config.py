# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opaque Harbor policies and the isolated-driver process protocol."""

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rigging.config_discovery import find_project_root
from rigging.tunnel import terminate_process_group

from marin.evaluation.isolated_driver import (
    ISOLATED_REQUEST_MODE,
    capture_driver,
    driver_failure,
    isolated_driver_environment,
)
from marin.external_dependencies import HARBOR
from marin.inference.iris import InferenceBackendState

_TRIAL_DRIVER = Path(__file__).with_name("trial_driver.py")
_DRIVER_PYTHONPATH = str(Path(__file__).parents[3])
# Harbor can exhaust a trial's upstream retry budget in tens of seconds when an endpoint disappears.
_BACKEND_POLL_SECONDS = 5.0
_DRIVER_TERMINATION_GRACE_SECONDS = 30.0


class HarborBackendsUnavailable(RuntimeError):
    """Harbor stopped because its model backends are not ready."""


# Object-store credentials and config the isolated driver needs to write each trial straight to a
# remote ``jobs_dir`` (CoreWeave S3, GCS). fsspec reads the ``FSSPEC_S3`` block natively; ``s3fs``
# reads the ``AWS_*`` variables and ``gcsfs`` reads the ``GOOGLE_*`` ones. Resolved present-only from
# the eval pod's ambient environment (an Iris CoreWeave task carries the ``AWS_*``/``FSSPEC_S3`` set
# via ``iris-task-env``; GCS runs on the workload's metadata-server identity with no key file).
_DRIVER_STORAGE_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "FSSPEC_S3",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
)

HARBOR_PACKAGES = (HARBOR.requirement(), *HARBOR.runtime_requirements)
HARBOR_RUNTIME = "; ".join(HARBOR_PACKAGES)

# The isolated driver runs against the fully pinned lock under this directory, not a loose ``--with``
# resolution: the git-branch and pre-release pins (harbor, litellm) drift daily, and only the locked
# set is validated to import and to carry the fsspec backends the remote ``jobs_dir`` needs.
_HARBOR_ENV_CONFIG = ("config", "external", "harbor")

logger = logging.getLogger(__name__)


def _harbor_env_dir() -> Path:
    """The locked isolated-driver project (``config/external/harbor``) in the Marin workspace."""
    workspace_root = find_project_root(Path(__file__))
    if workspace_root is None:
        raise RuntimeError("Harbor driver requires a Marin workspace to locate its pinned environment")
    return workspace_root.joinpath(*_HARBOR_ENV_CONFIG)


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
    return [
        "uv",
        "run",
        "--isolated",
        "--project",
        str(_harbor_env_dir()),
        "python",
        str(_TRIAL_DRIVER),
        command,
        *(str(path) for path in paths),
    ]


def _driver_environment(driver_env: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = isolated_driver_environment(_DRIVER_STORAGE_ENV_KEYS, driver_env)
    environment["PYTHONPATH"] = _DRIVER_PYTHONPATH
    return environment


def _stream_driver(
    command: list[str],
    driver_env: Mapping[str, str],
    backend_state: Callable[[], InferenceBackendState],
) -> None:
    process = subprocess.Popen(
        command,
        env=_driver_environment(driver_env),
        start_new_session=True,
    )
    try:
        while True:
            try:
                return_code = process.wait(timeout=_BACKEND_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if process.poll() is not None:
                    continue
                _raise_for_backend_state(backend_state())
    except BaseException:
        terminate_process_group(process, grace_period=_DRIVER_TERMINATION_GRACE_SECONDS)
        raise

    if return_code:
        _raise_for_backend_state(backend_state())
        exc = subprocess.CalledProcessError(return_code, command)
        raise driver_failure(exc) from exc


def _raise_for_backend_state(state: InferenceBackendState) -> None:
    if state is InferenceBackendState.RECOVERING:
        raise HarborBackendsUnavailable("inference backends are not ready")
    if state is InferenceBackendState.FINISHED:
        raise RuntimeError("inference backend finished while Harbor was running")


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
        request_path.chmod(ISOLATED_REQUEST_MODE)
        try:
            completed = capture_driver(_driver_command("preflight", request_path), _driver_environment())
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
    backend_state: Callable[[], InferenceBackendState],
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
        policy_path.chmod(ISOLATED_REQUEST_MODE)
        overlay_path.chmod(ISOLATED_REQUEST_MODE)
        command = _driver_command("run", policy_path, overlay_path)
        logger.info("running Harbor driver: %s", " ".join(command))
        _stream_driver(command, driver_env, backend_state)
