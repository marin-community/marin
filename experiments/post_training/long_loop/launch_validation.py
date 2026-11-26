# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate concrete graph configs before allocating remote workers."""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import yaml
from fray.current_client import current_client
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from marin.rl.skyrl import skyrl_cpu_command
from rigging.config_discovery import find_project_root
from rigging.filesystem.cluster_config import StoreType
from rigging.filesystem.s3_compat import s3_credentials
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior, get_secret, require_secret
from rigging.timing import Duration

from experiments.post_training.long_loop.serving import serving_release

logger = logging.getLogger(__name__)


def validate_launch(
    document: str,
    directory: Path,
    *,
    model_config: bytes | None = None,
    build_worker: bool = False,
    check_vocabulary: bool = False,
    check_task_inputs: bool = False,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "launch.yaml"
    source.write_text(document)
    compiled = directory / "compiled.yaml"
    command = skyrl_cpu_command(
        "--group",
        "harbor-test",
        "python",
        "-m",
        "experiments.post_training.long_loop.validate_launch",
        str(source),
        str(compiled),
    )
    if model_config is not None:
        metadata = directory / "model-config.json"
        metadata.write_bytes(model_config)
        command.append(str(metadata))
    if build_worker:
        command.append("--build-worker")
    if check_vocabulary:
        command.append("--check-vocabulary")
    if check_task_inputs:
        command.append("--check-task-inputs")
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    (directory / "validator.log").write_text(result.stderr)
    if result.returncode:
        raise ValueError(f"Strict SkyRL composition failed before allocation:\n{result.stderr}")
    receipt = json.loads(result.stdout)
    receipt["command"] = command
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def launch_secret_names(launch: dict) -> tuple[str, ...]:
    """Name the long-loop storage, tracking and configured train/eval credentials."""
    names = set()
    sessions = launch["skyrl"]["environment"]["task_sessions"]
    for settings in (
        sessions.get("remote_judge") or {},
        sessions.get("daytona") or {},
        launch["skyrl"]["terminal_bench"].get("harbor", {}),
    ):
        names.update(value for key, value in settings.items() if key.endswith("api_key_env") and value is not None)
    for callback in launch["skyrl"]["trainer"].get("callbacks", []):
        if callback["type"] == "harbor_evaluation":
            names.add(callback["daytona_api_key_env"])
    return tuple(sorted(names))


def regional_preflight(
    document: str,
    readable_uris: tuple[str, ...],
    check_task_inputs: bool = False,
    check_vocabulary: bool = False,
    *,
    required_secret_names: tuple[str, ...],
) -> None:
    """Read evaluation dependencies and construct its worker in an isolated CPU task."""
    launch = yaml.safe_load(document)
    inputs = launch["inputs"]
    if check_vocabulary and (credentials := s3_credentials(StoreType.COREWEAVE)):
        key, secret = credentials
        os.environ.update(AWS_ACCESS_KEY_ID=key, AWS_SECRET_ACCESS_KEY=secret)
    for uri in (
        *readable_uris,
        *(str(StoragePath(item["uri"]) / item["relative_path"]) for item in inputs["validation_data"]),
    ):
        with StoragePath(uri).open("rb") as stream:
            stream.read(1)
    resolved_names = tuple(sorted(set(launch_secret_names(launch)) | set(required_secret_names)))
    for name in resolved_names:
        require_secret(name)
    with tempfile.TemporaryDirectory(prefix="long-loop-regional-preflight-") as directory:
        receipt = validate_launch(
            document,
            Path(directory),
            build_worker=True,
            check_task_inputs=check_task_inputs,
            check_vocabulary=check_vocabulary,
        )
    receipt["resolved_secret_names"] = list(resolved_names)
    logger.info("Regional evaluation CPU preflight passed: %s", json.dumps(receipt, sort_keys=True))


def validate_before_allocation(
    document: str,
    *,
    readable_uris: tuple[str, ...] = (),
    secrets: dict[str, str] | None = None,
    check_task_inputs: bool = False,
    check_vocabulary: bool = False,
) -> dict:
    """Gate long-loop workers on regional CPU admission before GPU allocation."""
    with tempfile.TemporaryDirectory(prefix="long-loop-validation-") as directory:
        validation = validate_launch(document, Path(directory))
    launch = yaml.safe_load(document)
    supplied = secrets or {}
    names = tuple(sorted(set(launch_secret_names(launch)) | set(supplied)))
    optional = {
        name: value
        for name in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN", "WANDB_API_KEY")
        if (value := get_secret(name)) is not None
    }
    secrets = {**optional, **supplied, **{name: supplied.get(name) or require_secret(name) for name in names}}
    iris = launch["iris"]
    nested = get_job_info() is not None
    request = JobRequest(
        name=f"long-loop-eval-preflight-{uuid4().hex[:8]}",
        entrypoint=Entrypoint.from_callable(
            regional_preflight,
            args=(document, readable_uris, check_task_inputs, check_vocabulary),
            kwargs={"required_secret_names": names},
        ),
        resources=ResourceConfig.with_cpu(
            cpu=4,
            ram="16GB",
            disk="64GB",
            target_cluster=None if nested else iris["target_cluster"],
            container_profile=iris.get("container_profile"),
        ),
        environment=create_environment(
            extras=["cpu"],
            secrets=secrets,
            env_vars={
                name: value
                for name in ("DAYTONA_TARGET", "DAYTONA_API_URL")
                if (value := os.environ.get(name)) is not None
            },
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        timeout=Duration.from_minutes(30),
    )
    context = (
        nullcontext(None)
        if nested
        else open_iris_client(
            config_file=Path(iris["parent_cluster_config"] or iris["cluster_config"]), workspace=find_project_root()
        )
    )
    with context as iris_client:
        client = current_client() if nested else FrayIrisClient.from_iris_client(iris_client)
        handle = client.submit(request)
        logger.info("Waiting for regional evaluation CPU preflight %s", handle.job_id)
        handle.wait(raise_on_failure=True)
    validation["resolved_secret_names"] = list(names)
    validation["regional_preflight_job_id"] = handle.job_id
    return validation


def _vllm_cpu_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        environment.pop(name, None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(
        Path(tempfile.gettempdir()) / f"long-loop-context-{serving_release().source_commit}"
    )
    return environment


def validate_selected_context(document: str) -> dict:
    """Validate the actual export config on CPU before allocating any judge GPUs."""
    launch = yaml.safe_load(document)
    model = launch["inputs"]["model"]
    generator = launch["skyrl"]["generator"]
    overrides = generator.get("engine_init_kwargs", {}).get("hf_overrides", {})
    requested = launch["skyrl"]["context_budget"]["request_window_tokens"]
    content = (StoragePath(model["uri"]) / "config.json").read_bytes()
    with tempfile.TemporaryDirectory(prefix="long-loop-selected-context-") as temporary:
        root = Path(temporary)
        (root / "config.json").write_bytes(content)
        overrides_path = root / "overrides.json"
        overrides_path.write_text(json.dumps(overrides))
        receipt_path = root / "context.json"
        command = skyrl_cpu_command(
            "python",
            "-m",
            "experiments.post_training.long_loop.selected_context_probe",
            str(root),
            str(overrides_path),
            str(requested),
            str(receipt_path),
        )
        # Isolate GPU-wheel dependencies from the ordinary CPU launcher environment.
        command[command.index("--extra") + 1] = "vllm"
        result = subprocess.run(command, env=_vllm_cpu_environment(), capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise ValueError(f"Selected checkpoint context failed before judge allocation:\n{result.stderr}")
        receipt = json.loads(receipt_path.read_text())
    if receipt["config_sha256"] != hashlib.sha256(content).hexdigest():
        raise ValueError("Selected checkpoint context receipt refers to different config bytes")
    if (
        receipt["requested_window"] != requested
        or receipt["overrides"] != overrides
        or receipt["vllm_version"] != serving_release().version
        or not receipt["accepted"]
        or receipt["max_model_len"] < requested
    ):
        raise ValueError("Selected checkpoint context receipt does not match the launch")
    receipt["model_uri"] = model["uri"]
    receipt["model_identity"] = model["identity"]
    return receipt
