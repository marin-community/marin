# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for GrugMoE real-checkpoint validation scripts."""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
import posixpath
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import fsspec
import jax

EUROPE_WEST4_GCS_PREFIX = "gs://marin-eu-west4/"
DEFAULT_REGION = "europe-west4"
DEFAULT_TPU_TYPE = "v6e-4"
DEFAULT_RAM = "64g"
DEFAULT_DISK = "96g"
MARIN_GIT_SHA_ENV = "MARIN_GIT_SHA"


def join_path(base: str, *parts: str) -> str:
    parsed = urlparse(base)
    if parsed.scheme in {"", "file"}:
        return os.path.join(base, *parts)
    return posixpath.join(base.rstrip("/"), *parts)


def require_local_or_europe_west4(label: str, path: str) -> None:
    parsed = urlparse(path)
    if parsed.scheme == "gs":
        if path.startswith(EUROPE_WEST4_GCS_PREFIX):
            return
        raise ValueError(f"{label} must be under {EUROPE_WEST4_GCS_PREFIX}, got {path!r}")
    if parsed.scheme in {"", "file"}:
        return
    raise ValueError(f"{label} must be a local path or {EUROPE_WEST4_GCS_PREFIX} path, got {path!r}")


def fs_path(path: str):
    return fsspec.core.url_to_fs(path)


def exists(path: str) -> bool:
    fs, plain_path = fs_path(path)
    return fs.exists(plain_path)


def remove_tree(path: str) -> None:
    fs, plain_path = fs_path(path)
    if fs.exists(plain_path):
        fs.rm(plain_path, recursive=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    parent = path.rsplit("/", 1)[0]
    fs, plain_parent = fs_path(parent)
    fs.makedirs(plain_parent, exist_ok=True)
    with fsspec.open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> dict[str, Any]:
    require_local_or_europe_west4("json_path", path)
    with fsspec.open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def require_file(label: str, path: str) -> None:
    require_local_or_europe_west4(label, path)
    if not exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")


def default_output_dir(output_root: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return join_path(output_root, stamp)


def git_sha() -> str:
    env_sha = os.environ.get(MARIN_GIT_SHA_ENV)
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        return f"unavailable:{exc!r}"


def _direct_url(package: str) -> str:
    try:
        direct_url = md.distribution(package).read_text("direct_url.json")
    except md.PackageNotFoundError:
        return "not-installed"
    return direct_url.strip() if direct_url else ""


def _version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not-installed"


def runtime_snapshot(*, include_jax_devices: bool, include_grugmoe_spec: bool = False) -> dict[str, Any]:
    packages = {}
    for package in ("marin-core", "vllm", "tpu-inference", "jax", "libtpu"):
        packages[package] = {"version": _version(package), "direct_url": _direct_url(package)}
    snapshot: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "marin_sha": git_sha(),
        "packages": packages,
    }
    if include_grugmoe_spec:
        try:
            snapshot["grugmoe_spec"] = repr(importlib.util.find_spec("tpu_inference.models.jax.grugmoe"))
        except ModuleNotFoundError as exc:
            snapshot["grugmoe_spec"] = f"unavailable:{exc!r}"
    if include_jax_devices:
        snapshot.update(
            {
                "jax_process_index": jax.process_index(),
                "jax_process_count": jax.process_count(),
                "jax_local_device_count": jax.local_device_count(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    return snapshot


def submit_tpu_job(
    *,
    config: Any,
    entrypoint: Callable[[Any], Any],
    dependency_groups: Sequence[str],
    base_env_vars: dict[str, str],
    tpu_type: str,
    region: str,
    ram: str,
    disk: str,
    job_name: str,
    summary_label: str,
    summary_fields: dict[str, Any],
) -> None:
    # Fray is only needed for remote submission; local validation paths should not import it.
    from fray import current_client  # noqa: PLC0415
    from fray.cluster import ResourceConfig  # noqa: PLC0415
    from fray.types import Entrypoint, JobRequest, create_environment  # noqa: PLC0415
    from marin.training.run_environment import env_vars_for_dependency_groups  # noqa: PLC0415

    if region != DEFAULT_REGION:
        raise ValueError(f"This smoke is pinned to {DEFAULT_REGION}; got {region!r}")
    resources = ResourceConfig.with_tpu(tpu_type, regions=[region], ram=ram, disk=disk)
    env_vars = env_vars_for_dependency_groups(resources, list(dependency_groups), base_env_vars)
    request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(entrypoint, args=(config,)),
        resources=resources,
        environment=create_environment(
            extras=list(dependency_groups),
            env_vars=env_vars,
        ),
        max_retries_failure=0,
    )
    print(
        summary_label
        + "="
        + json.dumps(
            {
                "job_name": job_name,
                "tpu_type": tpu_type,
                "region": region,
                "ram": ram,
                "disk": disk,
                **summary_fields,
                "marin_sha": env_vars.get(MARIN_GIT_SHA_ENV),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    job = current_client().submit(request)
    print("submitted_job_id=" + str(job.job_id), flush=True)
    job.wait(raise_on_failure=True)
