# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect JAX, vLLM, and TPU runtime details inside an Iris task."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ENV_KEYS = (
    "TPU_ACCELERATOR_TYPE",
    "TPU_TYPE",
    "TPU_NAME",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_MESH_CONTROLLER_ADDRESS",
    "TPU_VISIBLE_DEVICES",
    "LIBTPU_INIT_ARGS",
    "MODEL_IMPL_TYPE",
    "JAX_PLATFORMS",
    "PJRT_DEVICE",
)

DIST_NAMES = (
    "jax",
    "jaxlib",
    "vllm-tpu",
    "torch",
    "torchvision",
    "libtpu",
    "libtpu-nightly",
)

DEVICE_ATTRS = (
    "id",
    "platform",
    "device_kind",
    "process_index",
    "slice_index",
    "host_id",
    "coords",
    "core_on_chip",
)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return repr(value)


def _distribution_info(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"name": name}
    try:
        info["version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        info["version"] = None

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        capture_output=True,
        check=False,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        info["pip_show"] = proc.stdout.strip()
    else:
        info["pip_show"] = None

    return info


def _module_info(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"name": name}
    try:
        module = importlib.import_module(name)
    except Exception:
        info["import_error"] = traceback.format_exc()
        return info

    info["file"] = getattr(module, "__file__", None)
    info["version"] = getattr(module, "__version__", None)
    return info


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _device_info(device: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": f"{type(device).__module__}.{type(device).__qualname__}",
        "repr": repr(device),
        "dir": sorted(attr for attr in dir(device) if not attr.startswith("__")),
    }
    for attr in DEVICE_ATTRS:
        info[attr] = _safe_value(_safe_getattr(device, attr))
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "label": args.label,
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "env": {key: os.environ.get(key) for key in ENV_KEYS if os.environ.get(key) is not None},
        "modules": {name: _module_info(name) for name in ("jax", "jaxlib", "vllm", "tpu_inference", "torch", "libtpu")},
        "distributions": [_distribution_info(name) for name in DIST_NAMES],
    }

    try:
        import jax
        import jaxlib

        devices = jax.devices()
        payload["jax"] = {
            "version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "default_backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "process_count": jax.process_count(),
            "devices": [_device_info(device) for device in devices],
        }
        if devices:
            client = getattr(devices[0], "client", None)
            payload["jax"]["client"] = {
                "type": None if client is None else f"{type(client).__module__}.{type(client).__qualname__}",
                "repr": _safe_value(client),
                "platform_version": _safe_value(_safe_getattr(client, "platform_version")),
                "runtime_type": _safe_value(_safe_getattr(client, "runtime_type")),
            }
    except Exception:
        payload["jax_error"] = traceback.format_exc()

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
