# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute Python package dependencies for Ray jobs."""

import enum
import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ray.runtime_env import RuntimeEnv

logger = logging.getLogger("ray")

# Packages to ignore when computing the runtime environment.
# These will always be instead sourced from the base environment.
IGNORE_DEPS = [
    "ray",
    "marin",
]


TORCH_GPU_PACKAGE_PREFIXES = (
    "torch",
    "torchvision",
    "torchaudio",
)


class AcceleratorType(enum.Enum):
    NONE = "none"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


def accelerator_type_from_extra(extra: list[str] | None = None) -> AcceleratorType:
    if not extra:
        return AcceleratorType.NONE

    extra_set = set(extra)
    # Note: "vllm" extra uses vllm-tpu which runs on TPUs, so it should use
    # the CPU PyTorch index (TPUs use CPU torch, not CUDA torch).
    if "tpu" in extra_set or "vllm" in extra_set:
        return AcceleratorType.TPU
    elif "gpu" in extra_set:
        return AcceleratorType.GPU
    elif "cpu" in extra_set:
        return AcceleratorType.CPU
    else:
        return AcceleratorType.NONE


def extra_flags(extra: list[str] | None = None) -> list[str]:
    if not extra:
        extra = []

    accel_type = accelerator_type_from_extra(extra)
    if accel_type == AcceleratorType.NONE:
        extra.append("cpu")

    extra_set = set(extra)

    cmd = []
    for ex in extra_set:
        if ex.strip():
            cmd.append(f"--extra={ex}")
    return cmd


@dataclass
class PackageSpec:
    package_specs: list[str]
    py_modules: list[str]


def compute_frozen_packages(extra: list[str] | None = None) -> PackageSpec:
    cmd = [
        "uv",
        "export",
        "--package",
        "marin",
        "--no-default-groups",
        "--no-annotate",
        "--no-hashes",
        "--prune=ray",
        *extra_flags(extra),
    ]
    result = subprocess.check_output(cmd, text=True)

    # Parse the requirements.txt format output
    py_modules = []
    package_specs = []
    for line in result.splitlines():
        line = line.strip()
        # Skip comments, empty lines, and editable installs
        if not line or line.startswith("#") or line.startswith("./"):
            continue
        ignored = [line.startswith(f"{dep}==") or line.startswith(f"{dep}[") for dep in IGNORE_DEPS]
        if any(ignored):
            continue
        if line.startswith("-e"):
            # convert to a py_module. this isn't used for now, instead see `build_python_path`
            py_modules.append(line[3:].strip())
        else:
            package_specs.append(line)

    return PackageSpec(package_specs=package_specs, py_modules=py_modules)


def build_python_path(submodules_dir: str = "submodules") -> list[str]:
    """Build the PYTHONPATH for the given submodules.

    Ray's installation process is... non-optimal. `py_modules` just injects
    the exact "py_module" itself into the PYTHONPATH, but not e.g. the src dir.

    You would think you'd be able to resolve that yourself but cleverly doing
    something like -e {module_path} but noooo, that would be too easy. For whatever
    reason, at install time, the py_modules are not yet in a usable state. So instead
    we have to just manually guess what our PYTHONPATH should be.
    """
    # Workspace member src directories + experiments directory
    paths = [
        "experiments",
        "lib/fray/src",
        "lib/haliax/src",
        "lib/levanter/src",
        "lib/marin/src",
        "lib/zephyr/src",
    ]

    if not os.path.exists(submodules_dir):
        return paths

    for submodule in os.listdir(submodules_dir):
        submodule_path = os.path.join(submodules_dir, submodule)
        if os.path.isdir(submodule_path):
            paths.append(submodule_path)
            src_path = os.path.join(submodule_path, "src")
            if os.path.isdir(src_path):
                paths.append(src_path)

    return paths


def build_runtime_env_for_packages(
    extra: list[str] | None = None,
    pip_packages: list[str] | None = None,
    env_vars: dict | None = None,
    cache_buster: str | None = None,
) -> RuntimeEnv:
    """Inject the appropriate UV environment for the given packages.

    Args:
        extra: List of extra package groups to include (e.g., ["eval", "vllm"]).
        pip_packages: Additional pip packages to install.
        env_vars: Environment variables to set.
        cache_buster: Optional string to bust Ray's pip environment cache.
            If provided (or if the RAY_PIP_CACHE_BUSTER environment variable is set),
            this string will be included as a fake package dependency,
            causing Ray to create a new pip environment instead of reusing a cached one.
            Useful when pip environments on some nodes may be stale or corrupted.
    """
    env_vars = dict(env_vars or {})
    pip_packages = pip_packages or []
    extra = extra or []

    # Allow cache buster to be set via environment variable
    if cache_buster is None:
        cache_buster = os.environ.get("RAY_PIP_CACHE_BUSTER")

    python_path = build_python_path()
    if "PYTHONPATH" in env_vars:
        python_path.extend(env_vars["PYTHONPATH"].split(":"))

    package_spec = compute_frozen_packages(extra)

    # Build list of extra index URLs for pip
    # We use PIP_EXTRA_INDEX_URL environment variable instead of --extra-index-url
    # in requirements file. This is because we pass packages as a list (not a file path)
    # to ensure proper serialization when runtime_env is passed through Ray actor chains.
    # File paths don't work because the file only exists on the node where runtime_env
    # was created, not on remote TPU workers.
    extra_index_urls = [
        "https://marin-community.github.io/chatnoir-resiliparse/simple",
    ]

    # Force the correct PyTorch wheel index for the requested accelerator type
    accel_type = accelerator_type_from_extra(extra)
    if accel_type in {AcceleratorType.CPU, AcceleratorType.TPU, AcceleratorType.NONE}:
        extra_index_urls.append("https://download.pytorch.org/whl/cpu")
    elif accel_type == AcceleratorType.GPU:
        extra_index_urls.append("https://download.pytorch.org/whl/cu128")

    # Set PIP_EXTRA_INDEX_URL environment variable (space-separated list of URLs)
    env_vars["PIP_EXTRA_INDEX_URL"] = " ".join(extra_index_urls)

    # Build list of packages (as a list, not a file path)
    # This ensures proper serialization when runtime_env is passed to remote workers
    packages_list: list[str] = []

    # Add cache buster as a fake package comment if provided
    # pip ignores lines starting with # but including it changes the package list hash
    if cache_buster:
        # Use a harmless package spec that pip will parse but is effectively a no-op
        # We add a comment-like marker in a way that affects the hash
        packages_list.append(f"# cache_buster={cache_buster}")

    torch_pkgs = []
    for pkg in package_spec.package_specs + pip_packages:
        # Defer torch-family installs to the end so that the extra index URL applies to them
        if pkg.startswith(TORCH_GPU_PACKAGE_PREFIXES):
            torch_pkgs.append(pkg)
            continue
        packages_list.append(pkg)

    packages_list.extend(torch_pkgs)

    return dict(  # type: ignore[return-value]
        env_vars=env_vars | {"PYTHONPATH": ":".join(python_path)},
        pip={"packages": packages_list, "pip_check": False},
    )
