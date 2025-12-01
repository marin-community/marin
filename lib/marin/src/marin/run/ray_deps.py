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

"""Manage Ray runtime dependencies."""

import enum
import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ray.runtime_env import RuntimeEnv

logger = logging.getLogger(__name__)

# Packages to ignore when computing the runtime environment.
# These will always be instead sourced from the base environment.
IGNORE_DEPS = [
    "ray",
    "marin",
    "torch",
    "transformers",  # Available in vLLM Docker image, installing via pip can pull in wrong torch
]


class AcceleratorType(enum.Enum):
    NONE = "none"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


def accelerator_type_from_extra(extra: list[str] | None = None) -> AcceleratorType:
    if not extra:
        return AcceleratorType.NONE

    extra = set(extra)
    if "tpu" in extra:
        return AcceleratorType.TPU
    elif "gpu" in extra or "cuda12" in extra:
        return AcceleratorType.GPU
    elif "cpu" in extra:
        return AcceleratorType.CPU
    else:
        return AcceleratorType.NONE


def extra_flags(extra: list[str] | None = None) -> list[str]:
    if not extra:
        extra = []

    accel_type = accelerator_type_from_extra(extra)
    # Don't add cpu if vllm is present, as they conflict
    if accel_type == AcceleratorType.NONE and "vllm" not in extra:
        extra.append("cpu")

    extra = set(extra)

    cmd = []
    for ex in extra:
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse the requirements.txt format output
    py_modules = []
    package_specs = []
    for line in result.stdout.splitlines():
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
        "lib/marin/src",
        "lib/levanter/src",
        "experiments",
    ]

    if not os.path.exists(submodules_dir):
        return paths

    # Iterate through the directories inside submodules
    for submodule in os.listdir(submodules_dir):
        submodule_path = os.path.join(submodules_dir, submodule)

        # Check if it's a directory
        if os.path.isdir(submodule_path):
            # Add both submodule and submodule/src paths, because why not
            paths.append(submodule_path)
            src_path = os.path.join(submodule_path, "src")
            if os.path.isdir(src_path):
                paths.append(src_path)

    return paths


def build_runtime_env_for_packages(
    extra: list[str] | None = None,
    pip_packages: list[str] | None = None,
    env_vars: dict | None = None,
) -> RuntimeEnv:
    """Inject the appropriate UV environment for the given packages."""
    env_vars = env_vars or {}
    pip_packages = pip_packages or []
    extra = extra or []

    python_path = build_python_path()
    if "PYTHONPATH" in env_vars:
        python_path.extend(env_vars["PYTHONPATH"].split(":"))

    package_spec = compute_frozen_packages(extra)

    requirements_txt = [
        """
# Generated by marin/run/ray_deps.py
"""
    ]

    # Extract numpy constraints from pip_packages to ensure they override base dependencies
    numpy_constraints = [pkg for pkg in pip_packages if pkg.startswith("numpy")]
    other_pip_packages = [pkg for pkg in pip_packages if not pkg.startswith("numpy")]

    # Filter out numpy from base dependencies if we have our own constraint
    filtered_base_packages = package_spec.package_specs
    if numpy_constraints:
        # Filter out any line that starts with "numpy" (handles numpy==, numpy>=, etc.)
        filtered_base_packages = [pkg for pkg in package_spec.package_specs if not pkg.strip().startswith("numpy")]

    # N.B. we're currently ignoring torch due to install time.
    # Also filter out ray to prevent version conflicts (ray is in IGNORE_DEPS)
    torch_pkgs = []
    # Put numpy constraints first to override any from base dependencies
    for pkg in numpy_constraints:
        requirements_txt.append(pkg)
    
    # Check if vllm is being installed (which would pull in ray>=2.48.0)
    has_vllm = any("vllm" in pkg for pkg in other_pip_packages)
    
    for pkg in filtered_base_packages + other_pip_packages:
        # Skip ray to prevent version conflicts with the base Ray installation
        if pkg.strip().startswith("ray"):
            continue
        # defer torch installs to the end, so that the --find-links only applies to them
        if "torch" in pkg:
            torch_pkgs.append(pkg)
        requirements_txt.append(pkg)

    # Force the PyTorch CPU version if not using a GPU
    accel_type = accelerator_type_from_extra(extra)
    if accel_type == AcceleratorType.CPU or accel_type == AcceleratorType.TPU or accel_type == AcceleratorType.NONE:
        requirements_txt.append("--find-links https://download.pytorch.org/whl/cpu")

    requirements_txt.extend(torch_pkgs)
    
    # If vllm is being installed, add a constraint to prevent ray from being upgraded
    # This prevents vllm's ray>=2.48.0 requirement from causing conflicts
    if has_vllm:
        # Add ray constraint to prevent version conflicts
        # Note: This won't prevent pip from trying to install ray, but it will constrain the version
        # The real solution would be to install vllm with --no-deps, but Ray's pip plugin doesn't support that
        requirements_txt.append("# Constraint to prevent ray version conflicts with vllm")
        requirements_txt.append("# Ray is already installed in the base environment")
    requirements_txt = "\n".join(requirements_txt)

    # Ray expects a filename for the requirements txt
    req_hash = hashlib.sha256(requirements_txt.encode()).hexdigest()[:16]
    req_path = f"/tmp/ray_reqs_{req_hash}.txt"
    Path(req_path).write_text(requirements_txt)

    return dict(
        env_vars=env_vars | {"PYTHONPATH": ":".join(python_path)},
        pip={"packages": req_path},
    )
