# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
    if "tpu" in extra_set:
        return AcceleratorType.TPU
    elif "gpu" in extra_set or "vllm" in extra_set:
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
        "--no-default-groups",
        "--no-annotate",
        "--no-hashes",
        "--prune=ray",
    ]

    # In monorepo, explicitly export the marin package with accelerator extras.
    # In external projects, export the current project without extras - the accelerator
    # variant is already baked into the marin dependency (e.g., marin[cpu]).
    if is_monorepo():
        cmd.insert(2, "marin")
        cmd.insert(2, "--package")
        cmd.extend(extra_flags(extra))

    result = subprocess.check_output(cmd, text=True)

    # Parse the requirements.txt format output
    py_modules = []
    package_specs = []
    for line in result.splitlines():
        line = line.strip()
        # Skip comments, empty lines, local paths, and editable installs
        # Local paths (./foo, ../foo, -e ./foo) don't work on remote clusters
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e"):
            # Editable installs won't work remotely - track as py_module for local use
            py_modules.append(line[3:].strip())
            continue
        if line.startswith("."):
            # Local path dependencies (./foo, ../foo) - skip for remote
            continue
        ignored = [line.startswith(f"{dep}==") or line.startswith(f"{dep}[") for dep in IGNORE_DEPS]
        if any(ignored):
            continue
        package_specs.append(line)

    return PackageSpec(package_specs=package_specs, py_modules=py_modules)


def is_monorepo() -> bool:
    """Detect if running from within the marin monorepo.

    The monorepo has a characteristic structure with lib/marin/src and experiments
    directories at the root. External projects using marin as a dependency will
    not have this structure.
    """
    return os.path.isdir("lib/marin/src") and os.path.isdir("experiments")


def build_python_path(submodules_dir: str = "submodules") -> list[str]:
    """Build the PYTHONPATH for Ray jobs.

    Ray's installation process is... non-optimal. `py_modules` just injects
    the exact "py_module" itself into the PYTHONPATH, but not e.g. the src dir.

    You would think you'd be able to resolve that yourself but cleverly doing
    something like -e {module_path} but noooo, that would be too easy. For whatever
    reason, at install time, the py_modules are not yet in a usable state. So instead
    we have to just manually guess what our PYTHONPATH should be.

    For monorepo: includes experiments/, lib/*/src paths and any submodules
    For external projects: includes only the working directory (dependencies
    come from the installed marin package)
    """
    if not is_monorepo():
        return ["."]

    # Workspace member src directories + experiments directory
    paths = [
        "experiments",
        "lib/harbor/src",
        "lib/fray/src",
        "lib/haliax/src",
        "lib/iris/src",
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
) -> RuntimeEnv:
    """Inject the appropriate UV environment for the given packages."""
    env_vars = dict(env_vars or {})
    pip_packages = pip_packages or []
    extra = extra or []

    python_path = build_python_path()
    if "PYTHONPATH" in env_vars:
        python_path.extend(env_vars["PYTHONPATH"].split(":"))

    package_spec = compute_frozen_packages(extra)

    requirements_txt = [
        """
# Generated by fray/cluster/ray/deps.py
"""
    ]
    # Add resiliparse custom index
    requirements_txt.append("--extra-index-url https://marin-community.github.io/chatnoir-resiliparse/simple")

    torch_pkgs = []
    for pkg in package_spec.package_specs + pip_packages:
        # Defer torch-family installs to the end so that the --extra-index-url only applies to them
        if pkg.startswith(TORCH_GPU_PACKAGE_PREFIXES):
            torch_pkgs.append(pkg)
            continue
        requirements_txt.append(pkg)

    # Force the correct PyTorch wheel index for the requested accelerator type
    accel_type = accelerator_type_from_extra(extra)
    if accel_type in {AcceleratorType.CPU, AcceleratorType.TPU, AcceleratorType.NONE}:
        requirements_txt.append("--extra-index-url https://download.pytorch.org/whl/cpu")
    elif accel_type == AcceleratorType.GPU:
        requirements_txt.append("--extra-index-url https://download.pytorch.org/whl/cu128")

    requirements_txt.extend(torch_pkgs)
    requirements_txt_str = "\n".join(requirements_txt)

    # Ray expects a filename for the requirements txt
    req_hash = hashlib.sha256(requirements_txt_str.encode()).hexdigest()[:16]
    req_path = f"/tmp/ray_reqs_{req_hash}.txt"
    Path(req_path).write_text(requirements_txt_str)

    return dict(  # type: ignore[return-value]
        env_vars=env_vars | {"PYTHONPATH": ":".join(python_path)},
        pip={"packages": req_path},
        pip_check=False,
    )
