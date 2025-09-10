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

import logging
import os
import subprocess
from dataclasses import dataclass

from ray.runtime_env import RuntimeEnv

logger = logging.getLogger(__name__)


def extra_flags(extra: list[str] | None = None) -> list[str]:
    # always include either the cpu or tpu extra
    if "tpu" in extra:
        extra = list(set(extra) - {"cpu"})
    else:
        extra = extra or []
        if "cpu" not in extra:
            extra.append("cpu")

    extra = extra or []
    extra = list(set(extra))
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
    cmd = ["uv", "export", "--no-annotate", "--no-hashes", "--prune=ray", *extra_flags(extra)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse the requirements.txt format output
    py_modules = []
    package_specs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # Skip comments, empty lines, and editable installs
        if not line or line.startswith("#"):
            continue
        if "ray=" in line or "ray[" in line:
            # ray is provided by the runtime, so we skip it
            continue
        if line.startswith("-e"):
            # convert to a py_module reference
            py_modules.append(line[3:].strip())
            # this sadly doesn't work.
            # package_specs.append(line)
        else:
            # strip +cpu references
            if "+cpu" in line:
                line = line.replace("+cpu", "")
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
    paths = ["src", "experiments"]

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
    return dict(
        env_vars=env_vars | {"PYTHONPATH": ":".join(python_path)},
        # We can't even use this sadly, because Ray doesn't allow specifying it at task time.
        # py_modules=package_spec.py_modules,
        pip={
            "packages": package_spec.package_specs + pip_packages,
        },
    )
