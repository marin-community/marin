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

"""
This script reads pyproject.toml and installs all dependencies in the [project.dependencies] section and
all dependencies in [project.optional-dependencies.*] sections.
"""

import logging
import subprocess
import tempfile
import os
from typing import Sequence

import toml

logger = logging.getLogger("ray")  # Initialize logger


# This is for CI, and we don't usually have these groups installed.
DEFAULT_SKIP_GROUPS = tuple(["cuda12", "tpu"])


def get_pip_dependencies(
    project_file: str = "pyproject.toml",
    optional_deps: list[str] | None = None,
    skip_groups: Sequence[str] = DEFAULT_SKIP_GROUPS,
):
    dependencies = []
    try:
        with open(project_file, "r") as f:
            pyproject = toml.load(f)
            dependencies.extend(pyproject.get("project", {}).get("dependencies", []))

            optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})

            if optional_deps:
                for dep in optional_deps:
                    if dep not in skip_groups:
                        dependencies.extend(optional_dependencies[dep])
            else:  # Add all the optional dependencies if nothing explicitly listed
                for dep in optional_dependencies:
                    if dep not in skip_groups:
                        dependencies.extend(optional_dependencies[dep])
    except FileNotFoundError:
        logger.error(f"File {project_file} not found.")
    except toml.TomlDecodeError:
        logger.error(f"Failed to parse {project_file}.")
    return dependencies


def install_all_pip_dependencies(pip_dep: list[str]):
    with tempfile.NamedTemporaryFile() as tempf:
        for dep in pip_dep:
            tempf.write(f"{dep}\n".encode())
        tempf.seek(0)
        subprocess.run(["pip", "install", "uv"])

        uv_install_command = [
            "uv",
            "pip",
            "install",
        ]

        if os.getenv("TPU_CI"):
            uv_install_command.append("--system")

        uv_install_command.extend(
            [
                "-r",
                tempf.name,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "-f",
                "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                "--index-strategy",
                "unsafe-best-match",
                "--prerelease=allow",
            ]
        )

        subprocess.run(uv_install_command)


if __name__ == "__main__":
    if os.getenv("TPU_CI"):
        pip_dep = get_pip_dependencies(optional_deps=["dev"])
    else:
        pip_dep = get_pip_dependencies()
    install_all_pip_dependencies(pip_dep)
