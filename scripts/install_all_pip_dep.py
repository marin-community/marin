"""
This script reads pyproject.toml and installs all dependencies in the [project.dependencies] section and
all dependencies in [project.optional-dependencies.*] sections.
This is majorly needed for unit testing
"""

import logging
import subprocess
import tempfile
import os
from typing import List

import toml

logger = logging.getLogger("ray")  # Initialize logger


def get_pip_dependencies(project_file: str = "pyproject.toml", optional_deps: list[str] | None = None):
    dependencies = []
    try:
        with open(project_file, "r") as f:
            pyproject = toml.load(f)
            dependencies.extend(pyproject.get("project", {}).get("dependencies", []))

            optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})

            if optional_deps:
                for dep in optional_deps:
                    dependencies.extend(optional_dependencies[dep])
            else:  # Add all the optional dependencies if nothing explicitly listed
                for dep in optional_dependencies:
                    dependencies.extend(optional_dependencies[dep])
    except FileNotFoundError:
        logger.error(f"File {project_file} not found.")
    except toml.TomlDecodeError:
        logger.error(f"Failed to parse {project_file}.")
    return dependencies


def install_all_pip_dependencies(pip_dep: List[str]):
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
