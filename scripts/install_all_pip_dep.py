"""
This script reads pyproject.toml and installs all dependencies in the [project.dependencies] section and
all dependencies in [project.optional-dependencies.*] sections.
This is majorly needed for unit testing
"""

import logging
import subprocess
import tempfile
from typing import List

import toml

logger = logging.getLogger("ray")  # Initialize logger


def get_all_pip_dependencies(project_file: str = "pyproject.toml"):
    dependencies = []
    try:
        with open(project_file, "r") as f:
            pyproject = toml.load(f)
            dependencies.extend(pyproject.get("project", {}).get("dependencies", []))
            optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})
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
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-r",
                tempf.name,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "--index-strategy",
                "unsafe-best-match",
                "--prerelease=allow",
            ]
        )


if __name__ == "__main__":
    pip_dep = get_all_pip_dependencies()
    install_all_pip_dependencies(pip_dep)
