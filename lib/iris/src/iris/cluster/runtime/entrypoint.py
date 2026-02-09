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

"""Convert user-facing Entrypoint + EnvironmentConfig into a structured RuntimeEntrypoint.

The RuntimeEntrypoint separates setup commands (uv sync, pip install, venv activation)
from the user's actual command. This lets each runtime handle them appropriately:
- DockerRuntime generates a bash script from the structured fields
- ProcessRuntime skips setup commands since the host env is already configured
"""

import shlex
from collections.abc import Sequence

from iris.cluster.types import Entrypoint
from iris.rpc import cluster_pb2


def _build_uv_sync_flags(extras: Sequence[str]) -> str:
    """Build uv sync flags from extras list.

    Accepts 'extra' or 'package:extra' syntax. The package prefix is stripped
    since --all-packages syncs every workspace member and --extra applies
    to whichever package defines that extra name.
    """
    sync_parts = ["--all-packages", "--no-group", "dev"]
    for e in extras:
        if ":" in e:
            _package, extra = e.split(":", 1)
            sync_parts.append(f"--extra {extra}")
        else:
            sync_parts.append(f"--extra {e}")
    return " ".join(sync_parts)


def _build_pip_install_args(pip_packages: Sequence[str]) -> str:
    """Build pip install args. Each package is quoted for shell safety (e.g. torch>=2.0)."""
    packages = ["cloudpickle", *list(pip_packages)]
    return " ".join(f'"{pkg}"' for pkg in packages)


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: cluster_pb2.EnvironmentConfig,
) -> cluster_pb2.RuntimeEntrypoint:
    """Build a structured RuntimeEntrypoint from a user Entrypoint + env config.

    The setup_commands handle environment preparation (copying bundle, syncing deps,
    activating venv). The run_command is the user's original command, kept separate
    so runtimes that don't need setup can skip it cleanly.
    """
    uv_sync_flags = _build_uv_sync_flags(list(env_config.extras))
    pip_install_args = _build_pip_install_args(list(env_config.pip_packages))

    # Use the client's Python version to ensure pickle compatibility.
    # cloudpickle can fail when deserializing functions pickled in a different
    # Python version (e.g., 3.11 -> 3.12 causes "TypeError: bad argument type
    # for built-in operation").
    python_version = env_config.python_version
    python_flag = f"--python {python_version}" if python_version else ""

    setup_commands = [
        "cd /app",
    ]
    # Use --link-mode copy to avoid hardlink warnings when cache and workdir
    # are on different filesystems (common with Docker bind mounts).
    link_mode_flag = "--link-mode copy"
    if uv_sync_flags:
        setup_commands.append(f"uv sync {link_mode_flag} {python_flag} {uv_sync_flags}".strip())
    else:
        setup_commands.append(f"uv sync {link_mode_flag} {python_flag}".strip())
    if pip_install_args:
        setup_commands.append(f"uv pip install {pip_install_args}")
    setup_commands.append("source .venv/bin/activate")
    setup_commands.append('echo "python=$(which python)"')
    setup_commands.append("python -c \"import sys; print('sys.path:', sys.path)\"")

    rt = cluster_pb2.RuntimeEntrypoint()
    rt.setup_commands[:] = setup_commands
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    return rt


def runtime_entrypoint_to_bash_script(rt: cluster_pb2.RuntimeEntrypoint) -> str:
    """Generate a bash setup script from a RuntimeEntrypoint.

    Used by DockerRuntime to produce the _setup_env.sh that runs setup commands
    then execs the user's command.
    """
    quoted_cmd = " ".join(shlex.quote(arg) for arg in rt.run_command.argv)
    lines = ["#!/bin/bash", "set -e"]
    lines.extend(rt.setup_commands)
    lines.append(f"exec {quoted_cmd}")
    return "\n".join(lines) + "\n"
