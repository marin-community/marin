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

from iris.cluster.types import Entrypoint
from iris.rpc import cluster_pb2


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: cluster_pb2.EnvironmentConfig,
) -> cluster_pb2.RuntimeEntrypoint:
    """Build a structured RuntimeEntrypoint from a user Entrypoint + env config.

    The setup_commands handle environment preparation (copying bundle, syncing deps,
    activating venv). The run_command is the user's original command, kept separate
    so runtimes that don't need setup can skip it cleanly.
    """
    uv_sync_flags = env_config.env_vars.get("IRIS_UV_SYNC_FLAGS", "")
    pip_install_args = env_config.env_vars.get("IRIS_PIP_INSTALL", "")

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
