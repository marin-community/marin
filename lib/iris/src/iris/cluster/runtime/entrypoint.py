# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert user-facing Entrypoint + EnvironmentConfig into a structured RuntimeEntrypoint.

The RuntimeEntrypoint separates setup commands (uv sync, pip install, venv activation)
from the user's actual command. This lets each runtime handle them appropriately:
- DockerRuntime generates a bash script from the structured fields
- ProcessRuntime skips setup commands since the host env is already configured
"""

import shlex
from collections.abc import Sequence

from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


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
        else:
            extra = e
        # Quote the extra name to prevent shell injection when building the sync command.
        sync_parts.extend(["--extra", shlex.quote(extra)])
    return " ".join(sync_parts)


def _build_pip_install_args(pip_packages: Sequence[str]) -> str:
    """Build pip install args. Each package is quoted for shell safety (e.g. torch>=2.0)."""
    packages = ["cloudpickle", "py-spy", "memray", *list(pip_packages)]
    # Use shlex.quote to safely escape each package spec for the shell.
    return " ".join(shlex.quote(pkg) for pkg in packages)


def build_runtime_entrypoint(
    entrypoint: Entrypoint,
    env_config: job_pb2.EnvironmentConfig,
) -> job_pb2.RuntimeEntrypoint:
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

    # Suppress uv output by default; set IRIS_DEBUG_UV_SYNC=1 in env_vars for verbose output.
    quiet_flag = "" if env_config.env_vars.get("IRIS_DEBUG_UV_SYNC") else "--quiet"

    setup_commands = [
        "cd /app",
    ]
    # Use --link-mode symlink to reference cached wheels directly from .venv,
    # avoiding redundant installation. Symlinks work across bind mounts.
    link_mode_flag = "--link-mode symlink"
    setup_commands.append("echo 'syncing deps'")
    # Use --frozen when uv.lock is present to skip resolution. ConfigMap-based
    # workdirs may drop uv.lock (>1MB limit), so fall back to normal resolve.
    frozen_flag = "$([ -f uv.lock ] && echo '--frozen' || echo '')"
    if uv_sync_flags:
        setup_commands.append(
            f"uv sync {quiet_flag} {frozen_flag} {link_mode_flag} {python_flag} {uv_sync_flags}".strip()
        )
    else:
        setup_commands.append(f"uv sync {quiet_flag} {frozen_flag} {link_mode_flag} {python_flag}".strip())
    # In rust-dev mode, uv sync creates .pth links for editable path sources but
    # doesn't invoke the build backend (maturin), so native extensions are missing.
    # Detect the mode via the RUST-DEV markers in pyproject.toml and explicitly
    # build any Rust crates found under rust/.
    setup_commands.append(
        "if grep -q 'path = \"rust/' pyproject.toml 2>/dev/null; then"
        " echo 'rust-dev mode: building native extensions';"
        " for crate in rust/*/pyproject.toml; do"
        f' uv pip install {quiet_flag} -e "$(dirname "$crate")";'
        " done;"
        " fi"
    )
    setup_commands.append("echo 'installing pip deps'")
    if pip_install_args:
        setup_commands.append(f"uv pip install {quiet_flag} {link_mode_flag} {pip_install_args}")
    setup_commands.append("echo 'activating venv'")
    setup_commands.append("source .venv/bin/activate")
    setup_commands.append('echo "python=$(which python)"')
    setup_commands.append("python -c \"import sys; print('sys.path:', sys.path)\"")
    setup_commands.append("echo 'running user command'")

    rt = job_pb2.RuntimeEntrypoint()
    rt.setup_commands[:] = setup_commands
    rt.run_command.argv[:] = entrypoint.command
    for k, v in entrypoint.workdir_files.items():
        rt.workdir_files[k] = v
    for k, v in entrypoint.workdir_file_refs.items():
        rt.workdir_file_refs[k] = v
    return rt


def runtime_entrypoint_to_bash_script(rt: job_pb2.RuntimeEntrypoint) -> str:
    """Generate a bash setup script from a RuntimeEntrypoint.

    Used by DockerRuntime to produce the _setup_env.sh that runs setup commands
    then execs the user's command.
    """
    quoted_cmd = " ".join(shlex.quote(arg) for arg in rt.run_command.argv)
    lines = ["#!/bin/bash", "set -e"]
    lines.extend(rt.setup_commands)
    lines.append(f"exec {quoted_cmd}")
    return "\n".join(lines) + "\n"
