# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the per-task setup script that prepares a worker's environment.

The setup script runs once on the worker before the user's command, with the
task's ``IRIS_*`` environment available (see ``build_common_iris_env``). It is a
plain bash string so callers can inspect, extend, or replace it.

``default_setup_script`` renders the standard ``uv sync`` flow. A caller that
wants a different environment — a pre-baked image, a scoped sync, or no setup at
all — sets ``setup_script`` on the environment config and bypasses the default
entirely (``EnvironmentConfig.setup_mode == SETUP_MODE_CUSTOM``).

The script creates and populates the venv at ``$IRIS_VENV``; it does not activate
it. The run phase activates ``$IRIS_VENV`` if it exists, so a custom or empty
script that leaves no venv simply runs in the image's own environment.
"""

import shlex
from collections.abc import Sequence

from iris.rpc import job_pb2

# Always installed so callable entrypoints (cloudpickle) and the profiler attach
# paths (py-spy/memray) work without the user declaring them.
_ALWAYS_PIP = ("cloudpickle", "py-spy", "memray")


def _uv_sync_target(packages: Sequence[str] | None) -> str:
    """Return the uv-sync package selector: every member, or a scoped subset."""
    if not packages:
        return "--all-packages"
    return " ".join(f"--package {shlex.quote(p)}" for p in packages)


def _extra_flags(extras: Sequence[str]) -> str:
    """Render ``--extra`` flags. Accepts ``extra`` or ``package:extra`` syntax.

    The package prefix is dropped; ``--extra`` applies to whichever member
    defines that extra name.
    """
    flags: list[str] = []
    for e in extras:
        extra = e.split(":", 1)[1] if ":" in e else e
        flags.extend(["--extra", shlex.quote(extra)])
    return " ".join(flags)


def default_setup_script(
    *,
    extras: Sequence[str] = (),
    pip_packages: Sequence[str] = (),
    python_version: str | None = None,
    packages: Sequence[str] | None = None,
    quiet: bool = True,
) -> str:
    """Render the standard uv-based setup script as a bash string.

    Args:
        extras: uv extras to enable (``extra`` or ``package:extra``).
        pip_packages: extra packages to ``uv pip install`` after the sync.
        python_version: pin the interpreter (matches the client for cloudpickle
            compatibility); omitted when empty.
        packages: workspace members to sync. ``None`` syncs every member
            (``--all-packages``); a list scopes the sync to those members so an
            unrelated member that fails to resolve cannot fail the job.
        quiet: suppress uv output.

    Returns:
        A bash snippet that creates and populates the venv at ``$IRIS_VENV``.
    """
    quiet_flag = "--quiet" if quiet else ""
    python_flag = f"--python {shlex.quote(python_version)}" if python_version else ""
    # --frozen when a lockfile is present skips resolution; ConfigMap-based
    # workdirs may drop uv.lock (>1MB limit), so fall back to a normal resolve.
    frozen_flag = "$([ -f uv.lock ] && echo '--frozen' || echo '')"
    # Symlink wheels from the uv cache into the venv instead of copying; works
    # across bind mounts.
    link_mode_flag = "--link-mode symlink"
    target = _uv_sync_target(packages)
    extra_flags = _extra_flags(extras)
    pip_args = " ".join(shlex.quote(p) for p in (*_ALWAYS_PIP, *pip_packages))

    sync_cmd = " ".join(
        part
        for part in [
            "uv sync",
            quiet_flag,
            frozen_flag,
            link_mode_flag,
            python_flag,
            target,
            "--no-group dev",
            extra_flags,
        ]
        if part
    )
    pip_cmd = " ".join(part for part in ["uv pip install", quiet_flag, link_mode_flag, pip_args] if part)

    lines = [
        'cd "$IRIS_WORKDIR"',
        "echo 'syncing deps'",
        sync_cmd,
        # uv sync writes .pth links for editable path sources but does not invoke
        # the build backend, so rust-dev mode (editable = true) leaves native
        # extensions unbuilt. Build every maturin member explicitly.
        "if grep -q 'editable = true' pyproject.toml 2>/dev/null; then"
        " echo 'rust-dev mode: building native extensions';"
        " for crate in lib/*/pyproject.toml; do"
        ' grep -q \'build-backend = "maturin"\' "$crate" 2>/dev/null &&'
        f' uv pip install {quiet_flag} -e "$(dirname "$crate")";'
        " done;"
        " fi",
        "echo 'installing pip deps'",
        pip_cmd,
    ]
    return "\n".join(lines) + "\n"


def resolve_setup_script(env: job_pb2.EnvironmentConfig) -> str:
    """Resolve the setup script for an environment config.

    A ``SETUP_MODE_CUSTOM`` config uses ``setup_script`` verbatim (including the
    empty string, which means a no-setup bring-your-own-environment job).
    Otherwise the default uv-based script is built from the declared inputs.
    """
    if env.setup_mode == job_pb2.SETUP_MODE_CUSTOM:
        return env.setup_script
    return default_setup_script(
        extras=list(env.extras),
        pip_packages=list(env.pip_packages),
        python_version=env.python_version or None,
        packages=list(env.sync_packages) or None,
        quiet=not env.env_vars.get("IRIS_DEBUG_UV_SYNC"),
    )
