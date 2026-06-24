# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the per-task setup scripts that prepare a worker's environment.

These are client-side helpers: the submitter resolves the scripts and ships them
to the worker, which runs them in order before the command (the worker never
builds or interprets them). The default environment is two distinct scripts so
iris's own requirements stay separate from the user's project setup:

- ``default_setup_script`` syncs the user's workspace (``uv sync`` + extras + pip).
- ``iris_runtime_setup_script`` installs iris's own runtime deps (cloudpickle for
  callable entrypoints, py-spy/memray for the profiler) into the same venv.

A caller that wants a different environment passes its own scripts and bypasses
both — a custom user script is never mixed with iris's runtime deps. An empty list
means no setup at all (bring-your-own image).

The scripts run with the task's ``IRIS_*`` environment available (see
``build_common_iris_env``) and populate the venv at ``$IRIS_VENV`` without
activating it. The run phase activates ``$IRIS_VENV`` if it exists, so a custom or
empty setup that leaves no venv simply runs in the image's own environment.
"""

import shlex
from collections.abc import Sequence

# Iris's own runtime deps: cloudpickle for callable entrypoints, py-spy/memray for
# the profiler attach paths. Installed separately from the user's project so they
# are never conflated with user-declared dependencies.
_IRIS_RUNTIME_DEPS = ("cloudpickle", "py-spy", "memray")


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
    ]
    if pip_packages:
        pip_args = " ".join(shlex.quote(p) for p in pip_packages)
        pip_cmd = " ".join(part for part in ["uv pip install", quiet_flag, link_mode_flag, pip_args] if part)
        lines += ["echo 'installing pip deps'", pip_cmd]
    return "\n".join(lines) + "\n"


def iris_runtime_setup_script(*, quiet: bool = True) -> str:
    """Render the script that installs iris's own runtime deps into ``$IRIS_VENV``.

    Installs cloudpickle (callable entrypoints) and py-spy/memray (the profiler)
    so iris features work without the user declaring them. This is best-effort and
    never fails the job: it is skipped unless a venv exists (so a bring-your-own
    image is left untouched rather than getting a stray shadowing venv) and a
    failed install only warns. A task running in a non-Python image is unaffected.
    """
    quiet_flag = "--quiet" if quiet else ""
    pkgs = " ".join(shlex.quote(p) for p in _IRIS_RUNTIME_DEPS)
    pip_cmd = " ".join(part for part in ["uv pip install", quiet_flag, "--link-mode symlink", pkgs] if part)
    return (
        'cd "$IRIS_WORKDIR" 2>/dev/null || true\n'
        'if [ -d "$IRIS_VENV" ]; then\n'
        "  echo 'installing iris runtime deps'\n"
        f"  {pip_cmd} || echo '[iris setup] runtime deps install failed; continuing'\n"
        "fi\n"
    )
