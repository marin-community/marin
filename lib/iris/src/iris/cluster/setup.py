# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the per-task setup scripts that prepare a worker's environment.

Client-side helpers: the submitter resolves the scripts and ships them to the
worker, which runs them in order before the command. The default environment is
two distinct scripts so iris's requirements stay separate from the user's project:

- ``default_setup_script`` syncs the user's workspace (``uv sync`` + extras + pip).
- ``iris_runtime_setup_script`` installs iris's runtime deps (cloudpickle for
  callable entrypoints, py-spy/memray for the profiler) into the same venv.

A caller can pass its own scripts to bypass both; an empty list means no setup at
all (bring-your-own image).

The scripts run with the task's ``IRIS_*`` environment available and populate the
venv at ``$IRIS_VENV`` without activating it. The run phase activates ``$IRIS_VENV``
if it exists, so a setup that leaves no venv runs in the image's own environment.
"""

import shlex
from collections.abc import Mapping, Sequence

# cloudpickle for callable entrypoints, py-spy/memray for the profiler attach paths.
_IRIS_RUNTIME_DEPS = ("cloudpickle", "py-spy", "memray")

# Set this env var (to any non-empty value) to surface uv's output during setup.
DEBUG_UV_SYNC_ENV = "IRIS_DEBUG_UV_SYNC"


def setup_is_quiet(env_vars: Mapping[str, str]) -> bool:
    """Whether setup scripts should suppress uv output (the default)."""
    return not env_vars.get(DEBUG_UV_SYNC_ENV)


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


def wants_gpu_extra(extras: Sequence[str]) -> bool:
    """Whether any requested extra is the ``gpu`` extra (``extra`` or ``package:extra``)."""
    return any((e.split(":", 1)[1] if ":" in e else e) == "gpu" for e in extras)


def cuda_toolchain_setup_script() -> str:
    """Render a setup script that exposes the venv's CUDA toolchain for JAX/Pallas.

    ``jax[cuda13]`` installs the CUDA compiler wheels into the venv — ptxas/nvlink
    (``nvidia-cuda-nvcc``) under ``nvidia/cu*/bin`` and ``libdevice.10.bc``
    (``nvidia-nvvm``) under ``nvidia/cu*/nvvm/libdevice`` — but nothing exposes
    them, so JAX reports "No PTX compilation provider" and Mosaic GPU lowering
    cannot find ``libdevice.10.bc``.

    Symlinks the toolchain binaries into the venv's ``bin`` (already on ``PATH``
    once the venv is activated) and stages ``libdevice.10.bc`` into XLA's default
    CUDA data dir (``./cuda_sdk_lib``) and the working directory — the locations
    XLA and Mosaic probe — so no run-phase env injection is needed. Keyed on a
    ``ptxas`` under ``nvidia/cu*`` so it spans CUDA major versions and is a no-op
    when no toolchain is installed.
    """
    return r"""set -e
cuda_bin=""
for _d in "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin; do
  if [ -x "$_d/ptxas" ]; then cuda_bin="$_d"; break; fi
done
if [ -z "$cuda_bin" ]; then echo 'no CUDA toolchain to stage'; exit 0; fi
echo 'staging CUDA toolchain'
ln -sf "$cuda_bin"/* "$IRIS_VENV/bin/"
_libdevice="$(dirname "$cuda_bin")/nvvm/libdevice/libdevice.10.bc"
if [ -f "$_libdevice" ]; then
  mkdir -p "$IRIS_WORKDIR/cuda_sdk_lib/nvvm/libdevice"
  cp -f "$_libdevice" "$IRIS_WORKDIR/cuda_sdk_lib/nvvm/libdevice/libdevice.10.bc"
  cp -f "$_libdevice" "$IRIS_WORKDIR/libdevice.10.bc"
fi
"""


def iris_runtime_setup_script(*, quiet: bool = True) -> str:
    """Render the script that installs iris's own runtime deps into ``$IRIS_VENV``.

    Installs cloudpickle (callable entrypoints) and py-spy/memray (the profiler)
    so iris features work without the user declaring them. Best-effort: skipped
    unless a venv exists (a bring-your-own image is left untouched) and a failed
    install only warns, so it never fails the job.
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
