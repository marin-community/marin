# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and compose the environment layers that prepare an Iris task.

The submitter resolves one ordered layer sequence. A layer's setup runs before
the command; its activation is sourced after virtualenv activation. Layer
lifetime controls whether a child environment replacement keeps it.

The default environment is two distinct setup steps so Iris's requirements stay
separate from the user's project:

- ``default_setup_script`` syncs the user's workspace (``uv sync`` + extras + pip).
- ``iris_runtime_setup_script`` installs iris's runtime deps (cloudpickle for
  callable entrypoints, py-spy/memray for the profiler) into the same venv.

A caller can replace the default through ``SetupPlan.custom`` or bypass setup
through ``SetupPlan.empty``.

The scripts run with the task's ``IRIS_*`` environment available and populate the
venv at ``$IRIS_VENV`` without activating it. The run phase activates
``$IRIS_VENV`` if it exists, then sources layer activation. A setup that leaves
no venv runs in the image's own environment.
"""

import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from rigging.telemetry.probes.nccl_client import NCCL_RAS_ENABLE_ENV

from iris.rpc import job_pb2


class EnvironmentLayerLifetime(StrEnum):
    """How an environment layer behaves when a job submits a child."""

    ENVIRONMENT = "environment"
    JOB_TREE = "job_tree"


@dataclass(frozen=True)
class EnvironmentLayer:
    """Setup and activation contributed by one environment layer.

    ``setup`` runs before the task command in the setup phase. ``activate`` is
    sourced after virtualenv activation, so its exports affect the command.
    """

    setup: str = ""
    activate: str = ""
    lifetime: EnvironmentLayerLifetime = EnvironmentLayerLifetime.ENVIRONMENT

    def __post_init__(self) -> None:
        if not self.setup.strip() and not self.activate.strip():
            raise ValueError("An environment layer must provide setup or activation")

    @classmethod
    def environment(cls, *, setup: str = "", activate: str = "") -> "EnvironmentLayer":
        return cls(setup=setup, activate=activate, lifetime=EnvironmentLayerLifetime.ENVIRONMENT)

    @classmethod
    def job_tree(cls, *, setup: str = "", activate: str = "") -> "EnvironmentLayer":
        return cls(setup=setup, activate=activate, lifetime=EnvironmentLayerLifetime.JOB_TREE)

    def to_proto(self) -> job_pb2.EnvironmentLayer:
        lifetime = {
            EnvironmentLayerLifetime.ENVIRONMENT: job_pb2.ENVIRONMENT_LAYER_LIFETIME_ENVIRONMENT,
            EnvironmentLayerLifetime.JOB_TREE: job_pb2.ENVIRONMENT_LAYER_LIFETIME_JOB_TREE,
        }[self.lifetime]
        return job_pb2.EnvironmentLayer(
            setup_script=self.setup,
            activation_script=self.activate,
            lifetime=lifetime,
        )

    @classmethod
    def from_proto(cls, proto: job_pb2.EnvironmentLayer) -> "EnvironmentLayer":
        lifetimes = {
            job_pb2.ENVIRONMENT_LAYER_LIFETIME_ENVIRONMENT: EnvironmentLayerLifetime.ENVIRONMENT,
            job_pb2.ENVIRONMENT_LAYER_LIFETIME_JOB_TREE: EnvironmentLayerLifetime.JOB_TREE,
        }
        try:
            lifetime = lifetimes[proto.lifetime]
        except KeyError as error:
            raise ValueError(f"Unknown environment layer lifetime: {proto.lifetime}") from error
        return cls(setup=proto.setup_script, activate=proto.activation_script, lifetime=lifetime)


def normalized_environment_config(environment: job_pb2.EnvironmentConfig) -> job_pb2.EnvironmentConfig:
    """Return the layer representation accepted by the current controller."""
    normalized = job_pb2.EnvironmentConfig()
    normalized.CopyFrom(environment)
    if not normalized.setup_layers:
        normalized.setup_layers.extend(
            EnvironmentLayer.environment(setup=script).to_proto()
            for script in normalized.setup_scripts
            if script.strip()
        )
    normalized.ClearField("setup_scripts")
    return normalized


class SetupPlanMode(StrEnum):
    """Whether a requested setup plan extends or replaces the environment."""

    EXTEND = "extend"
    REPLACE = "replace"


@dataclass(frozen=True)
class SetupPlan:
    """An environment-layer change requested by a job submission."""

    mode: SetupPlanMode
    layers: tuple[EnvironmentLayer, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is SetupPlanMode.EXTEND and any(
            layer.lifetime is not EnvironmentLayerLifetime.JOB_TREE for layer in self.layers
        ):
            raise ValueError("An extending setup plan may contain only job-tree layers")

    @classmethod
    def default(
        cls,
        *,
        extras: Sequence[str] = (),
        pip_packages: Sequence[str] = (),
        sync_packages: Sequence[str] | None = None,
    ) -> "SetupPlan":
        """Build the standard uv environment, including GPU staging when needed."""
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        resolved_extras = tuple(extras)
        layers = [
            EnvironmentLayer.environment(
                setup=default_setup_script(
                    extras=resolved_extras,
                    pip_packages=pip_packages,
                    python_version=python_version,
                    packages=sync_packages,
                )
            )
        ]
        if wants_gpu_extra(resolved_extras):
            layers.append(
                EnvironmentLayer.environment(
                    setup=cuda_toolchain_setup_script(),
                    activate=gpu_runtime_activation_script(),
                )
            )
        return cls(mode=SetupPlanMode.REPLACE, layers=tuple(layers))

    @classmethod
    def custom(cls, scripts: Sequence[str], *, extras: Sequence[str] = ()) -> "SetupPlan":
        layers = [EnvironmentLayer.environment(setup=script) for script in scripts if script.strip()]
        if wants_gpu_extra(extras):
            layers.append(EnvironmentLayer.environment(activate=gpu_runtime_activation_script()))
        return cls(mode=SetupPlanMode.REPLACE, layers=tuple(layers))

    @classmethod
    def empty(cls) -> "SetupPlan":
        return cls(mode=SetupPlanMode.REPLACE)

    @classmethod
    def extend_job_tree(cls, layers: Sequence[EnvironmentLayer]) -> "SetupPlan":
        return cls(mode=SetupPlanMode.EXTEND, layers=tuple(layers))

    @classmethod
    def resolved(cls, layers: Sequence[EnvironmentLayer]) -> "SetupPlan":
        return cls(mode=SetupPlanMode.REPLACE, layers=tuple(layers))

    def with_layer(self, layer: EnvironmentLayer) -> "SetupPlan":
        return replace(self, layers=(*self.layers, layer))


def resolve_setup_layers(
    requested: SetupPlan | None,
    parent_layers: Sequence[EnvironmentLayer] | None,
) -> list[EnvironmentLayer]:
    """Resolve a job's setup layers against its parent's resolved layers."""
    if parent_layers is None:
        if requested is None:
            return list(SetupPlan.default().layers)
        if requested.mode is SetupPlanMode.EXTEND:
            return [*SetupPlan.default().layers, *requested.layers]
        return list(requested.layers)

    if requested is None:
        return list(parent_layers)
    if requested.mode is SetupPlanMode.EXTEND:
        return [*parent_layers, *requested.layers]

    inherited = [layer for layer in parent_layers if layer.lifetime is EnvironmentLayerLifetime.JOB_TREE]
    environment = [layer for layer in requested.layers if layer.lifetime is EnvironmentLayerLifetime.ENVIRONMENT]
    job_tree = [layer for layer in requested.layers if layer.lifetime is EnvironmentLayerLifetime.JOB_TREE]
    return [*environment, *inherited, *job_tree]


# cloudpickle for callable entrypoints, py-spy/memray for the profiler attach paths.
_IRIS_RUNTIME_DEPS = ("cloudpickle", "py-spy", "memray")


def gpu_runtime_activation_script() -> str:
    """Return shell activation that supplies Iris's default NCCL diagnostics."""
    return f"""\
export {NCCL_RAS_ENABLE_ENV}="${{{NCCL_RAS_ENABLE_ENV}-1}}"
export NCCL_DEBUG="${{NCCL_DEBUG-INFO}}"
export NCCL_DEBUG_SUBSYS="${{NCCL_DEBUG_SUBSYS-INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS}}"
export NCCL_DEBUG_TIMESTAMP="${{NCCL_DEBUG_TIMESTAMP-[%F %T.%3f]}}"
"""


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
) -> str:
    """Render the standard uv-based setup script as a bash string.

    uv runs at its default verbosity so its progress (``Resolved``,
    ``Downloading <pkg>``, ``Installed``) streams into the task logs; this is the
    only signal a live setup gives, so it is never suppressed.

    Args:
        extras: uv extras to enable (``extra`` or ``package:extra``).
        pip_packages: extra packages to ``uv pip install`` after the sync.
        python_version: pin the interpreter (matches the client for cloudpickle
            compatibility); omitted when empty.
        packages: workspace members to sync. ``None`` syncs every member
            (``--all-packages``); a list scopes the sync to those members so an
            unrelated member that fails to resolve cannot fail the job. The
            development dependency group is omitted in both cases.

    Returns:
        A bash snippet that creates and populates the venv at ``$IRIS_VENV``.
    """
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
            frozen_flag,
            link_mode_flag,
            python_flag,
            target,
            "--no-dev",
            extra_flags,
        ]
        if part
    )

    lines = [
        "set -e",
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
        ' uv pip install -e "$(dirname "$crate")";'
        " done;"
        " fi",
    ]
    if pip_packages:
        pip_args = " ".join(shlex.quote(p) for p in pip_packages)
        pip_cmd = " ".join(["uv pip install", link_mode_flag, pip_args])
        lines += ["echo 'installing pip deps'", pip_cmd]
    return "\n".join(lines) + "\n"


def wants_gpu_extra(extras: Sequence[str]) -> bool:
    """Whether any requested extra is the ``gpu`` extra (``extra`` or ``package:extra``)."""
    return any((e.split(":", 1)[1] if ":" in e else e) == "gpu" for e in extras)


# The NVVM bitcode library JAX/XLA load to compile GPU kernels.
_LIBDEVICE_FILE = "libdevice.10.bc"
# XLA's built-in default --xla_gpu_cuda_data_dir, resolved relative to the workdir.
_XLA_CUDA_DATA_DIR = "cuda_sdk_lib"
# These are the only CUDA 12/13 distributions in the resolved GPU environment
# that both install files under the same nvidia namespace.  Reinstalling them
# last makes the requested CUDA 13 wheel own its shared-library paths again.
CUDA_13_LIBRARY_PACKAGES = ("nvidia-cudnn-cu13", "nvidia-nccl-cu13")


def cuda_toolchain_setup_script() -> str:
    """Return a setup script that exposes the venv's CUDA toolchain to JAX/Pallas.

    Appended to a GPU job's setup so Mosaic GPU kernels compile and JAX sees the
    CUDA 13 shared libraries after mixed CUDA package installs. It puts the
    ``jax[cuda13]`` toolchain (``ptxas``/``nvlink``) on ``PATH``, stages
    ``libdevice.10.bc`` where XLA looks, and restores CUDA 13 cuDNN and NCCL
    precedence when those packages are installed. A no-op when the venv carries
    no CUDA toolchain.
    """
    cuda_13_library_packages = " ".join(CUDA_13_LIBRARY_PACKAGES)
    return rf"""set -e
cuda_bin=""
for _d in "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin; do
  if [ -x "$_d/ptxas" ]; then cuda_bin="$_d"; break; fi
done
if [ -z "$cuda_bin" ]; then echo 'no CUDA toolchain to stage'; exit 0; fi
echo 'staging CUDA toolchain'
ln -sf "$cuda_bin"/* "$IRIS_VENV/bin/"
_libdevice="$(dirname "$cuda_bin")/nvvm/libdevice/{_LIBDEVICE_FILE}"
if [ -f "$_libdevice" ]; then
  mkdir -p "$IRIS_WORKDIR/{_XLA_CUDA_DATA_DIR}/nvvm/libdevice"
  cp -f "$_libdevice" "$IRIS_WORKDIR/{_XLA_CUDA_DATA_DIR}/nvvm/libdevice/{_LIBDEVICE_FILE}"
  cp -f "$_libdevice" "$IRIS_WORKDIR/{_LIBDEVICE_FILE}"
fi
for _cuda13_package in {cuda_13_library_packages}; do
  _cuda13_version=""
  if [ -x "$IRIS_VENV/bin/python" ]; then
    _cuda13_version="$(
      "$IRIS_VENV/bin/python" - "$_cuda13_package" <<'PY'
import importlib.metadata as md
import sys

try:
    print(md.version(sys.argv[1]))
except md.PackageNotFoundError:
    pass
PY
    )"
  fi
  if [ -n "$_cuda13_version" ]; then
    echo "restoring CUDA 13 library precedence for $_cuda13_package"
    uv pip install --python "$IRIS_VENV/bin/python" \
      --link-mode symlink \
      --reinstall-package "$_cuda13_package" \
      "$_cuda13_package==$_cuda13_version"
  fi
done
"""


def iris_runtime_setup_script() -> str:
    """Render the script that installs iris's own runtime deps into ``$IRIS_VENV``.

    Installs cloudpickle (callable entrypoints) and py-spy/memray (the profiler)
    so iris features work without the user declaring them. Best-effort: skipped
    unless a venv exists (a bring-your-own image is left untouched) and a failed
    install only warns, so it never fails the job.
    """
    pkgs = " ".join(shlex.quote(p) for p in _IRIS_RUNTIME_DEPS)
    pip_cmd = " ".join(["uv pip install", "--link-mode symlink", pkgs])
    return (
        'cd "$IRIS_WORKDIR" 2>/dev/null || true\n'
        'if [ -d "$IRIS_VENV" ]; then\n'
        "  echo 'installing iris runtime deps'\n"
        f"  {pip_cmd} || echo '[iris setup] runtime deps install failed; continuing'\n"
        "fi\n"
    )
