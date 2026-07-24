# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import shlex
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
_FORWARDED_ENV_PREFIXES = (
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_",
    "LIBTPU_INIT_ARGS",
    "NCCL_",
    "JAX_",
    "MARIN_NGC_XLA_",
)
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)
_NGC_XLA_CUDA_PLUGIN_PATH = "/opt/jaxlibs/jax_cuda13_pjrt/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
_NGC_XLA_STOCK_CUDA_PLUGIN_SHA256 = "d368aae61feb78c14b7549ff034c7f1a637f2222d9421f868e680679234a5265"
_NGC_XLA_LIBRARY_PATHS = (
    "/usr/lib/aarch64-linux-gnu/nvshmem/13",
    "/usr/lib/x86_64-linux-gnu/nvshmem/13",
    "/usr/local/cuda/targets/aarch64-linux/lib",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/compat/lib",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
)
_NVIDIA_JAX_IMAGE_PREFIX = "nvcr.io/nvidia/jax:"
_NVIDIA_JAX_UV_VERSION = "0.11.21"
_CPU_TORCH_WHEEL = "https://download.pytorch.org/whl/cpu/" "torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl"
_NVIDIA_JAX_PROTECTED_PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda13-pjrt",
    "jax-cuda13-plugin",
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-python",
    "cuda-toolkit",
    "nvidia-cublas",
    "nvidia-cublas-cu12",
    "nvidia-cuda-crt",
    "nvidia-cuda-cupti",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-cufft-cu12",
    "nvidia-cufile",
    "nvidia-cufile-cu12",
    "nvidia-curand",
    "nvidia-curand-cu12",
    "nvidia-cusolver",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-cusparselt-cu13",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvshmem-cu12",
    "nvidia-nvshmem-cu13",
    "nvidia-nvtx",
    "nvidia-nvtx-cu12",
    "nvidia-nvvm",
    "torch",
    "torchvision",
)


def nvidia_jax_overlay_setup_script() -> str:
    """Build a Marin overlay without replacing an NGC image's accelerator stack."""
    root_protected_packages = tuple(
        package for package in _NVIDIA_JAX_PROTECTED_PACKAGES if package not in {"torch", "torchvision"}
    )
    root_protected_flags = (" \\\n  ").join(f"--no-install-package {package}" for package in root_protected_packages)
    gpu_protected_flags = (" \\\n  ").join(
        f"--no-install-package {package}" for package in _NVIDIA_JAX_PROTECTED_PACKAGES
    )
    return f"""set -eu
cd "$IRIS_WORKDIR"
test ! -e "$IRIS_VENV"
sha256sum \\
  /opt/jax/jax/__init__.py \\
  /opt/jaxlibs/jaxlib/jaxlib/__init__.py \\
  /opt/jaxlibs/jaxlib/jaxlib/_jax.so \\
  /opt/jaxlibs/jaxlib/jaxlib/libjax_common.so \\
  > /tmp/ngc-jax-before.sha256
/usr/bin/python -m pip install \\
  --disable-pip-version-check --no-deps --prefix /tmp/marin-ngc-uv \\
  uv=={_NVIDIA_JAX_UV_VERSION}
uv=/tmp/marin-ngc-uv/local/bin/uv
test -x "$uv"
"$uv" venv --system-site-packages --python /usr/bin/python "$IRIS_VENV"
export VIRTUAL_ENV="$IRIS_VENV"
export PATH="$IRIS_VENV/bin:$PATH"
"$uv" sync --quiet --active --inexact --frozen --link-mode symlink \\
  --python "$IRIS_VENV/bin/python" --package marin-root \\
  {root_protected_flags}
"$uv" pip install --python "$IRIS_VENV/bin/python" --no-deps "{_CPU_TORCH_WHEEL}"
"$uv" sync --quiet --active --inexact --frozen --link-mode symlink \\
  --python "$IRIS_VENV/bin/python" --package marin-levanter --no-group dev --extra gpu \\
  {gpu_protected_flags}
sha256sum \\
  /opt/jax/jax/__init__.py \\
  /opt/jaxlibs/jaxlib/jaxlib/__init__.py \\
  /opt/jaxlibs/jaxlib/jaxlib/_jax.so \\
  /opt/jaxlibs/jaxlib/jaxlib/libjax_common.so \\
  > /tmp/ngc-jax-after.sha256
diff -u /tmp/ngc-jax-before.sha256 /tmp/ngc-jax-after.sha256
test ! -e "$IRIS_VENV/lib/python3.12/site-packages/jax"
test ! -e "$IRIS_VENV/lib/python3.12/site-packages/jaxlib"
test -d "$IRIS_VENV/lib/python3.12/site-packages/"nvidia_cutlass_dsl_libs_cu13-*.dist-info
test ! -e "$IRIS_VENV/lib/python3.12/site-packages/"nvidia_cutlass_dsl_libs_base-*.dist-info
"$IRIS_VENV/bin/python" - <<'PY'
import os

import cutlass
import jax
import jaxlib
import torch
from cutlass._mlir._mlir_libs import _cutlass_ir

assert jax.__file__.startswith("/opt/jax/"), jax.__file__
assert jaxlib.__file__.startswith("/opt/jaxlibs/"), jaxlib.__file__
venv = os.environ["VIRTUAL_ENV"] + "/"
assert cutlass.__file__.startswith(venv), cutlass.__file__
assert _cutlass_ir.__file__.startswith(venv), _cutlass_ir.__file__
assert torch.__file__.startswith(venv), torch.__file__
assert "+cpu" in torch.__version__, torch.__version__
print(f"preserved NGC JAX {{jax.__version__}} from {{jax.__file__}}")
print(f"preserved NGC JAXLIB {{jaxlib.__version__}} from {{jaxlib.__file__}}")
print(f"overlaid CUDA-13 CUTLASS DSL {{cutlass.__version__}} from {{cutlass.__file__}}")
PY
"""


def _ngc_xla_cuda_plugin_setup_script(artifact_uri: str, expected_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 must be 64 lowercase hexadecimal characters")
    copy_command = (
        "from pathlib import Path; import sys; "
        "from experiments.grug.moe.standalone.ngc_xla_kernel_cache_probe import copy_artifact; "
        "print(copy_artifact(sys.argv[1], Path(sys.argv[2]), sys.argv[3]))"
    )
    return f"""set -eu
test -f {_NGC_XLA_CUDA_PLUGIN_PATH}
printf '%s  %s\\n' {_NGC_XLA_STOCK_CUDA_PLUGIN_SHA256} {_NGC_XLA_CUDA_PLUGIN_PATH} | sha256sum --check --status
"$IRIS_VENV/bin/python" -c {shlex.quote(copy_command)} \\
  {shlex.quote(artifact_uri)} \\
  {_NGC_XLA_CUDA_PLUGIN_PATH} \\
  {expected_sha256}
"$IRIS_VENV/bin/python" - <<'PY'
from pathlib import Path

import jax
import jaxlib

jax.devices()
maps = Path("/proc/self/maps").read_text()
assert (
    "/opt/jaxlibs/jax_cuda13_pjrt/jax_plugins/xla_cuda13/xla_cuda_plugin.so" in maps
), "instrumented NGC CUDA PJRT plugin not present in /proc/self/maps"
assert jax.__file__.startswith("/opt/jax/"), jax.__file__
assert jaxlib.__file__.startswith("/opt/jaxlibs/"), jaxlib.__file__
print("loaded instrumented NGC CUDA PJRT plugin")
PY
"""


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_ENV_PREFIXES) and k not in _FORWARDED_ENV_EXCLUDE
    }


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray/Iris-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
    processes_per_task: int = 1,
    task_env_vars: dict[str, str] | None = None,
    task_setup_scripts: tuple[str, ...] = (),
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    forwarded_env = _forwarded_env_vars()
    if task_env_vars is not None:
        forwarded_env.update(task_env_vars)
    env_vars = resolve_training_env(base_env=forwarded_env, resources=resources)
    setup_scripts = (
        [nvidia_jax_overlay_setup_script()]
        if resources.image is not None and resources.image.startswith(_NVIDIA_JAX_IMAGE_PREFIX)
        else None
    )
    ngc_xla_artifact = forwarded_env.get("MARIN_NGC_XLA_CUDA_PLUGIN_URI")
    ngc_xla_sha256 = forwarded_env.get("MARIN_NGC_XLA_CUDA_PLUGIN_SHA256")
    if ngc_xla_artifact is not None or ngc_xla_sha256 is not None:
        if ngc_xla_artifact is None or ngc_xla_sha256 is None:
            raise ValueError("NGC XLA overlay requires both the artifact URI and SHA-256")
        if setup_scripts is None:
            raise ValueError("NGC XLA overlay requires an nvcr.io/nvidia/jax task image")
        setup_scripts.append(_ngc_xla_cuda_plugin_setup_script(ngc_xla_artifact, ngc_xla_sha256))
        current_library_path = env_vars.get("LD_LIBRARY_PATH")
        ngc_library_path = ":".join(_NGC_XLA_LIBRARY_PATHS)
        env_vars["LD_LIBRARY_PATH"] = (
            f"{ngc_library_path}:{current_library_path}" if current_library_path else ngc_library_path
        )
    if task_setup_scripts:
        if setup_scripts is None:
            setup_scripts = []
        setup_scripts.extend(task_setup_scripts)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(
            env_vars=env_vars,
            extras=extras_for_resources(resources),
            setup_scripts=setup_scripts,
        ),
        max_retries_failure=max_retries_failure,
        processes_per_task=processes_per_task,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
