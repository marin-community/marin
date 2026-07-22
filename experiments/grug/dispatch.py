# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
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
_FORWARDED_ENV_PREFIXES = ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_")
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)
_NVIDIA_JAX_IMAGE_PREFIX = "nvcr.io/nvidia/jax:"
_NVIDIA_JAX_UV_VERSION = "0.11.21"
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
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    setup_scripts = (
        [nvidia_jax_overlay_setup_script()]
        if resources.image is not None and resources.image.startswith(_NVIDIA_JAX_IMAGE_PREFIX)
        else None
    )
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
