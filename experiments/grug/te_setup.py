# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer Engine install recipe for Grug GPU jobs.

This is the only known-good way to get TE onto Marin's GB200 image: the CCCL headers, the cuDNN
frontend, and the CUDA 13 TE core must be installed *before* the JAX extension builds without
isolation, and the NCCL wheel needs a task-local ``libnccl.so`` link for the extension to link
against. Every step and every environment variable below was established by a bounded probe job;
see marin-community/marin#8141.

The resulting TE 2.17.1 build imports cleanly but its context-parallel backward fails cuDNN
workspace sizing on this image, so these constants exist to make the next attempt cheap, not
because a TE run has succeeded.
"""

import sys
from collections.abc import Sequence

from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script

# Concrete path of ``$IRIS_VENV`` inside an Iris task. Environment values reach the task as a
# literal map with no shell expansion, so the build variables below cannot use the variable itself;
# the setup script, which is bash, does.
IRIS_TASK_VENV = "/app/.venv"

TRANSFORMER_ENGINE_SETUP_SCRIPT = r"""set -e
cd "$IRIS_WORKDIR"
uv pip install --python "$IRIS_VENV/bin/python" \
  nvidia-cuda-cccl==13.3.3.4.1 \
  nvidia-cudnn-frontend==1.25.0 \
  transformer_engine_cu13==2.17.1
site_packages="$("$IRIS_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
nccl_lib="$site_packages/nvidia/nccl/lib"
if [ ! -e "$nccl_lib/libnccl.so" ]; then
  ln -s libnccl.so.2 "$nccl_lib/libnccl.so"
fi
export LIBRARY_PATH="$nccl_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$nccl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv pip install --python "$IRIS_VENV/bin/python" \
  --no-build-isolation --no-deps transformer_engine_jax==2.17.1
uv pip install --python "$IRIS_VENV/bin/python" --no-deps transformer_engine==2.17.1
"""


def transformer_engine_build_env(python_version: str | None = None) -> dict[str, str]:
    """Task environment for building and running the pinned TE wheels.

    ``python_version`` selects the venv's ``site-packages`` and defaults to the submitter's, the
    same version :func:`transformer_engine_setup_scripts` builds the venv with.
    """
    python_version = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = f"{IRIS_TASK_VENV}/lib/python{python_version}/site-packages"
    return {
        # TE 2.17.1 falls back to CUDA 12 when JAX 0.11 does not expose its private runtime-version API.
        "CUDA_VERSION": "13.0",
        # The GPU workspace contains CUDA 12 packages for optional dependencies. cuDNN frontend must
        # bind to the CUDA 13 runtime used by JAX and the TE core, at build time and at run time.
        "CUDNN_FRONTEND_CUDART_LIB_NAME": "libcudart.so.13",
        # CUDA 13's unified pip layout nests the CCCL headers below the directory TE discovers.
        "CPLUS_INCLUDE_PATH": (
            f"{site_packages}/nvidia/cu13/include/cccl:"
            f"{site_packages}/nvidia/nvtx/include:"
            f"{site_packages}/include"
        ),
        "NVTE_BUILD_USE_NVIDIA_WHEELS": "1",
        "NVTE_CUDA_ARCHS": "100",
        "NVTE_WITH_NCCL_EP": "0",
    }


def transformer_engine_setup_scripts(extras: Sequence[str] = ("gpu",)) -> Sequence[str]:
    """Full setup-script chain for a GPU task that needs Transformer Engine.

    Pass with :func:`transformer_engine_build_env` as ``extra_env_vars``. Supplying setup scripts
    replaces the task's whole setup, so the standard venv build and the CUDA toolchain staging are
    rendered here rather than left to Iris; ``extras`` therefore has to name every extra the task
    needs, since Iris ignores the resource extras once scripts are given.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        default_setup_script(extras=extras, python_version=python_version),
        cuda_toolchain_setup_script(),
        TRANSFORMER_ENGINE_SETUP_SCRIPT,
    )
