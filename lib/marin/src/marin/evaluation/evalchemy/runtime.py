# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Evalchemy client runtime for endpoint-backed evaluations.

Evalchemy runs from an isolated uvx environment on the standard Iris task image. The immutable Git
revision identifies the CLI, graders, and benchmark data without requiring a separate task image.
"""

from marin.external_dependencies import EVALCHEMY

EVALCHEMY_REQUIREMENT = EVALCHEMY.requirement()

# Evalchemy's current dependency graph has Python 3.12 wheels. Letting uvx select Python 3.13 makes
# pandas 2.2.2 build from source during a cold start.
EVALCHEMY_PYTHON_VERSION = "3.12"
_PYTHON_ABI = f"cp{EVALCHEMY_PYTHON_VERSION.replace('.', '')}"

# The Marin client uploads both GCS and CoreWeave S3 artifacts. Evalchemy's lean endpoint core
# includes fsspec but intentionally leaves cloud filesystem implementations to its caller.
#
# Keep CPU-only PyTorch as a compatibility floor for benchmark orchestration and grading code.
# Inference stays in the separately served model process; direct wheel URLs prevent uv from
# resolving a CUDA build when a benchmark extra declares an unconstrained torch dependency.
_CPU_TORCH_VERSION = "2.11.0"
_CPU_TORCH_X86_64 = (
    "torch @ https://download.pytorch.org/whl/cpu/"
    f"torch-{_CPU_TORCH_VERSION}%2Bcpu-{_PYTHON_ABI}-{_PYTHON_ABI}-manylinux_2_28_x86_64.whl"
    " ; sys_platform == 'linux' and platform_machine == 'x86_64'"
)
_CPU_TORCH_AARCH64 = (
    "torch @ https://download.pytorch.org/whl/cpu/"
    f"torch-{_CPU_TORCH_VERSION}%2Bcpu-{_PYTHON_ABI}-{_PYTHON_ABI}-manylinux_2_28_aarch64.whl"
    " ; sys_platform == 'linux' and platform_machine == 'aarch64'"
)
EVALCHEMY_EXTRA_PACKAGES = ("s3fs", "gcsfs", _CPU_TORCH_X86_64, _CPU_TORCH_AARCH64)


def evalchemy_requirement(extras: tuple[str, ...] = ()) -> str:
    """Pin benchmark extras to the same immutable Evalchemy revision as the endpoint core."""
    return EVALCHEMY.requirement(extras)
