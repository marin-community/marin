# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Marin TPU-vLLM fork revisions for the isolated (uvx) serving path.

``marin-serve --tpu`` provisions the TPU vLLM stack from these forks when it runs
outside a workspace checkout (see :class:`marin.inference.vllm_server.IsolatedTpuVllm`),
where the root ``pyproject`` is unavailable. They mirror the git sources pinned in that
``pyproject``'s ``[tool.uv.sources]``; ``test_tpu_vllm_pins_match_pyproject`` keeps them
in sync so a fork refresh (``refresh-tpu-vllm-forks``) cannot silently drift the serving
pins from the workspace lock the in-checkout path uses.
"""

# Keep these equal to the ``[tool.uv.sources]`` git/rev for ``vllm`` and ``tpu-inference``
# in the root pyproject; the sync test enforces it.
VLLM_FORK_URL = "https://github.com/marin-community/vllm.git"
VLLM_FORK_REV = "afb26719464d5957e695bde478ae93a160b11d14"
TPU_INFERENCE_FORK_URL = "https://github.com/marin-community/tpu-inference.git"
TPU_INFERENCE_FORK_REV = "734d2842aa883c8f7bcff87a4b437a366f3adbc0"


def vllm_fork_ref() -> str:
    """``uvx --from`` requirement for the pinned vLLM fork."""
    return f"vllm @ git+{VLLM_FORK_URL}@{VLLM_FORK_REV}"


def tpu_inference_fork_ref() -> str:
    """``uvx --with`` requirement for the pinned tpu-inference fork."""
    return f"tpu-inference @ git+{TPU_INFERENCE_FORK_URL}@{TPU_INFERENCE_FORK_REV}"
