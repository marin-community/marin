# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Marin TPU-vLLM fork revisions for the isolated (uvx) serving path.

``marin-serve iris --tpu`` provisions the TPU vLLM stack from these forks when it runs
outside a workspace checkout (see :class:`marin.inference.vllm_server.IsolatedTpuVllm`),
where the root ``pyproject`` is unavailable. They mirror the git sources pinned in that
``pyproject``'s ``[tool.uv.sources]``; ``test_tpu_vllm_pins_match_pyproject`` keeps them
in sync so a fork refresh (``refresh-tpu-vllm-forks``) cannot silently drift the serving
pins from the workspace lock the in-checkout path uses.
"""

import re

# Keep these equal to the ``[tool.uv.sources]`` git/rev for ``vllm`` and ``tpu-inference``
# in the root pyproject; the sync test enforces it.
VLLM_FORK_URL = "https://github.com/marin-community/vllm.git"
VLLM_FORK_REV = "82e158743218550c2590161f299dcaffbdcd7746"
TPU_INFERENCE_FORK_URL = "https://github.com/marin-community/tpu-inference.git"
TPU_INFERENCE_FORK_REV = "0f60bf64458475cc2deccf12f47d6a048b2277a9"
# uvx otherwise resolves unconstrained transitive releases independently on
# every clean worker. Keep the release universe fixed for this qualified pair
# of fork revisions; bump it deliberately when refreshing the serving stack.
TPU_VLLM_EXCLUDE_NEWER = "2026-07-20T00:00:00Z"
_FULL_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


def fork_source_revision(requirement: str, *, package: str) -> str:
    """Extract the immutable Git revision from an isolated-launch requirement."""
    _, separator, revision = requirement.rpartition("@")
    if not separator or _FULL_GIT_REVISION.fullmatch(revision) is None:
        raise ValueError(f"{package} must be pinned to a full Git commit, got {requirement!r}")
    return revision


def vllm_fork_ref() -> str:
    """``uvx --from`` requirement for the pinned vLLM fork."""
    return f"vllm @ git+{VLLM_FORK_URL}@{VLLM_FORK_REV}"


def tpu_inference_fork_ref() -> str:
    """``uvx --with`` requirement for the pinned tpu-inference fork."""
    return f"tpu-inference @ git+{TPU_INFERENCE_FORK_URL}@{TPU_INFERENCE_FORK_REV}"
