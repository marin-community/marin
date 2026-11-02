# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch the installed vLLM Triton attention helper with marin-community/vllm#109.

An all-masked leading tile resets the running softmax maximum to zero, which makes sliding-window
attention depend on the prefill chunk boundary. The pinned wheel (39e62869693c) predates the fix.
Patch the source file before any process imports vLLM so Triton compiles the corrected kernel.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

from loguru import logger

HELPER = Path("v1/attention/ops/triton_attention_helpers.py")
OLD = """    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
"""
NEW = """    m_j_for_exp = tl.where(m_j > float("-inf"), m_j, 0.0)
"""
REPLACEMENTS = (
    (OLD, NEW),
    ("    P = tl.exp(S - m_j[:, None])\n", "    P = tl.exp(S - m_j_for_exp[:, None])\n"),
    ("    alpha = tl.exp(M - m_j)\n", "    alpha = tl.exp(M - m_j_for_exp)\n"),
)


def apply_triton_softmax_fix() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM is not installed")
    path = Path(spec.origin).parent / HELPER
    source = path.read_text()
    if NEW in source:
        return
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            raise RuntimeError(f"{path} does not match the vLLM revision the Triton softmax fix targets: {old!r}")
        source = source.replace(old, new)
    # Many processes on a node may race here; each writes the same bytes atomically.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(source)
    os.replace(handle.name, path)
    logger.info(f"Applied the Triton masked-tile softmax fix to {path}")
