# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic seeds for Marin-side mismatch-probe analysis.

Mirrors ``vendor/skyrl/skyrl-train/skyrl_train/mismatch_probe/protocol.py``: the
vendored fork is not importable from the Marin tree, so the derivation is kept
byte-identical on both sides. The trainer side owns the canonical definition.
"""

from __future__ import annotations

import hashlib


def request_seed(master_seed: int, prompt_id: str, repetition: int) -> int:
    """Derive a stable per-answer vLLM seed independent of Python hash salt."""
    payload = f"{master_seed}:{len(prompt_id)}:{prompt_id}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
