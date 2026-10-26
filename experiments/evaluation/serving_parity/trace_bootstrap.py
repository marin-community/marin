# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hash-guarded static-window logging without attention arithmetic changes."""

import hashlib
import importlib.util
from pathlib import Path

PRISTINE_SHA256 = "8bba8f9f71acdc618fdda20614b5c49a0f6949a176ad929db9a24b5c38a828bd"
ANCHOR = "                # With update_kv_cache=False the kernel returns the cache\n"
TRACE = (
    "                # Static layer trace only; no output or cache changes.\n"
    '                print("SERVING_PARITY_LAYER_STATIC", layer.layer_name,\n'
    '                      "sliding_window=" + str(self.sliding_window), flush=True)\n'
)


def patch_trace() -> dict:
    root = Path(importlib.util.find_spec("tpu_inference").origin).parent
    target = root / "layers/vllm/backends/flash_attn.py"
    source = target.read_text()
    pristine = source.replace(TRACE + ANCHOR, ANCHOR)
    if hashlib.sha256(pristine.encode()).hexdigest() != PRISTINE_SHA256 or pristine.count(ANCHOR) != 1:
        raise RuntimeError("Unexpected fork source; refusing static trace patch")
    expected = pristine.replace(ANCHOR, TRACE + ANCHOR)
    if source not in (pristine, expected):
        raise RuntimeError("Unexpected static trace instrumentation")
    if source != expected:
        target.write_text(expected)
    return dict(
        kind="static_layer_trace",
        arithmetic_changed=False,
        before_sha256=PRISTINE_SHA256,
        after_sha256=hashlib.sha256(expected.encode()).hexdigest(),
    )
