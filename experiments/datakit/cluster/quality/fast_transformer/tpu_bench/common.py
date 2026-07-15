# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the TPU throughput benchmark (issue #7187).

Kept separate from the production ``score.py`` so the benchmark's instrumentation and
window-carrying schema don't leak into the deployed path. Everything here reuses the
production model / tokenizer / windowing so the numbers reflect the real code.
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from rigging.filesystem import open_url

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID, encode_texts
from experiments.datakit.cluster.quality.fast_transformer.scorer import CHUNK_CHARS, MODEL_META, MODEL_REMAP

# Peak bf16 FLOP/s per v6e chip (fray/device_flops.py: v6e bf16 = 918e12).
V6E_BF16_PEAK_FLOPS = 918e12

# Calibration json name in a scorer dir (matches score.py's MODEL_CALIB).
MODEL_CALIB = "calib_bme.json"


def doc_windows(text: str) -> list[str]:
    """The begin/middle/end ~512-token windows of a doc (mirrors ``score_bme``)."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    m = len(text) // 2
    return [text[:CHUNK_CHARS], text[max(0, m - CHUNK_CHARS // 2) : m + CHUNK_CHARS // 2], text[-CHUNK_CHARS:]]


def load_remap_meta(model_dir: str) -> tuple[dict[int, int], str, int]:
    """Load (remap, tokenizer_name, max_tokens) from a scorer dir -- no ``.eqx`` needed."""
    model_dir = model_dir.rstrip("/")
    with open_url(f"{model_dir}/{MODEL_META}", "r") as fh:
        meta = json.loads(fh.read())
    with open_url(f"{model_dir}/{MODEL_REMAP}", "r") as fh:
        remap = {int(k): int(v) for k, v in json.loads(fh.read()).items()}
    return remap, meta["tokenizer"], int(meta["max_tokens"])


def remap_to_array(remap: dict[int, int], vocab_size: int) -> np.ndarray:
    """Dense lookup table: raw HF token id -> compact id (UNK for pruned). Vectorizes the
    per-token ``remap.get`` loop so packing is a gather, not a Python dict loop."""
    hi = max(remap) + 1
    lut = np.full(max(hi, 1), UNK_ID, dtype=np.int32)
    for raw, compact in remap.items():
        lut[raw] = compact
    return lut


def pack_windows(texts: list[str], tokenizer_name: str, lut: np.ndarray, max_tokens: int) -> np.ndarray:
    """Tokenize + remap + right-pad a list of window texts to ``[N, max_tokens]`` int32.

    Uses the dense ``lut`` (vectorized gather) instead of the production per-token dict
    loop; the benchmark reports both so we can attribute the pack cost.
    """
    encoded = encode_texts(tokenizer_name, texts, max_tokens)
    ids = np.full((len(texts), max_tokens), PAD_ID, dtype=np.int32)
    lut_n = lut.shape[0]
    for i, row in enumerate(encoded):
        if not row:
            continue
        raw = np.asarray(row[:max_tokens], dtype=np.int64)
        raw = np.where(raw < lut_n, raw, 0)  # out-of-lut -> UNK via lut[0] path below
        ids[i, : raw.shape[0]] = lut[raw]
    return ids


@contextmanager
def accumulate(store: dict, key: str) -> Iterator[None]:
    """Accumulate elapsed wall-seconds into ``store[key]`` (per-worker timing breakdown)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


def write_result_json(path: str, payload: dict) -> None:
    with open_url(path, "w") as fh:
        fh.write(json.dumps(payload, indent=2))
