# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the TPU throughput benchmark.

Kept separate from the production ``score.py`` so the benchmark's instrumentation doesn't
leak into the deployed path. Everything here reuses the production model / tokenizer /
windowing so the numbers reflect the real code.
"""

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import numpy as np
import tiktoken
from rigging.filesystem import marin_prefix, open_url, prefix_join

# Disable the tokenizers-lib internal rayon parallelism: this pipeline drives its own thread
# pool over row groups, so each thread's ``encode_batch`` runs single-threaded (the Rust encode
# still releases the GIL) rather than contending over one shared process-wide rayon pool.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from levanter.tokenizers import load_tokenizer

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, TIKTOKEN_PREFIX, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.scorer import CHUNK_CHARS, MODEL_META, MODEL_REMAP

# Calibration json name in a scorer dir.
MODEL_CALIB = "calib_bme.json"

# Datakit assets, relative to marin_prefix() so one path resolves to the region-appropriate copy
# on each cluster -- GCS on marin, S3 on CoreWeave.
DEFAULT_CORPUS = "normalized/nemotron_cc_v2/high_quality_b451aefe/outputs/main/part-*-of-04136.parquet"
FASTTEXT_MODEL = "datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin"


def resolve_dataset_path(path: str) -> str:
    """Root a cluster-relative datakit path at ``marin_prefix()``; pass absolute URLs through.

    Resolve on the driver, where the cluster config has injected ``MARIN_PREFIX``, so a relative
    corpus/model path picks up the local object store (GCS on marin, S3 on CoreWeave) with no
    cross-region read.
    """
    if "://" in path or path.startswith("/"):
        return path
    return prefix_join(marin_prefix(), path)


def doc_windows(text: str) -> list[str]:
    """The begin/middle/end ~512-token windows of a doc: the whole text if short, else the
    first, middle, and last ``CHUNK_CHARS``-char slices."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    m = len(text) // 2
    return [text[:CHUNK_CHARS], text[max(0, m - CHUNK_CHARS // 2) : m + CHUNK_CHARS // 2], text[-CHUNK_CHARS:]]


def load_remap_meta(model_dir: str) -> tuple[dict[int, int], str, int]:
    """Load (remap, tokenizer_name, max_tokens) from a scorer dir -- no ``.eqx`` needed."""
    with open_url(prefix_join(model_dir, MODEL_META), "r") as fh:
        meta = json.loads(fh.read())
    with open_url(prefix_join(model_dir, MODEL_REMAP), "r") as fh:
        remap = {int(k): int(v) for k, v in json.loads(fh.read()).items()}
    return remap, meta["tokenizer"], int(meta["max_tokens"])


def remap_to_array(remap: dict[int, int]) -> np.ndarray:
    """Dense lookup table indexed by raw HF token id -> compact id (UNK for pruned ids).

    Sized to ``max(remap) + 1``; callers guard raw ids beyond that range before indexing.
    """
    lut = np.full(max(max(remap) + 1, 1), UNK_ID, dtype=np.int32)
    for raw, compact in remap.items():
        lut[raw] = compact
    return lut


class BatchTokenizer(Protocol):
    """A tokenizer that encodes a batch of texts to raw id lists off the GIL."""

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...


class TiktokenBatchTokenizer:
    """Adapt a tiktoken encoding to the ``encode_batch`` interface the pipeline expects.

    ``num_threads=1`` because the pipeline already fans tokenization across a fork pool and a
    read-thread pool; tiktoken's own batch threads would oversubscribe the host.
    """

    def __init__(self, encoding_name: str):
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._encoding.encode_ordinary_batch(texts, num_threads=1)


def load_shared_tokenizer(tokenizer_name: str) -> BatchTokenizer:
    """Load a thread-shareable batch tokenizer by name.

    A ``tiktoken:<encoding>`` name (e.g. ``tiktoken:o200k_base``) loads a tiktoken BPE
    encoder; any other name loads the levanter HF tokenizer. Both are safe to share across
    the pipeline's read/fork workers: the levanter wrapper calls the Rust ``tokenizers``
    encoder with an immutable ``&self`` (unlike HF's ``PreTrainedTokenizerFast``, which
    rewrites truncation state per call), and tiktoken's encoder is likewise stateless per
    call, so concurrent ``encode_batch`` is safe and each call runs on one core.
    """
    if tokenizer_name.startswith(TIKTOKEN_PREFIX):
        return TiktokenBatchTokenizer(tokenizer_name.removeprefix(TIKTOKEN_PREFIX))
    return load_tokenizer(tokenizer_name)


def pack_windows(texts: list[str], tokenizer: BatchTokenizer, lut: np.ndarray, max_tokens: int) -> np.ndarray:
    """Tokenize (shared wrapper) + remap + right-pad window texts to ``[N, max_tokens]`` int32.

    ``encode_batch`` does not truncate, so ids are cut to ``max_tokens`` here.
    """
    encoded = tokenizer.encode_batch(texts)
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
