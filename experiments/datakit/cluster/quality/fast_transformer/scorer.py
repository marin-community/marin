# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a trained fast-transformer and score arbitrary documents.

``train.py`` fits the model and ``data.py`` builds a compact vocabulary remap from
the training corpus; to score *new* text we need both the serialised model and that
remap. :class:`PooledScorer` bundles them, ``load_pooled_scorer`` builds one from a
model dir. ``bme_windows`` / ``pool_bme`` are the whole-doc (begin/middle/end)
scoring halves that production scoring runs around :meth:`PooledScorer.encode` and
:meth:`PooledScorer.predict_ids`; ``score_bme`` composes them for calibration fitting.
This module deliberately depends only on the model + inference forward, not on the
training loop or the zephyr/iris pipeline.
"""

import json
import os
import tempfile
from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
import numpy as np
from rigging.filesystem.factory import open_url

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID, encode_texts
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig

# bme scores begin/middle/end ~512-token (~2000-char) windows of the whole doc and
# mean-pools them, so a shared boilerplate prefix no longer dominates the score.
CHUNK_CHARS = 2_000

MODEL_STEM = "pooled_junkgate2"  # the deployed model artifact stem


def artifact_names(stem: str) -> tuple[str, str, str]:
    """The (.eqx, remap.json, meta.json) artifact filenames for a model stem."""
    return f"{stem}.eqx", f"{stem}_remap.json", f"{stem}_meta.json"


MODEL_EQX, MODEL_REMAP, MODEL_META = artifact_names(MODEL_STEM)


@dataclass(frozen=True)
class PooledScorer:
    """A trained fast-transformer plus its tokenizer + vocab remap, ready to score."""

    model: FastTransformer
    remap_table: np.ndarray
    tokenizer_name: str
    max_tokens: int

    @classmethod
    def load(cls, model_path: str, remap_path: str, meta_path: str) -> "PooledScorer":
        """Load from a serialised model, a remap JSON, and a meta JSON (config + tokenizer)."""
        with open_url(meta_path, "r") as fh:
            meta = json.loads(fh.read())
        with open_url(remap_path, "r") as fh:
            remap = {int(k): int(v) for k, v in json.loads(fh.read()).items()}
        # Rebuild from the full saved config so no field silently falls back to a dataclass
        # default (a sweep checkpoint may set a non-default final_pool / mlp_ratio). vocab_size
        # is authoritative from the remap; max_tokens falls back to the top-level meta for older
        # checkpoints that saved only a partial config.
        c = dict(meta["config"])
        c["vocab_size"] = len(remap) + 2  # PAD + UNK
        c.setdefault("max_tokens", meta["max_tokens"])
        config = FastTransformerConfig(**c)
        template = FastTransformer(config, key=jr.PRNGKey(0))
        # eqx deserialise needs a local file path
        model = eqx.tree_deserialise_leaves(model_path, template)
        return cls(
            model=model,
            remap_table=remap_table(remap),
            tokenizer_name=meta["tokenizer"],
            max_tokens=meta["max_tokens"],
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Compact token ids as a dense ``[N, max_tokens]`` array, PAD-padded on the right.

        This is the CPU half of scoring; :meth:`predict_ids` is the accelerator half,
        so a caller can run the two on different threads.
        """
        return remap_ids(encode_texts(self.tokenizer_name, texts, self.max_tokens), self.remap_table, self.max_tokens)

    def predict_ids(self, ids: np.ndarray) -> np.ndarray:
        """Quality score in ``[0, 1]`` per row of :meth:`encode` output."""
        return predict(self.model, ids)

    def score(self, texts: list[str]) -> np.ndarray:
        """Quality score in ``[0, 1]`` per document."""
        return self.predict_ids(self.encode(texts))


def remap_table(remap: dict[int, int]) -> np.ndarray:
    """Dense raw-id -> compact-id lookup; raw ids outside the table map to UNK."""
    table = np.full(max(remap) + 1, UNK_ID, dtype=np.int32)
    for raw, compact in remap.items():
        table[raw] = compact
    return table


def remap_ids(rows: list[list[int]], table: np.ndarray, max_tokens: int) -> np.ndarray:
    """Vectorized remap of tokenized rows into a ``[N, max_tokens]`` PAD-padded array."""
    ids = np.full((len(rows), max_tokens), PAD_ID, dtype=np.int32)
    last = len(table) - 1
    for i, row in enumerate(rows):
        raw = np.asarray(row[:max_tokens], dtype=np.int64)
        ids[i, : len(raw)] = np.where(raw <= last, table[np.minimum(raw, last)], UNK_ID)
    return ids


def load_pooled_scorer(model_dir: str) -> PooledScorer:
    """Load a `PooledScorer` from a model dir (streams the .eqx to a local path,
    which eqx deserialisation requires)."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    return PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")


def bme_windows(texts: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Begin/middle/end ~512-token windows of each doc, flattened, with each doc's
    ``[start, end)`` span into the flat list. Short docs (<= one chunk) are one window."""
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        if len(t) <= CHUNK_CHARS:
            cs = [t]
        else:
            m = len(t) // 2
            cs = [t[:CHUNK_CHARS], t[max(0, m - CHUNK_CHARS // 2) : m + CHUNK_CHARS // 2], t[-CHUNK_CHARS:]]
        spans.append((len(flat), len(flat) + len(cs)))
        flat.extend(cs)
    return flat, spans


def pool_bme(window_scores: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Mean-pool per-window scores back to one score per doc."""
    return np.array([window_scores[a:b].mean() for a, b in spans])


def score_bme(scorer: PooledScorer, texts: list[str]) -> np.ndarray:
    """Mean-pool the FT score over begin/middle/end ~512-token windows of each doc."""
    flat, spans = bme_windows(texts)
    return pool_bme(scorer.score(flat), spans)
