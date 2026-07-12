# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a trained fast-transformer and score arbitrary documents.

``train.py`` fits the model and ``data.py`` builds a compact vocabulary remap from
the training corpus; to score *new* text we need both the serialised model and that
remap. :class:`PooledScorer` bundles them, ``load_pooled_scorer`` builds one from a
model dir, and ``score_bme`` is the whole-doc (begin/middle/end) scoring used by both
production scoring and calibration fitting. This module deliberately depends only on
the model + inference forward, not on the training loop or the zephyr/iris pipeline.
"""

import json
import os
import tempfile
from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
import numpy as np
from rigging.filesystem import open_url

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID, encode_texts
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig

BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)
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
    remap: dict[int, int]
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
        return cls(model=model, remap=remap, tokenizer_name=meta["tokenizer"], max_tokens=meta["max_tokens"])

    def score(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Quality score in ``[0, 1]`` per document."""
        out = np.empty(len(texts), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = encode_texts(self.tokenizer_name, chunk, self.max_tokens)
            ids = np.full((len(chunk), self.max_tokens), PAD_ID, dtype=np.int32)
            for i, row in enumerate(encoded):
                mapped = [self.remap.get(t, UNK_ID) for t in row[: self.max_tokens]]
                ids[i, : len(mapped)] = mapped
            out[start : start + len(chunk)] = predict(self.model, ids)
        return out


def load_pooled_scorer(model_dir: str) -> PooledScorer:
    """Load a `PooledScorer` from a model dir (streams the .eqx to a local path,
    which eqx deserialisation requires)."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    return PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")


def score_bme(scorer: PooledScorer, texts: list[str]) -> np.ndarray:
    """Mean-pool the FT score over begin/middle/end ~512-token windows of each doc.
    Short docs (<= one chunk) reduce to a single scored window."""
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
    s = scorer.score(flat)
    return np.array([s[a:b].mean() for a, b in spans])
