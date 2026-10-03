# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a trained fast-transformer and score documents given as token ids.

``train.py`` fits the model and ``data.py`` builds a compact vocabulary remap from
the training corpus; to score *new* documents we need both the serialised model and
that remap. :class:`PooledScorer` bundles them and ``load_pooled_scorer`` builds one
from a model dir. ``score_windowed`` scores pre-sliced token windows per document
(the scoring stage slices its own); ``score_bme`` slices begin/middle/end windows
from whole-document ids first (calibration fitting and tests). Windows are the
tokenize stage's ``input_ids`` (BOS/EOS included), each at most ``max_tokens`` long
and padded to ``max_tokens`` so ``predict`` compiles once. This module deliberately depends only on
the model + inference forward, not on the training loop or the zephyr/iris pipeline.
"""

import json
import os
import tempfile
from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
import numpy as np
from rigging.filesystem.factory import open_url

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, bme_windows, remap_table
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig

# CPU workers: 30.7 vs 44.2 CPU-s per 75k windows against predict's default, scores
# bit-identical; TPU callers keep ``inference.predict``'s default.
PREDICT_BATCH_SIZE = 64

MODEL_STEM = "pooled_junkgate2"  # the deployed model artifact stem
MODEL_EQX = f"{MODEL_STEM}.eqx"
MODEL_REMAP = f"{MODEL_STEM}_remap.json"
MODEL_META = f"{MODEL_STEM}_meta.json"


@dataclass(frozen=True)
class PooledScorer:
    """A trained fast-transformer plus its vocab remap table, ready to score token windows."""

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

    def score_windows(self, windows: list[np.ndarray], batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
        """Quality score in ``[0, 1]`` per token window (each padded to ``max_tokens``)."""
        ids = np.full((len(windows), self.max_tokens), PAD_ID, dtype=np.int32)
        last = len(self.remap_table) - 1
        for i, window in enumerate(windows):
            ids[i, : len(window)] = self.remap_table[np.minimum(window, last)]
        return predict(self.model, ids, batch_size=batch_size)


def load_pooled_scorer(model_dir: str) -> PooledScorer:
    """Load a `PooledScorer` from a model dir (streams the .eqx to a local path,
    which eqx deserialisation requires)."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    return PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")


def score_windowed(scorer: PooledScorer, docs: list[list[np.ndarray]]) -> np.ndarray:
    """Mean-pool the FT score over each doc's pre-sliced token windows.
    All windows go through one ``score_windows`` call; spans map back per doc."""
    flat: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for windows in docs:
        spans.append((len(flat), len(flat) + len(windows)))
        flat.extend(windows)
    s = scorer.score_windows(flat)
    return np.array([s[a:b].mean() for a, b in spans])


def score_bme(scorer: PooledScorer, docs: list[np.ndarray]) -> np.ndarray:
    """Mean-pool the FT score over begin/middle/end ``max_tokens`` windows of each doc's ids.
    Short docs (<= one window) reduce to a single scored window."""
    return score_windowed(scorer, [bme_windows(d, scorer.max_tokens) for d in docs])
