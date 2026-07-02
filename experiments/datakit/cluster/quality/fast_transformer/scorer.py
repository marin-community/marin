# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inference wrapper: score arbitrary documents with a trained fast-transformer.

``train.py`` fits the model and ``data.py`` builds a compact vocabulary remap from
the training corpus; to score *new* text we need both the serialised model and that
remap. :class:`PooledScorer` bundles them so a corpus can be scored the same way the
training eval was, returning a quality score in ``[0, 1]`` per document.
"""

import json
from dataclasses import dataclass

import equinox as eqx
import jax.random as jr
import numpy as np
from rigging.filesystem import open_url

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID, encode_texts
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import _predict


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
        vocab_size = len(remap) + 2  # PAD + UNK
        c = meta["config"]
        config = FastTransformerConfig(
            vocab_size=vocab_size,
            max_tokens=meta["max_tokens"],
            pool_window=c["pool_window"],
            pool_kind=c["pool_kind"],
            embed_dim=c["embed_dim"],
            hidden_dim=c["hidden_dim"],
            num_layers=c["num_layers"],
            num_heads=c["num_heads"],
        )
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
            out[start : start + len(chunk)] = _predict(self.model, ids)
        return out
