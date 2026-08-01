# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast-transformer embedding student used by the Arctic POC."""

from dataclasses import asdict
from typing import Any

import jax.random as jr
import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.data import UNK_ID, load_tokenizer
from experiments.datakit.cluster.quality.fast_transformer.embedding import pack_remapped_windows, predict_embeddings
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastEmbeddingTransformer,
    FastTransformerConfig,
)
from ladder_config import teacher_windows_from_view

TOKENIZER_NAME = "intfloat/multilingual-e5-small"
MAX_TOKENS = 512
TOKENS_PER_DOCUMENT_WINDOW = 160
POOL_WINDOW = 64
OUTPUT_DIMENSION = 256
COMPACT_VOCAB_SIZE = 65_536
WINDOWS_PER_DOCUMENT = 3

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {
        "embed_dim": 256,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
    },
    "slim": {
        "embed_dim": 96,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
    },
}


def fast_student_config(name: str, vocab_size: int = COMPACT_VOCAB_SIZE) -> FastTransformerConfig:
    """Return one fixed student treatment config."""
    try:
        treatment = MODEL_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"Unknown fast-student config: {name}") from error
    return FastTransformerConfig(
        vocab_size=vocab_size,
        max_tokens=MAX_TOKENS,
        pool_window=POOL_WINDOW,
        pool_kind="meanmaxmin",
        dropout=0.1,
        final_pool="mean",
        **treatment,
    )


def provisional_remap(raw_vocab_size: int, compact_vocab_size: int = COMPACT_VOCAB_SIZE) -> np.ndarray:
    """Return a speed-only remap with the same runtime shape as the trained vocabulary."""
    remap = np.full(raw_vocab_size, UNK_ID, dtype=np.int32)
    kept = min(raw_vocab_size, compact_vocab_size - 2)
    remap[:kept] = np.arange(kept, dtype=np.int32) + 2
    return remap


def packed_document_ids(
    texts: list[str],
    raw_to_compact: np.ndarray,
    tokenizer_name: str = TOKENIZER_NAME,
) -> np.ndarray:
    """Tokenize head, middle, and tail windows into one fixed-width student input."""
    if raw_to_compact.ndim != 1:
        raise ValueError(f"Expected a one-dimensional token remap, got {raw_to_compact.shape}")
    tokenizer: Any = load_tokenizer(tokenizer_name)
    document_windows = [teacher_windows_from_view(text) for text in texts]
    flat_windows = [window for windows in document_windows for window in windows]
    raw_ids = tokenizer(
        flat_windows,
        add_special_tokens=False,
        truncation=True,
        max_length=TOKENS_PER_DOCUMENT_WINDOW,
    )["input_ids"]
    grouped_ids = [
        raw_ids[start : start + WINDOWS_PER_DOCUMENT]
        for start in range(0, len(raw_ids), WINDOWS_PER_DOCUMENT)
    ]
    return pack_remapped_windows(
        grouped_ids,
        raw_to_compact,
        MAX_TOKENS,
        TOKENS_PER_DOCUMENT_WINDOW,
    )


class FastStudent:
    """Text-to-vector wrapper for one fast-transformer embedding model."""

    def __init__(
        self,
        model: FastEmbeddingTransformer,
        raw_to_compact: np.ndarray,
        tokenizer_name: str = TOKENIZER_NAME,
    ) -> None:
        self.model = model
        self.raw_to_compact = raw_to_compact
        self.tokenizer_name = tokenizer_name

    @classmethod
    def random(cls, config_name: str, raw_to_compact: np.ndarray, seed: int) -> "FastStudent":
        config = fast_student_config(config_name)
        model = FastEmbeddingTransformer(config, OUTPUT_DIMENSION, key=jr.PRNGKey(seed))
        return cls(model, raw_to_compact)

    def __call__(self, texts: list[str], batch_size: int = 4_096) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            ids = packed_document_ids(
                texts[start : start + batch_size],
                self.raw_to_compact,
                self.tokenizer_name,
            )
            outputs.append(predict_embeddings(self.model, ids, batch_size=batch_size))
        if not outputs:
            return np.empty((0, OUTPUT_DIMENSION), dtype=np.float32)
        vectors = np.concatenate(outputs)
        if not np.isfinite(vectors).all():
            raise ValueError("Fast student returned non-finite vectors")
        return vectors

    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer_name,
            "output_dimension": OUTPUT_DIMENSION,
            "tokens_per_document_window": TOKENS_PER_DOCUMENT_WINDOW,
            "windows_per_document": WINDOWS_PER_DOCUMENT,
            "config": asdict(self.model.backbone.config),
        }
