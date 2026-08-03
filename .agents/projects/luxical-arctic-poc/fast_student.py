# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast-transformer embedding student used by the Arctic POC."""

import functools
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import jax.random as jr
import numpy as np
import pyarrow as pa
from huggingface_hub import hf_hub_download
from ladder_config import teacher_windows_from_view
from luxical.tokenization import ArrowTokenizer

from experiments.datakit.cluster.quality.fast_transformer.data import UNK_ID, load_tokenizer
from experiments.datakit.cluster.quality.fast_transformer.embedding import pack_remapped_windows, predict_embeddings
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastEmbeddingTransformer,
    FastTransformerConfig,
)

E5_TOKENIZER_NAME = "intfloat/multilingual-e5-small"
LUXICAL_TOKENIZER_NAME = "luxical-one-arrow"
TOKENIZER_NAME = LUXICAL_TOKENIZER_NAME
BASELINE_REPO = "DatologyAI/luxical-one"
BASELINE_FILE = "luxical_one_rc4.npz"
BASELINE_REVISION = "474cfeb959dd473b3d1cd61da630f566037e69e2"
MAX_TOKENS = 256
TOKENS_PER_DOCUMENT_WINDOW = MAX_TOKENS
POOL_WINDOW = 64
OUTPUT_DIMENSION = 256
COMPACT_VOCAB_SIZE = 65_536
WINDOWS_PER_DOCUMENT = 1
SOURCE_WINDOWS_PER_DOCUMENT = 3
CHARACTERS_PER_SOURCE_WINDOW = 256

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "large": {
        "embed_dim": 512,
        "hidden_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
    },
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


def tokenizer_vocab_size(tokenizer_name: str) -> int:
    """Return the addressable raw vocabulary size for one tokenizer treatment."""
    if tokenizer_name == LUXICAL_TOKENIZER_NAME:
        _, vocab_size = luxical_tokenizer()
        return vocab_size
    tokenizer: Any = load_tokenizer(tokenizer_name)
    return len(tokenizer)


@functools.lru_cache(maxsize=1)
def luxical_tokenizer() -> tuple[ArrowTokenizer, int]:
    """Load only the pinned stock Luxical tokenizer state."""
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        tokenizer_state = archive["tokenizer"].tobytes().decode("utf-8")
    vocabulary = json.loads(tokenizer_state)["model"]["vocab"]
    vocab_size = max(vocabulary.values()) + 1
    return ArrowTokenizer(tokenizer_state), vocab_size


def packed_document_ids(
    texts: list[str],
    raw_to_compact: np.ndarray,
    tokenizer_name: str = TOKENIZER_NAME,
) -> np.ndarray:
    """Tokenize head, middle, and tail windows into one fixed-width student input."""
    if raw_to_compact.ndim != 1:
        raise ValueError(f"Expected a one-dimensional token remap, got {raw_to_compact.shape}")
    grouped_ids = raw_document_window_ids(texts, tokenizer_name)
    return pack_remapped_windows(
        grouped_ids,
        raw_to_compact,
        MAX_TOKENS,
        TOKENS_PER_DOCUMENT_WINDOW,
    )


def raw_document_window_ids(
    texts: list[str],
    tokenizer_name: str = TOKENIZER_NAME,
) -> list[list[Sequence[int]]]:
    """Tokenize one bounded head, middle, and tail view per document."""
    flat_windows = [fast_document_view(text) for text in texts]
    if tokenizer_name == LUXICAL_TOKENIZER_NAME:
        tokenizer, _ = luxical_tokenizer()
        token_lists = tokenizer.tokenize(pa.array(flat_windows), add_special_tokens=False)
        raw_ids = [row[:TOKENS_PER_DOCUMENT_WINDOW] for row in token_lists.to_numpy(zero_copy_only=False)]
    else:
        tokenizer: Any = load_tokenizer(tokenizer_name)
        raw_ids = tokenizer(
            flat_windows,
            add_special_tokens=False,
            truncation=True,
            max_length=TOKENS_PER_DOCUMENT_WINDOW,
        )["input_ids"]
    grouped_ids = [
        raw_ids[start : start + WINDOWS_PER_DOCUMENT] for start in range(0, len(raw_ids), WINDOWS_PER_DOCUMENT)
    ]
    return grouped_ids


def fast_document_view(text: str) -> str:
    """Return one short view that keeps characters from three document regions."""
    if len(text) <= 3 * CHARACTERS_PER_SOURCE_WINDOW:
        return text
    head, middle, tail = teacher_windows_from_view(text)
    middle_start = max(0, len(middle) // 2 - CHARACTERS_PER_SOURCE_WINDOW // 2)
    return "\n".join(
        (
            head[:CHARACTERS_PER_SOURCE_WINDOW],
            middle[middle_start : middle_start + CHARACTERS_PER_SOURCE_WINDOW],
            tail[-CHARACTERS_PER_SOURCE_WINDOW:],
        )
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
    def random(
        cls,
        config_name: str,
        raw_to_compact: np.ndarray,
        seed: int,
        tokenizer_name: str = TOKENIZER_NAME,
    ) -> "FastStudent":
        config = fast_student_config(config_name, vocab_size=int(raw_to_compact.max()) + 1)
        model = FastEmbeddingTransformer(config, OUTPUT_DIMENSION, key=jr.PRNGKey(seed))
        return cls(model, raw_to_compact, tokenizer_name)

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
            "source_windows_per_document": SOURCE_WINDOWS_PER_DOCUMENT,
            "characters_per_source_window": CHARACTERS_PER_SOURCE_WINDOW,
            "config": asdict(self.model.backbone.config),
        }
