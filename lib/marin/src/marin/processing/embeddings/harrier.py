# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Harrier inference for local CPU and single-GPU execution."""

from dataclasses import dataclass
from functools import cache
from typing import Literal, cast

import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

HARRIER_REPO = "microsoft/harrier-oss-v1-0.6b"
HARRIER_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
HARRIER_DIM = 1024
HARRIER_MAX_TOKENS = 8_192


@dataclass(frozen=True)
class HarrierConfig:
    model: str
    revision: str
    tokenizer_revision: str
    dimensions: int
    max_tokens: int
    compute_dtype: str
    device: Literal["cpu", "cuda"]
    batch_size: int = 16
    batch_token_limit: int = 16_384
    cpu_threads: int = 4


class HarrierEmbedder:
    def __init__(self, config: HarrierConfig) -> None:
        import torch  # noqa: PLC0415 - optional CPU/GPU inference dependency

        self.config = config
        if config.device == "cpu":
            torch.set_num_threads(config.cpu_threads)
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(config.model, revision=config.tokenizer_revision, padding_side="right"),
        )
        self.model = (
            AutoModel.from_pretrained(config.model, revision=config.revision, dtype=dtypes[config.compute_dtype])
            .to(config.device)
            .eval()
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch  # noqa: PLC0415 - optional CPU/GPU inference dependency

        config = self.config
        if not texts:
            return np.empty((0, config.dimensions), dtype=np.float32)
        encoded = self.tokenizer(texts, padding=False, truncation=False)
        token_lengths = np.asarray([len(tokens) for tokens in encoded["input_ids"]])
        if int(token_lengths.max()) > config.max_tokens:
            raise ValueError(f"Harrier input exceeds the configured {config.max_tokens}-token limit")
        order = np.argsort(token_lengths, kind="stable")
        vectors = []
        start = 0
        while start < len(texts):
            end = start + 1
            while end < min(start + config.batch_size, len(texts)):
                padded_tokens = int(token_lengths[order[end]]) * (end - start + 1)
                if padded_tokens > config.batch_token_limit:
                    break
                end += 1
            positions = order[start:end]
            inputs = self.tokenizer.pad(
                {key: [values[index] for index in positions] for key, values in encoded.items()},
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            inputs = inputs.to(config.device)
            positions = (lengths - 1).to(config.device)
            with torch.inference_mode():
                hidden = self.model(**inputs, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(lengths), device=config.device), positions].float()
                vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            start = end
        result = np.concatenate(vectors)
        if result.shape != (len(texts), config.dimensions):
            raise ValueError(f"Unexpected Harrier embedding shape: {result.shape}")
        return result[np.argsort(order)]


@cache
def load_harrier(config: HarrierConfig) -> HarrierEmbedder:
    """Initialize one inference backend per worker process and configuration."""
    return HarrierEmbedder(config)
