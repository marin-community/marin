# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Harrier inference for local CPU and single-GPU execution."""

from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import numpy as np

HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"
HARRIER_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
HARRIER_DIMENSIONS = 1024


@cache
def _harrier_model(
    model: str, revision: str, tokenizer_revision: str, compute_dtype: str, device: str
) -> tuple[Any, Any]:
    import torch  # noqa: PLC0415 - optional CPU/GPU inference dependency
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415 - optional inference dependency

    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(model, revision=tokenizer_revision, padding_side="right")
    encoder = AutoModel.from_pretrained(model, revision=revision, dtype=dtypes[compute_dtype]).to(device).eval()
    return tokenizer, encoder


@dataclass(frozen=True)
class HarrierEmbedder:
    """Load a model once per process and reject inputs that would be truncated."""

    model: str
    revision: str
    tokenizer_revision: str
    dimensions: int
    max_tokens: int
    compute_dtype: str
    device: Literal["cpu", "cuda"]
    batch_size: int = 16
    cpu_threads: int = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch  # noqa: PLC0415 - optional CPU/GPU inference dependency

        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if self.device == "cpu":
            torch.set_num_threads(self.cpu_threads)
        tokenizer, model = _harrier_model(
            self.model, self.revision, self.tokenizer_revision, self.compute_dtype, self.device
        )
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            inputs = tokenizer(
                texts[start : start + self.batch_size], padding=True, truncation=False, return_tensors="pt"
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            if int(lengths.max()) > self.max_tokens:
                raise ValueError(f"Harrier input exceeds the configured {self.max_tokens}-token limit")
            inputs = inputs.to(self.device)
            positions = (lengths - 1).to(self.device)
            with torch.inference_mode():
                hidden = model(**inputs, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(lengths), device=self.device), positions].float()
                vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        result = np.concatenate(vectors)
        if result.shape != (len(texts), self.dimensions):
            raise ValueError(f"Unexpected Harrier embedding shape: {result.shape}")
        return result
