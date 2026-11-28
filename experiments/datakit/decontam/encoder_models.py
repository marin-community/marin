# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decontamination models: two seeds of a fine-tuned mmBERT-base that score (item, excerpt) pairs.

:class:`EncoderModels` is the :class:`marin.datakit.decon.PairScorer` of the reference pipeline; it reads the
pair text of :mod:`experiments.datakit.decontam.pair_text`. ``encoder/`` holds the training and the measurements.
"""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.datakit.decontam.pair_text import OUTPUTS, item_texts, with_pair_text

SCORE_BATCH = 64
MAX_INPUT_TOKENS = 1024
# Later models score only the pairs that the first model scores at least this high; the others keep the first
# model's score. On 1,673,046 scored catalog pairs, 96.7% score below it, and 6 of 14,075 pairs with a
# two-model mean of at least 0.5 had a first score below it.
LATER_MODELS_FLOOR = 0.01


@dataclass(frozen=True)
class EncoderModels:
    """The encoder classifiers of the decontamination, one per seed; a pair's score is the mean of the models'
    probabilities that the excerpt holds the item. Each path is a storage prefix of a saved Hugging Face model
    with its tokenizer. ``version`` names the weights (e.g. ``"it16-s0-s1"``) and enters the score step's
    identity in place of the region-specific paths."""

    paths: tuple[str, ...]
    version: str

    def load(self, items_dir: str, device: str) -> Callable[[pd.DataFrame], np.ndarray]:
        import torch  # noqa: PLC0415  # optional dep: torch, needed only on the score workers

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("the score workers have a GPU but torch has no CUDA; see ScoreWorkers.environment")
            # The cuDNN attention backend builds no plan for these padded bf16 batches; the others do.
            torch.backends.cuda.enable_cudnn_sdp(False)
        models, tokenizers = [], []
        for path in self.paths:
            with tempfile.TemporaryDirectory() as directory:
                for file in StoragePath(prefix_join(path, "*")).glob():
                    (Path(directory) / file.name).write_bytes(file.read_bytes())
                models.append(AutoModelForSequenceClassification.from_pretrained(directory).to(device).eval())
                tokenizers.append(AutoTokenizer.from_pretrained(directory))
        if any(t.get_vocab() != tokenizers[0].get_vocab() for t in tokenizers):
            raise ValueError("the decontamination models must share a tokenizer")
        texts = item_texts(items_dir)
        head = OUTPUTS.index("label_q")

        @torch.no_grad()
        def score(pairs: pd.DataFrame) -> np.ndarray:
            framed = with_pair_text(pairs, texts)
            order = np.argsort(framed["right"].str.len().to_numpy())
            result = np.zeros(len(framed))
            for start in range(0, len(order), SCORE_BATCH):
                rows = order[start : start + SCORE_BATCH]
                encoded = tokenizers[0](
                    list(framed["left"].iloc[rows]),
                    list(framed["right"].iloc[rows]),
                    truncation="longest_first",
                    max_length=MAX_INPUT_TOKENS,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    first = torch.sigmoid(models[0](**inputs).logits[:, head].float())
                    values = [first]
                    keep = (first >= LATER_MODELS_FLOOR).nonzero(as_tuple=True)[0]
                    for model in models[1:]:
                        later = first.clone()
                        if len(keep):
                            subset = {key: value[keep] for key, value in inputs.items()}
                            later[keep] = torch.sigmoid(model(**subset).logits[:, head].float())
                        values.append(later)
                result[rows] = torch.stack(values, dim=1).mean(dim=1).cpu().numpy()
            return result

        return score
