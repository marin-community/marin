# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proof-Pile 2 dataset as a lazy ``Dataset`` handle.

The llama3-tokenized cache is pinned to its existing location so referencing it
never re-tokenizes the corpus (~55B tokens).
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# Raw download location.
_PROOFPILE_RAW = "raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927"

# Pinned llama3-tokenized cache; never re-tokenize this corpus (~55B tokens).
_PROOFPILE_LLAMA3_PIN = "tokenized/proofpile_2-4a35c7/"


def proofpile_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """Proof-Pile 2 as a tokenized ``Dataset`` handle."""
    return tokenized(
        "proofpile_2-llama3",
        tokenizer=tokenizer,
        version="2026.06.28",
        paths=[_PROOFPILE_RAW],
        pin=_PROOFPILE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )


if __name__ == "__main__":
    dataset_main({"proofpile_2": proofpile_dataset()})
