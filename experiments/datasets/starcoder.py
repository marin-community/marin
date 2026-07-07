# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StarCoder dataset as a lazy ``Dataset`` handle.

The llama3-tokenized cache is pinned to its existing location so referencing it
never re-tokenizes the corpus (~250B tokens).
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# Raw download location.
_STARCODER_RAW = "raw/starcoderdata-720c8c"

# Pinned llama3-tokenized cache; never re-tokenize this corpus (~250B tokens).
_STARCODER_LLAMA3_PIN = "tokenized/starcoderdata-12f018/"


def starcoder_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """StarCoder as a tokenized ``Dataset`` handle (text lives under the ``content`` key)."""
    return tokenized(
        "starcoderdata-llama3",
        tokenizer=tokenizer,
        version="2026.06.28",
        paths=[_STARCODER_RAW],
        text_key="content",
        pin=_STARCODER_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )


if __name__ == "__main__":
    dataset_main({"starcoderdata": starcoder_dataset()})
