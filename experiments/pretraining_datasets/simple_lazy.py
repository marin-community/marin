# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-corpus datasets (starcoder, proofpile) as lazy ``Dataset`` handles.

The llama3-tokenized caches are pinned to their existing locations so referencing
them never re-tokenizes the corpora (~250B and ~55B tokens).
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import tokenized

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import PROOFPILE_LLAMA3_PIN, STARCODER_LLAMA3_PIN

# Raw download locations (pinned in simple.py's `_build_downloads`).
_STARCODER_RAW = "raw/starcoderdata-720c8c"
_PROOFPILE_RAW = "raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927"


def starcoder_dataset(*, tokenizer: str = llama3_tokenizer) -> Dataset:
    """StarCoder as a tokenized ``Dataset`` handle (text lives under the ``content`` key)."""
    return tokenized(
        "starcoderdata",
        tokenizer=tokenizer,
        version="llama3",
        paths=[_STARCODER_RAW],
        text_key="content",
        pin=STARCODER_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )


def proofpile_dataset(*, tokenizer: str = llama3_tokenizer) -> Dataset:
    """Proof-Pile 2 as a tokenized ``Dataset`` handle."""
    return tokenized(
        "proofpile_2",
        tokenizer=tokenizer,
        version="llama3",
        paths=[_PROOFPILE_RAW],
        pin=PROOFPILE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )
