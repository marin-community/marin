# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-corpus datasets (starcoder, proofpile) as lazy ``Dataset`` handles.

The llama3-tokenized caches are pinned to their existing locations so referencing
them never re-tokenizes the corpora (~250B and ~55B tokens). This is the handle
version of the ``starcoderdata`` / ``proofpile_2`` entries in ``simple.py``'s
``tokenized`` dict.
"""

from levanter.data.text import TextLmDatasetFormat
from marin.execution.lazy import Dataset
from marin.experiment.data import tokenize

from experiments.llama import llama3_tokenizer

# Raw download locations (pinned in simple.py's `_build_downloads`).
_STARCODER_RAW = "raw/starcoderdata-720c8c"
_PROOFPILE_RAW = "raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927"

# Existing llama3-tokenized caches (pinned in simple.py's `tokenized`).
_STARCODER_LLAMA3_PIN = "tokenized/starcoderdata-12f018/"
_PROOFPILE_LLAMA3_PIN = "tokenized/proofpile_2-4a35c7/"


def starcoder_dataset(*, tokenizer: str = llama3_tokenizer) -> Dataset:
    """StarCoder as a tokenized ``Dataset`` handle (text lives under the ``content`` key)."""
    pinned = _STARCODER_LLAMA3_PIN if tokenizer == llama3_tokenizer else None
    return tokenize(
        "tokenized/starcoderdata",
        "llama3",
        train_paths=[_STARCODER_RAW],
        tokenizer=tokenizer,
        format=TextLmDatasetFormat(text_key="content"),
        pinned_path=pinned,
    )


def proofpile_dataset(*, tokenizer: str = llama3_tokenizer) -> Dataset:
    """Proof-Pile 2 as a tokenized ``Dataset`` handle."""
    pinned = _PROOFPILE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None
    return tokenize(
        "tokenized/proofpile_2",
        "llama3",
        train_paths=[_PROOFPILE_RAW],
        tokenizer=tokenizer,
        pinned_path=pinned,
    )
