# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-corpus datasets (starcoder, proofpile) as lazy ``Dataset`` handles.

The llama3-tokenized caches are pinned to their existing locations so referencing
them never re-tokenizes the corpora (~250B and ~55B tokens).
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# Raw download locations.
_STARCODER_RAW = "raw/starcoderdata-720c8c"
_PROOFPILE_RAW = "raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927"
_DCLM_BASELINE_RAW = "raw/dclm/a3b142c"

# Pinned llama3-tokenized caches; never re-tokenize these corpora
# (dclm-baseline ~3.8T, starcoderdata ~250B, proof-pile-2 ~55B tokens).
_DCLM_BASELINE_LLAMA3_PIN = "tokenized/dclm_baseline-0206f1/"
_STARCODER_LLAMA3_PIN = "tokenized/starcoderdata-12f018/"
_PROOFPILE_LLAMA3_PIN = "tokenized/proofpile_2-4a35c7/"


def dclm_baseline_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """DCLM-baseline as a tokenized ``Dataset`` handle."""
    return tokenized(
        "dclm_baseline-llama3",
        tokenizer=tokenizer,
        version="2026.06.28",
        paths=[_DCLM_BASELINE_RAW],
        pin=_DCLM_BASELINE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )


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


def proofpile_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """Proof-Pile 2 as a tokenized ``Dataset`` handle."""
    return tokenized(
        "proofpile_2-llama3",
        tokenizer=tokenizer,
        version="2026.06.28",
        paths=[_PROOFPILE_RAW],
        pin=_PROOFPILE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )
