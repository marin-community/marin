# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The DCLM 1B/1x training mixture as lazy ``Dataset`` handles.

The catalog supplies inert handles (mechanism); the weights are the corpus's token
proportions (policy). An experiment assembles them with ``mixture()``.
"""

from collections.abc import Mapping

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.llama import llama3_tokenizer

# DCLM 1B/1x mixture proportions (token counts measured with the neox tokenizer):
# dclm-baseline ~3.8T, starcoderdata ~250B, proof-pile-2 ~55B.
DCLM_MIXTURE_WEIGHTS: Mapping[str, float] = {
    "dclm_baseline": 3.8,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}

# Raw download location.
_DCLM_BASELINE_RAW = "raw/dclm/a3b142c"

# Pinned llama3-tokenized cache; never re-tokenize this corpus (~3.8T tokens).
_DCLM_BASELINE_LLAMA3_PIN = "tokenized/dclm_baseline-0206f1/"


def dclm_baseline_dataset(*, tokenizer: str = llama3_tokenizer) -> ArtifactStep[TokenizedCache]:
    """DCLM-baseline as a tokenized ``Dataset`` handle."""
    return tokenized(
        "dclm_baseline-llama3",
        tokenizer=tokenizer,
        version="2026.06.28",
        paths=[_DCLM_BASELINE_RAW],
        pin=_DCLM_BASELINE_LLAMA3_PIN if tokenizer == llama3_tokenizer else None,
    )


def dclm_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """The DCLM training components as lazy handles, keyed to match ``DCLM_MIXTURE_WEIGHTS``."""
    return {
        "dclm_baseline": dclm_baseline_dataset(tokenizer=tokenizer),
        "starcoderdata": starcoder_dataset(tokenizer=tokenizer),
        "proofpile_2": proofpile_dataset(tokenizer=tokenizer),
    }


if __name__ == "__main__":
    dataset_main(dclm_datasets())
