# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The DCLM 1B/1x training mixture as lazy ``Dataset`` handles.

The catalog supplies inert handles (mechanism); the weights are the corpus's token
proportions (policy). An experiment assembles them with ``mixture()``.
"""

from collections.abc import Mapping

from marin.execution.lazy import Dataset

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple_lazy import dclm_baseline_dataset, proofpile_dataset, starcoder_dataset

# DCLM 1B/1x mixture proportions (token counts measured with the neox tokenizer):
# dclm-baseline ~3.8T, starcoderdata ~250B, proof-pile-2 ~55B.
DCLM_MIXTURE_WEIGHTS: Mapping[str, float] = {
    "dclm_baseline": 3.8,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}


def dclm_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Dataset]:
    """The DCLM training components as lazy handles, keyed to match ``DCLM_MIXTURE_WEIGHTS``."""
    return {
        "dclm_baseline": dclm_baseline_dataset(tokenizer=tokenizer),
        "starcoderdata": starcoder_dataset(tokenizer=tokenizer),
        "proofpile_2": proofpile_dataset(tokenizer=tokenizer),
    }
