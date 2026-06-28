# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOLMINO dataset as lazy ``Dataset`` handles (the flat dataset catalog).

One handle per split, tokenizing each split's glob from the fixed raw path. The
llama3-tokenized splits are pinned to their existing caches so referencing them
never re-tokenizes the multi-TiB corpus. This is the catalog only — handles, no
weights; the mixture weights are policy and live in the experiment that chooses
them.
"""

from marin.datakit.download.dolmino import DOLMINO_DATASETS
from marin.execution.lazy import Dataset
from marin.experiment.data import tokenized

from experiments.llama import llama3_tokenizer

# Dolmino revision bb54cab downloaded to this fixed path (allenai/dolmino-mix-1124).
_DOLMINO_BASE = "raw/dolmino-mix-1124-157960/bb54cab/data"

# NB: hashes predate a hashing change; never recompute these caches.
DOLMINO_LLAMA3_OVERRIDES = {
    "dclm": "tokenized/dolmino/dclm-6c18eb",
    "flan": "tokenized/dolmino/flan-d71ec1",
    "math/codesearchnet-owmfilter": "tokenized/dolmino/math/codesearchnet-owmfilter-fd2640",
    "math/dolmino_math_synth": "tokenized/dolmino/math/dolmino_math_synth-11f876",
    "math/gsm8k": "tokenized/dolmino/math/gsm8k-902e8b",
    "math/mathcoder2-synthmath": "tokenized/dolmino/math/mathcoder2-synthmath-bc8dd2",
    "math/metamath-owmfilter": "tokenized/dolmino/math/metamath-owmfilter-fafa84",
    "math/tinyGSM-MIND": "tokenized/dolmino/math/tinyGSM-MIND-6c3016",
    "math/tulu_math": "tokenized/dolmino/math/tulu_math-414a4d",
    "pes2o": "tokenized/dolmino/pes2o-d22243",
    "stackexchange": "tokenized/dolmino/stackexchange-271a84",
    "wiki": "tokenized/dolmino/wiki-c31b74",
    "dolmino_dclm": "tokenized/dolmino/dclm-6c18eb",
}


def tokenize_dolmino(*, tokenizer: str = llama3_tokenizer) -> dict[str, Dataset]:
    """One :class:`Dataset` handle per Dolmino split, keyed by ``dolmino/<split>``."""
    return {
        f"dolmino/{split}": tokenized(
            f"dolmino/{split}",
            tokenizer=tokenizer,
            paths=[f"{_DOLMINO_BASE}/{split}/{pattern}" for pattern in files],
            pin=DOLMINO_LLAMA3_OVERRIDES.get(split) if tokenizer == llama3_tokenizer else None,
        )
        for split, files in DOLMINO_DATASETS.items()
    }


def tokenize_dolmino_subset(name: str, tokenizer: str = llama3_tokenizer) -> Dataset:
    """The :class:`Dataset` handle for a single named Dolmino split."""
    assert name in DOLMINO_DATASETS, f"Split {name} not found in DOLMINO_DATASETS"
    return tokenize_dolmino(tokenizer=tokenizer)[f"dolmino/{name}"]


def tokenize_dolmino_math(tokenizer: str = llama3_tokenizer) -> Dataset:
    """Combined math-only :class:`Dataset` handle (all ``math/*`` splits merged)."""
    math_paths = [
        f"{_DOLMINO_BASE}/{split}/{pattern}"
        for split, files in DOLMINO_DATASETS.items()
        if "math" in split
        for pattern in files
    ]
    return tokenized(
        "dolmino/all_math",
        tokenizer=tokenizer,
        paths=math_paths,
        pin="tokenized/dolmino/all_math-9d507c" if tokenizer == llama3_tokenizer else None,
    )
