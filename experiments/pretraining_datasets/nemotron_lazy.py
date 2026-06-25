# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron CC splits as lazy ``Dataset`` handles (the flat dataset catalog).

The llama3-tokenized splits are pinned to their existing locations
(``NEMOTRON_LLAMA3_OVERRIDES``) so referencing them never re-tokenizes the
multi-TiB corpus. This is the catalog only — handles, no weights. The mixture
weights are policy and live in the experiment that chooses them.
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import tokenized

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_SPLIT_TOKENIZE_RESOURCES,
    _get_nemotron_split_paths,
)


def nemotron_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Dataset]:
    """One :class:`Dataset` handle per Nemotron CC split, keyed by split name."""
    datasets: dict[str, Dataset] = {}
    for split in NEMOTRON_DATASETS:
        pinned = NEMOTRON_LLAMA3_OVERRIDES.get(split) if tokenizer == llama3_tokenizer else None
        datasets[split] = tokenized(
            f"nemotron_cc/{split}",
            tokenizer=tokenizer,
            version="llama3",
            paths=[str(path.name) for path in _get_nemotron_split_paths(split)],
            pin=pinned,
            resources=NEMOTRON_SPLIT_TOKENIZE_RESOURCES,
        )
    return datasets
