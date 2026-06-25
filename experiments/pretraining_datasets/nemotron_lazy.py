# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron CC splits as lazy ``Dataset`` handles.

The llama3-tokenized splits are pinned to their existing locations
(``NEMOTRON_LLAMA3_OVERRIDES``) so referencing them never re-tokenizes the
multi-TiB corpus. This is the handle version of ``tokenize_nemotron`` /
``nemotron_mix`` in ``nemotron.py``.
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import tokenize

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_SPLIT_TOKENIZE_RESOURCES,
    NEMOTRON_WEIGHTS,
    _get_nemotron_split_paths,
)


def nemotron_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Dataset]:
    """One :class:`Dataset` handle per Nemotron CC split, keyed by ``nemotron_cc/<split>``."""
    datasets: dict[str, Dataset] = {}
    for split in NEMOTRON_DATASETS:
        train_paths = [str(path.name) for path in _get_nemotron_split_paths(split)]
        pinned = NEMOTRON_LLAMA3_OVERRIDES.get(split) if tokenizer == llama3_tokenizer else None
        datasets[f"nemotron_cc/{split}"] = tokenize(
            f"tokenized/nemotron_cc/{split}",
            "llama3",
            train_paths=train_paths,
            tokenizer=tokenizer,
            resources=NEMOTRON_SPLIT_TOKENIZE_RESOURCES,
            pinned_path=pinned,
        )
    return datasets


def nemotron_components(*, tokenizer: str = llama3_tokenizer) -> dict[Dataset, float]:
    """The Nemotron splits with their mixture weights, ready for ``mixture(...)``."""
    datasets = nemotron_datasets(tokenizer=tokenizer)
    return {datasets[f"nemotron_cc/{split}"]: NEMOTRON_WEIGHTS[f"nemotron_cc/{split}"] for split in NEMOTRON_DATASETS}
