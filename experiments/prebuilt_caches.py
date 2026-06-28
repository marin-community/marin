# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Prebuilt Levanter cache subcaches hosted on HuggingFace, ready for direct use.

Two fineweb-edu subcaches are available:

1. A 10B token subcache — a subset of the original fineweb-edu dataset.
2. A 10M token subcache — a smaller subset for testing.

Example usage::

    from experiments.prebuilt_caches import fineweb_edu_10B_dataset
    from marin.experiment.data import mixture

    data = lambda ctx: mixture(ctx, {fineweb_edu_10B_dataset(): 1.0})
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import pretokenized

from experiments.marin_tokenizer import marin_tokenizer

fineweb_edu_10B_repo_id = "marin-community/fineweb-edu-pretokenized-10B"
fineweb_edu_10M_repo_id = "marin-community/fineweb-edu-pretokenized-10M"


def fineweb_edu_10B_dataset() -> Dataset:
    """Pretokenized fineweb-edu 10B-token subcache as a lazy Dataset handle."""
    return pretokenized(
        "fineweb-edu-10B",
        repo_id=fineweb_edu_10B_repo_id,
        tokenizer=marin_tokenizer,
    )


def fineweb_edu_10M_dataset() -> Dataset:
    """Pretokenized fineweb-edu 10M-token subcache as a lazy Dataset handle."""
    return pretokenized(
        "fineweb-edu-10M",
        repo_id=fineweb_edu_10M_repo_id,
        tokenizer=marin_tokenizer,
    )
