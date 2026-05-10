# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Long-context datasets for post-training experiments.

Dataset families are organized in separate modules:
- finepdfs: FinePDFs and FinePDFs-edu (English, path-based dicts for mixtures)
- longmino: Dolma3 Longmino pool with length-bucket breakdowns

Institutional Books 1.0 lives in ``marin.datakit.download.institutional_books``
as the canonical single-source declaration; it's registered in
``marin.datakit.sources`` and consumed from there.
"""

from experiments.long_context_datasets.finepdfs import (
    finepdfs_by_language,
    finepdfs_edu_by_language,
    finepdfs_edu_token_counts,
    finepdfs_token_counts,
    finepdfs_validation_by_language,
)
from experiments.long_context_datasets.longmino import (
    longmino_bucket_token_counts,
    longmino_by_bucket,
)

__all__ = [
    "finepdfs_by_language",
    "finepdfs_edu_by_language",
    "finepdfs_edu_token_counts",
    "finepdfs_token_counts",
    "finepdfs_validation_by_language",
    "longmino_bucket_token_counts",
    "longmino_by_bucket",
]
