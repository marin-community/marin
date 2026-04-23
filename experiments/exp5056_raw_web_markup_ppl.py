# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Registry helpers for raw web, markup, and image-text perplexity-gap slices.
"""

import os
from collections.abc import Mapping

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"

ACTIVE_RAW_WEB_MARKUP_DATASETS: dict[str, RawTextEvaluationDataset] = {}


def prefixed_raw_web_markup_validation_sets(
    datasets: Mapping[str, RawTextEvaluationDataset],
) -> dict[str, RawTextEvaluationDataset]:
    """Prefix raw-web-markup slice names with ``raw_web_markup/``."""
    return {os.path.join(RAW_WEB_MARKUP_PREFIX, slice_name): dataset for slice_name, dataset in datasets.items()}


def raw_web_markup_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw web/markup evaluation slices keyed by ``raw_web_markup/<slice>``."""
    return prefixed_raw_web_markup_validation_sets(ACTIVE_RAW_WEB_MARKUP_DATASETS)
