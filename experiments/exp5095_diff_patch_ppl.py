# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Registry helpers for diff and patch perplexity-gap slices.

Slices cover agent-facing surface forms (raw ``git diff``, commit-message-plus-diff,
PR-review-plus-diff, issue-to-patch). Source builders are intentionally deferred;
this module only establishes the ``diff_patch/<slice>`` namespace and the active
registry that downstream experiments read.
"""

import os
from collections.abc import Mapping

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset

DIFF_PATCH_PREFIX = "diff_patch"

ACTIVE_DIFF_PATCH_DATASETS: dict[str, RawTextEvaluationDataset] = {}


def prefixed_diff_patch_validation_sets(
    datasets: Mapping[str, RawTextEvaluationDataset],
) -> dict[str, RawTextEvaluationDataset]:
    """Prefix diff/patch slice names with ``diff_patch/``."""
    return {os.path.join(DIFF_PATCH_PREFIX, slice_name): dataset for slice_name, dataset in datasets.items()}


def diff_patch_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Diff/patch evaluation slices keyed by ``diff_patch/<slice>``."""
    return prefixed_diff_patch_validation_sets(ACTIVE_DIFF_PATCH_DATASETS)
