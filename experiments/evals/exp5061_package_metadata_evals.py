# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5061 package-metadata raw-text eval slices.

This first pass keeps scope deliberately small: materialize a bounded public npm
registry sample so PPL-gap runs can score package/dependency metadata without
depending on BigQuery or bulk mirrors.

Issue: https://github.com/marin-community/marin/issues/5061 (parent #5005).
"""

from __future__ import annotations

import posixpath

from marin.datakit.download.npm_registry_metadata import (
    NPM_REGISTRY_SLICE_KEY,
    NpmRegistryMetadataSource,
    npm_registry_metadata_step,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep

ISSUE_5061 = 5061
EPIC_5005 = 5005

NPM_REGISTRY_SOURCE = NpmRegistryMetadataSource(slice_key=NPM_REGISTRY_SLICE_KEY)
NPM_REGISTRY_RAW = npm_registry_metadata_step(NPM_REGISTRY_SOURCE)


def package_metadata_raw_validation_sets(
    *,
    raw_root: str | None = None,
    package_metadata_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return the first-pass package-metadata raw validation slices."""

    if raw_root is None and package_metadata_raw is None:
        package_metadata_raw = NPM_REGISTRY_RAW

    if raw_root is not None:
        source: str | ExecutorStep = posixpath.join(raw_root, "packages/npm/registry.jsonl.gz")
    else:
        assert package_metadata_raw is not None
        source = package_metadata_raw.cd("data.jsonl.gz")

    return {
        NPM_REGISTRY_SLICE_KEY: raw_text_dataset(
            source,
            tags=("package_metadata", f"epic:{EPIC_5005}", f"issue:{ISSUE_5061}", NPM_REGISTRY_SLICE_KEY),
        )
    }
