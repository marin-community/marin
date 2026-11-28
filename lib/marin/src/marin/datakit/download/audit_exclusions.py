# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned row exclusion lists from the Marin SFT data audit dataset."""

from collections.abc import Sequence

import pyarrow.parquet as pq
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

AUDIT_DATASET_ID = "Min-Li/marin-sft-data-audit"


def audit_exclusion_step(name: str, *, revision: str, exclusion_files: Sequence[str]) -> StepSpec:
    """Create a pinned download of exclusion lists from the audit dataset."""
    return download_hf_step(name, hf_dataset_id=AUDIT_DATASET_ID, revision=revision, hf_urls_glob=list(exclusion_files))


def load_excluded_ids(exclusion_path: str, *, exclusion_files: Sequence[str], id_column: str) -> frozenset[str]:
    """Load the row IDs that are on any of the downloaded exclusion lists."""
    excluded_ids: set[str] = set()
    for exclusion_file in exclusion_files:
        with open_url(prefix_join(exclusion_path, exclusion_file), "rb") as source:
            excluded_ids.update(pq.read_table(source, columns=[id_column]).column(id_column).to_pylist())
    return frozenset(excluded_ids)
