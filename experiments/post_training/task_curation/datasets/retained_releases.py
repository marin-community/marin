# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified private combined-retention pins and their archived input lineage."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.remediation_binding import RetainedReleaseReceipt


@cache
def retained_release_receipts() -> dict[str, RetainedReleaseReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("retained_release_receipts"),
        RetainedReleaseReceipt,
        "retained release",
        load_json=read_registry_json,
    )
