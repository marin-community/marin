# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified private combined-retention pins and their archived input lineage."""

from pathlib import Path

from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.remediation_binding import RetainedReleaseReceipt

RETAINED_RELEASE_RECEIPTS = read_receipt_shards(
    Path(__file__).with_name("retained_release_receipts"), RetainedReleaseReceipt, "retained release"
)
