# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified oracle-filtered private release pins and original population evidence."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.oracle_release import OracleFilteredReleaseReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def oracle_release_receipts() -> dict[str, OracleFilteredReleaseReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("oracle_release_receipts"),
        OracleFilteredReleaseReceipt,
        "oracle release",
        load_json=read_registry_json,
    )
