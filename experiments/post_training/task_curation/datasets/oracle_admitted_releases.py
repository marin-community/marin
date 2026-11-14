# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle-filtered pins with current population qualification."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.oracle_admitted_release import OracleAdmittedReleaseReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def oracle_admitted_release_receipts() -> dict[str, OracleAdmittedReleaseReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("oracle_admitted_release_receipts"),
        OracleAdmittedReleaseReceipt,
        "admitted oracle release",
        load_json=read_registry_json,
    )
