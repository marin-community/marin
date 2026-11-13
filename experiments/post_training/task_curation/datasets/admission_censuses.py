# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-population current-pin runnable-grader censuses for Atlas admission."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.admission_binding import AdmissionCensusReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def admission_census_receipts() -> dict[str, AdmissionCensusReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("admission_census_receipts"),
        AdmissionCensusReceipt,
        "admission census",
        load_json=read_registry_json,
    )
