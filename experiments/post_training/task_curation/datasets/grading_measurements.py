# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete grading censuses for the configured frozen source populations."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def grading_census_receipts() -> dict[str, GradingCensusReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("grading_census_receipts"),
        GradingCensusReceipt,
        "grading census",
        load_json=read_registry_json,
    )
