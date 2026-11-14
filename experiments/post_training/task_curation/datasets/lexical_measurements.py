# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete frozen-population lexical measurements keyed by current catalog ID."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def lexical_measurements() -> dict[str, LexicalMeasurementReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("lexical_measurement_receipts"),
        LexicalMeasurementReceipt,
        "lexical",
        load_json=read_registry_json,
    )
