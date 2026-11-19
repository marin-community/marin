# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current-pin sampled content Results from the private immutable registry."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt


@cache
def sample_qc_receipts() -> dict[str, SampleQCReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("sample_qc_receipts"),
        SampleQCReceipt,
        "sampled content Result",
        load_json=read_registry_json,
    )
