# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete derived contract censuses for the configured frozen source populations."""

from functools import cache
from pathlib import Path

from experiments.post_training.task_curation.derived_contract_tags import DerivedContractReceipt
from experiments.post_training.task_curation.receipt_shards import read_receipt_shards
from experiments.post_training.task_curation.registry import read_registry_json


@cache
def derived_contract_receipts() -> dict[str, DerivedContractReceipt]:
    return read_receipt_shards(
        Path(__file__).with_name("derived_contract_receipts"),
        DerivedContractReceipt,
        "derived contract census",
        load_json=read_registry_json,
    )
