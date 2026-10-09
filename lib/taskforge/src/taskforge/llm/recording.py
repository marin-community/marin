# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of model calls: where a call is recorded."""

from dataclasses import dataclass

from taskforge.ledger.records import Ledger


@dataclass(frozen=True)
class CallLedger:
    """Where GLM calls and tool calls are recorded, all under one item, round and step."""

    ledger: Ledger
    item_id: str
    round: int
    step: str
