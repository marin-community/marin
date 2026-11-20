# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskforge.ledger.records import LedgerEntry


class ListLedger:
    """In-memory ledger that keeps every recorded entry in order."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
