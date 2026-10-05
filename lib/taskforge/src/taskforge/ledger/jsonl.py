# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only local JSONL ledger, one file per item.

Each entry is written with a single ``write`` on an ``O_APPEND`` descriptor, so concurrent writers
in any number of threads or processes never interleave within a line. A crash can leave at most a
trailing line without its newline; the reader skips it.
"""

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from taskforge.ledger.records import LedgerEntry, check_item_id, entry_from_json, entry_to_json

logger = logging.getLogger(__name__)

LEDGER_SUFFIX = ".jsonl"


class JsonlLedger:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, item_id: str) -> Path:
        check_item_id(item_id)
        return self.root / f"{item_id}{LEDGER_SUFFIX}"

    def record(self, entry: LedgerEntry) -> None:
        path = self.path_for(entry.item_id)
        line = (json.dumps(entry_to_json(entry), sort_keys=True) + "\n").encode()
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            written = os.write(fd, line)
        finally:
            os.close(fd)
        if written != len(line):
            raise OSError(f"short append to {path}: wrote {written} of {len(line)} bytes")


def read_entries(path: Path) -> Iterator[LedgerEntry]:
    """Yield the complete entries in one item's ledger file, in append order."""
    with path.open("rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                logger.warning("skipping torn trailing line in %s (%d bytes)", path, len(raw))
                return
            yield entry_from_json(json.loads(raw))


def ledger_files(root: Path) -> list[Path]:
    return sorted(root.glob(f"*{LEDGER_SUFFIX}"))
