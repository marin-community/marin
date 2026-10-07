# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only local JSONL ledger, one file per item.

Each entry is written with a single ``write`` on an ``O_APPEND`` descriptor under an exclusive
``flock``, so concurrent writers in any number of threads or processes never interleave within a
line. A crash can leave at most a trailing line without its newline. The reader skips it, and the
next append truncates it first, so the fragment never joins the following entry.
"""

import fcntl
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from taskforge.ledger.records import LedgerEntry, check_item_id, entry_from_json, entry_to_json

logger = logging.getLogger(__name__)

LEDGER_SUFFIX = ".jsonl"
TAIL_CHUNK = 65536


def _complete_end(fd: int, size: int) -> int:
    """The offset just past the last newline in the first ``size`` bytes of ``fd``, or 0."""
    end = size
    while end > 0:
        start = max(0, end - TAIL_CHUNK)
        newline = os.pread(fd, end - start, start).rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        end = start
    return 0


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
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                end = _complete_end(fd, size)
                logger.warning("truncating torn trailing line in %s (%d bytes)", path, size - end)
                os.ftruncate(fd, end)
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
