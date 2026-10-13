# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, LedgerEntry

APPENDS_PER_WRITER = 200
WRITERS = 4


def make_entry(item_id: str, step: str, **kw) -> LedgerEntry:
    return LedgerEntry(item_id=item_id, round=0, step=step, kind=EntryKind.STEP, started=1.0, ended=2.0, **kw)


def test_round_trip_keeps_every_field_one_file_per_item(tmp_path):
    ledger = JsonlLedger(tmp_path)
    full = LedgerEntry(
        item_id="a",
        round=3,
        step="validate",
        kind=EntryKind.LLM_CALL,
        started=1.25,
        ended=4.5,
        model="glm-5.3",
        tokens_in=1,
        tokens_out=2,
        tokens_reasoning=3,
        finish_reason="length",
        code_hash="c",
        input_hash="i",
        output_hash="o",
        cause="RateLimited",
        attrs={"k": "v"},
    )
    ledger.record(full)
    ledger.record(make_entry("a", "second"))
    ledger.record(make_entry("b", "other"))

    assert list(read_entries(ledger.path_for("a"))) == [full, make_entry("a", "second")]
    assert list(read_entries(ledger.path_for("b"))) == [make_entry("b", "other")]


def test_reader_skips_torn_trailing_line(tmp_path):
    ledger = JsonlLedger(tmp_path)
    ledger.record(make_entry("a", "done"))
    with ledger.path_for("a").open("ab") as f:
        f.write(b'{"item_id": "a", "rou')

    assert [e.step for e in read_entries(ledger.path_for("a"))] == ["done"]


def test_append_after_a_torn_line_drops_the_fragment(tmp_path):
    ledger = JsonlLedger(tmp_path)
    ledger.record(make_entry("a", "done"))
    with ledger.path_for("a").open("ab") as f:
        f.write(b'{"item_id": "a", "rou')
    ledger.record(make_entry("a", "resumed"))

    assert [e.step for e in read_entries(ledger.path_for("a"))] == ["done", "resumed"]


def test_append_after_a_torn_first_line_starts_the_file_over(tmp_path):
    ledger = JsonlLedger(tmp_path)
    tmp_path.joinpath("a.jsonl").write_bytes(b'{"item_id": "a", "rou')
    ledger.record(make_entry("a", "resumed"))

    assert [e.step for e in read_entries(ledger.path_for("a"))] == ["resumed"]


WRITER_PROGRAM = """
import sys
from pathlib import Path
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.ledger.records import EntryKind, LedgerEntry
root, writer, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ledger = JsonlLedger(root)
for i in range(count):
    ledger.record(LedgerEntry(item_id="shared", round=0, step=f"w{writer}-{i}", kind=EntryKind.STEP,
                              started=1.0, ended=2.0, attrs={"pad": "x" * 2000}))
"""


def test_concurrent_processes_never_interleave_lines(tmp_path):
    procs = [
        subprocess.Popen([sys.executable, "-c", WRITER_PROGRAM, str(tmp_path), str(w), str(APPENDS_PER_WRITER)])
        for w in range(WRITERS)
    ]
    assert [p.wait() for p in procs] == [0] * WRITERS

    steps = [e.step for e in read_entries(JsonlLedger(tmp_path).path_for("shared"))]
    assert sorted(steps) == sorted(f"w{w}-{i}" for w in range(WRITERS) for i in range(APPENDS_PER_WRITER))
