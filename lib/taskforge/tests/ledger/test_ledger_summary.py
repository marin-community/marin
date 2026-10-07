# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.ledger.records import EntryKind, LedgerEntry

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ledger_summary.py"


def test_summary_totals_wall_time_and_tokens_per_kind_and_step(tmp_path):
    ledger = JsonlLedger(tmp_path)
    for item, start in (("a", 0.0), ("b", 1.0)):
        ledger.record(
            LedgerEntry(
                item_id=item,
                round=0,
                step="author",
                kind=EntryKind.LLM_CALL,
                started=start,
                ended=start + 4.0,
                tokens_in=100,
                tokens_out=10,
                tokens_reasoning=5,
            )
        )
    ledger.record(
        LedgerEntry(item_id="a", round=0, step="run", kind=EntryKind.STEP, started=4.0, ended=6.0, cause="Exit1")
    )

    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--format", "json"], check=True, capture_output=True, text=True
    ).stdout
    summary = json.loads(out)

    llm = summary["by_kind"]["llm_call"]
    assert (llm["count"], llm["busy"], llm["elapsed"]) == (2, 8.0, 5.0)
    assert (llm["tokens_in"], llm["tokens_out"], llm["tokens_reasoning"]) == (200, 20, 10)
    step = summary["by_step"]["step/run"]
    assert (step["count"], step["failed"], step["busy"]) == (1, 1, 2.0)
