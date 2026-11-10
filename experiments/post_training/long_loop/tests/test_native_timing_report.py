# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.post_training.long_loop.runtime_timing import write_report
from experiments.post_training.long_loop.timing import EventLog


def test_adopted_native_journal_is_verified_and_reported(tmp_path):
    journal = tmp_path / "native.jsonl"
    log = EventLog(journal, "original-sft", "sft", "rank0")
    with log.span("compile"):
        pass
    eval_root = tmp_path / "eval"
    results = eval_root / "exports/dumped_evals/eval_only"
    results.mkdir(parents=True)
    row = {
        "data_source": "tiny",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1],
        "stop_reason": "stop",
    }
    (results / "results.jsonl").write_text(json.dumps(row))
    output = tmp_path / "report"
    output.mkdir()
    pin = {"source": str(journal), "sha256": hashlib.sha256(journal.read_bytes()).hexdigest()}
    write_report(
        output=str(output),
        stages=[str(eval_root)],
        expected_rollouts=1,
        sft_geometry="CPU fixture",
        additional_timing=[pin],
    )
    assert str(journal) in json.loads((output / "timing-sources.json").read_text())
    assert "original-sft" in (output / "timing.md").read_text()
    assert "compile" in (output / "timing.md").read_text()
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(ValueError, match="Pinned native timing journal changed"):
        write_report(
            output=str(output),
            stages=[str(eval_root)],
            expected_rollouts=1,
            sft_geometry="CPU fixture",
            additional_timing=[pin],
        )
