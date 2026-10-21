# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import sys

import pytest

from experiments.post_training.sft_shootout_rows import build
from experiments.post_training.sft_shootout_rows.test_recovery import tmax_trace, tokenizer  # noqa: F401
from experiments.post_training.sweep_data.prepare_agentic import BATCH, PAD_TOKEN
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT


def line(ids: list[int], mask: list[int], width: int) -> str:
    return json.dumps(training_row(ids, mask, width, PAD_TOKEN), separators=(",", ":"))


def test_a_widened_replay_row_is_the_row_training_row_gives_at_the_wider_cap():
    ids, mask = [5, 6, 7, 8], [0, 1, 1, 0]
    assert build.widen_row(line(ids, mask, 8), 12) == line(ids, mask, 12)
    assert build.widen_row(line(ids, mask, 8), 8) == line(ids, mask, 8)
    with pytest.raises(ValueError, match="Cannot widen"):
        build.widen_row(line(ids, mask, 8), 6)


def test_split_rows_and_padding_rows_take_the_row_cap(tmp_path):
    entries = [
        (line([1, 2, 3], [0, 1, 1], 12), {"family": "tmax", "supervised_tokens": 2, "context_tokens": 3}),
        (line([4, 5], [0, 1], 12), {"family": "tmax", "supervised_tokens": 1, "context_tokens": 2}),
    ]
    path = tmp_path / "split.jsonl.gz"
    index, split = build.write_split(path, entries, "seed", 12)
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(text) for text in stream]
    assert len(rows) == BATCH and split["padding_rows"] == BATCH - 2 and len(index) == 2
    assert {len(row["input_ids"]) for row in rows} == {len(row["loss_weights"]) for row in rows} == {12}
    with pytest.raises(ValueError, match="Bad row shape"):
        build.write_split(tmp_path / "other.jsonl.gz", entries, "seed", 16)


def test_the_row_cap_moves_the_over_context_drop(tokenizer):  # noqa: F811
    trace = tmax_trace() | {"task_id": "task", "relay": False, "problems": [], "passed": True, "sample": 0, "retry": 0}
    pool = {"task": {"heldout": False, "family": "tmax", "import_names": []}}
    length = len(build.render_trace(tokenizer, trace["messages"], trace["tools"]).ids)
    kept, dropped = build.candidates("plain", [trace], pool, tokenizer, length - 1)
    assert not kept and [d["reason"] for d in dropped] == ["over_context"]
    kept, dropped = build.candidates("plain", [trace], pool, tokenizer, length)
    assert list(kept) == ["tmax-trial"] and not dropped
    row = json.loads(build.entry(kept["tmax-trial"], length + 5)[0])
    assert len(row["input_ids"]) == length + 5 and row["input_ids"][length:] == [PAD_TOKEN] * 5


def test_a_row_cap_below_the_replay_width_is_refused(monkeypatch):
    argv = ["build", "--arm", "plain", "--traces", "x.jsonl", "--pool", "p", "--pool-manifest", "m"]
    argv += ["--replay", "r", "--output", "o", "--test-inputs"]
    monkeypatch.setattr(sys, "argv", [*argv, "--row-cap", str(CONTEXT - 1)])
    with pytest.raises(ValueError, match="at least"):
        build.parse_args()
    monkeypatch.setattr(sys, "argv", argv)
    assert build.parse_args().row_cap == CONTEXT
