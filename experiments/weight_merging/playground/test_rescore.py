# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.weight_merging.playground.rescore import score_rows


def test_deferred_scoring_preserves_gsm_and_input_records(monkeypatch):
    rows = [
        {
            "candidate": "merge",
            "item_id": "math",
            "dataset": "gsm8k",
            "mode": "greedy",
            "seed": 0,
            "completion": "#### 42",
            "correct": True,
        },
        {
            "candidate": "merge",
            "item_id": "code",
            "dataset": "mbpp",
            "mode": "greedy",
            "seed": 0,
            "completion": "def f(): return 42",
            "correct": None,
            "pending_execution": True,
        },
    ]
    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", lambda item, text: {"correct": True})
    scored, differences = score_rows(rows, {"code": {}}, parity=False)
    assert scored[0] == rows[0]
    assert scored[1]["correct"] is True
    assert scored[1]["pending_execution"] is False
    assert rows[1]["correct"] is None
    assert differences == []


def test_parity_records_any_changed_execution_outcome(monkeypatch):
    row = {
        "candidate": "original",
        "item_id": "code",
        "dataset": "mbpp",
        "mode": "greedy",
        "seed": 0,
        "completion": "def f(): return 42",
        "correct": True,
    }
    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", lambda item, text: {"correct": False})
    _, differences = score_rows([row], {"code": {}}, parity=True)
    assert len(differences) == 1
    assert differences[0]["before"] is True and differences[0]["after"] is False
