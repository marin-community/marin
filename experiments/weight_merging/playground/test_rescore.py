# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.weight_merging.playground.remote.grades import GradeStore
from experiments.weight_merging.playground.rescore import check_cleanup_cache, score_rows


def test_deferred_scoring_preserves_gsm_and_input_records(monkeypatch, tmp_path):
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
    scored, differences = score_rows(
        rows, {"code": {}}, cache=GradeStore(str(tmp_path), {"scorer": "test"}), parity=False
    )
    assert scored[0] == rows[0]
    assert scored[1]["correct"] is True
    assert scored[1]["pending_execution"] is False
    assert rows[1]["correct"] is None
    assert differences == []


def test_parity_records_any_changed_execution_outcome(monkeypatch, tmp_path):
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
    _, differences = score_rows([row], {"code": {}}, cache=GradeStore(str(tmp_path), {"scorer": "test"}), parity=True)
    assert len(differences) == 1
    assert differences[0]["before"] is True and differences[0]["after"] is False


def test_completed_grades_survive_another_item_failing_and_retry(monkeypatch, tmp_path):
    cache = GradeStore(str(tmp_path), {"scorer": "fixed"})
    rows = [
        {
            "candidate": "merge",
            "item_id": item,
            "dataset": "mbpp",
            "mode": "greedy",
            "seed": 0,
            "completion": item,
            "correct": None,
            "pending_execution": True,
        }
        for item in ("good", "failure")
    ]

    def first(item, text):
        if text == "failure":
            raise RuntimeError("daemon unavailable")
        return {"correct": True}

    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", first)
    with pytest.raises(RuntimeError, match="daemon unavailable"):
        score_rows(rows, {"good": {}, "failure": {}}, cache=cache)
    assert cache.get(cache.identity(rows[0], {})) == {"correct": True}
    executed = []

    def retry(item, text):
        executed.append(text)
        return {"correct": False}

    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", retry)
    timings = []
    result, _ = score_rows(rows, {"good": {}, "failure": {}}, cache=cache, execution_timings=timings)
    assert executed == ["failure"]
    assert [row["correct"] for row in result] == [True, False]
    measured = {row["key"][1]: row for row in timings}
    assert measured["good"]["reused"] and not measured["failure"]["reused"]
    assert measured["good"]["elapsed_seconds"] == cache.get_record(cache.identity(rows[0], {}))["elapsed_seconds"]
    changed = {**rows[0], "completion": "changed"}
    assert cache.get(cache.identity(changed, {})) is None
    assert cache.get(cache.identity(rows[0], {"tests": ["changed test"]})) is None


def test_cleanup_cache_audit_checks_saved_grades_and_synthetic_timeout(monkeypatch, tmp_path):
    cache = GradeStore(str(tmp_path), {"scorer": "known-old"})
    rows = [
        {"candidate": "model", "dataset": "mbpp", "item_id": str(i), "mode": "greedy", "seed": 0, "completion": str(i)}
        for i in range(6)
    ]
    grades = [{"correct": True, "error": "exit-0"}] * 5 + [{"correct": False, "error": "timeout"}]
    for row, grade in zip(rows[:5], grades[:5], strict=True):
        cache.put(cache.identity(row, {}), grade)
    executed = []

    def reexecute(item, text):
        executed.append(text)
        return grades[5] if text.startswith("while True:") else grades[int(text)]

    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", reexecute)
    checks = check_cleanup_cache(rows, {row["item_id"]: {} for row in rows}, cache)
    assert executed == ["0", "1", "2", "3", "4", "while True:\n    pass\n"]
    assert len(checks) == 6 and checks[-1]["grade"]["error"] == "timeout"
    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", lambda *args: grades[0])
    with pytest.raises(ValueError, match="did not time out"):
        check_cleanup_cache(rows, {row["item_id"]: {} for row in rows}, cache)
    monkeypatch.setattr("experiments.weight_merging.playground.rescore.score", lambda *args: {"correct": False})
    with pytest.raises(ValueError, match="changed cached grade"):
        check_cleanup_cache(rows, {row["item_id"]: {} for row in rows}, cache)
