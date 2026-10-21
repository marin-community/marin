# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.post_training.sft_shootout_rows.build import exclude_trials


def trace(name: str, retry: int, conversation: int = 0, task: str = "swesmith-00001") -> dict:
    return {
        "trial_id": f"{name}-{retry}",
        "trial_name": name,
        "retry": retry,
        "conversation": conversation,
        "task_id": task,
    }


TRACES = [trace("a__x", 0), trace("a__x", 1), trace("b__y", 0), trace("b__y", 0, conversation=1), trace("c__z", 0)]


def write(tmp_path, entries: list[dict]) -> str:
    path = tmp_path / "exclude.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries) + "\n")
    return str(path)


def test_listed_attempts_are_dropped_and_recorded(tmp_path):
    entries = [
        {"trial_name": "a__x", "retry": 1, "task_id": "swesmith-00001", "reason": "ENOSPC"},
        {"trial_name": "b__y"},  # every attempt, every conversation
        {"trial_name": "gone__q", "retry": 0},  # in another run type: recorded, removes nothing
    ]
    uri = write(tmp_path, entries)
    kept, record = exclude_trials(TRACES, uri)
    assert [(t["trial_name"], t["retry"]) for t in kept] == [("a__x", 0), ("c__z", 0)]
    assert record["sha256"] == hashlib.sha256((tmp_path / "exclude.jsonl").read_bytes()).hexdigest()
    assert [e["trace_rows_removed"] for e in record["entries"]] == [1, 2, 0]
    assert record["entries"][0]["reason"] == "ENOSPC"


def test_an_entry_naming_the_wrong_task_is_refused(tmp_path):
    uri = write(tmp_path, [{"trial_name": "c__z", "task_id": "swesmith-09999"}])
    with pytest.raises(ValueError, match="not swesmith-09999"):
        exclude_trials(TRACES, uri)
