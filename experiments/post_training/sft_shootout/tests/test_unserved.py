# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.sft_shootout.harbor_glm import unserved_trials
from experiments.post_training.sft_shootout.relay_followup import round_counts


def trial(root, name: str, task: str, exception: str | None, reward: float, retry: int = 0) -> None:
    attempt = root / name / "attempts" / f"{retry:03d}"
    (attempt / "agent").mkdir(parents=True)
    (attempt / "agent" / "trial_correlation_id.txt").write_text(f"id-{name}-{retry}")
    result = {
        "task_name": f"tasks/{task}",
        "trial_name": name,
        "started_at": f"2026-10-17T08:0{retry}:00Z",
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": exception} if exception else None,
    }
    (attempt / "result.json").write_text(json.dumps(result))


def make_round(tmp_path):
    # a: served and passed with a GLM turn; b: raised with no reply at all; c: raised after a GLM reply;
    # d: raised on its first attempt, then served on the retry.
    trial(tmp_path, "a", "task-a", None, 1.0)
    trial(tmp_path, "b", "task-b", "NonZeroAgentExitCodeError", 0.0)
    trial(tmp_path, "c", "task-c", "NonZeroAgentExitCodeError", 0.0)
    trial(tmp_path, "d", "task-d", "NonZeroAgentExitCodeError", 0.0)
    trial(tmp_path, "d", "task-d", None, 0.0, retry=1)
    requests = [
        {"trial": "id-a-0", "status": 200},
        {"trial": "id-b-0", "status": 502},
        {"trial": "id-c-0", "status": 200},
        {"trial": "id-d-1", "status": 200},
    ]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    rows = [
        {"trial_id": "id-a-0", "tokens": {"glm": {"turns": 2}}},
        {"trial_id": "id-c-0", "tokens": {"glm": {"turns": 1}}},
        {"trial_id": "id-d-1", "tokens": {"student": {"turns": 3}}},
    ]
    (tmp_path / "rebuilt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_unserved_counts_raised_trials_without_a_reply(tmp_path):
    make_round(tmp_path)
    assert unserved_trials(tmp_path, tmp_path / "requests.jsonl") == {"trials": 4, "unserved": 1, "share": 0.25}


def test_raised_trials_without_glm_turns_are_not_runs(tmp_path):
    make_round(tmp_path)
    episodes, runs = round_counts(str(tmp_path))
    assert episodes == {"task-a": 1}
    assert runs == {"task-a": 1, "task-c": 1, "task-d": 1}
