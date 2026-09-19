import json

import pytest

from experiments.post_training.analyze_score_centering import summarize_run, verify_membership


def test_saved_eval_uses_completed_correct_answers_and_checks_membership(tmp_path):
    session = tmp_path / "dumped_evals" / "global_step_0_evals"
    session.mkdir(parents=True)
    rows = [
        {"input_prompt": "a", "env_extras": {"reward_spec": {"ground_truth": "1"}}, "score": 1, "stop_reason": "length"},
        {"input_prompt": "b", "env_extras": {"reward_spec": {"ground_truth": "2"}}, "score": -1, "stop_reason": "stop"},
        {
            "input_prompt": "c",
            "env_extras": {"reward_spec": {"ground_truth": "3"}},
            "score": 1,
            "stop_reason": "end_turn",
        },
    ]
    (session / "val-math500.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (session / "aggregated_results.jsonl").write_text(
        json.dumps({"eval/val-math500/response_tokens_mean": 12, "eval/all/response_tokens_mean": 12}) + "\n"
    )

    result = summarize_run("arm", str(tmp_path), "https://unused.example")
    overall = next(row for row in result if row["dataset"] == "all")
    assert overall["completed_correct"] == 1
    assert overall["completed_correct_rate"] == pytest.approx(1 / 3)
    assert overall["correct_any_stop"] == 2
    assert overall["raw_reward_mean"] == pytest.approx(1 / 3)
    assert overall["length_stop_fraction"] == pytest.approx(1 / 3)
    assert verify_membership(result) == overall["membership_sha256"]

    changed = [dict(row) for row in result]
    next(row for row in changed if row["dataset"] == "all")["membership_sha256"] = "different"
    with pytest.raises(ValueError, match="membership differs"):
        verify_membership(result + changed)
