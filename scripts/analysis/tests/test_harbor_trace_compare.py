# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from scripts.analysis.harbor_trace_compare import Trace, compare, config_mismatches, load_local, timeout_counterfactual


def _write_trial(root, name, *, reward=None, error=None, request_times=()):
    trial = root / name
    (trial / "agent").mkdir(parents=True)
    result = {
        "task_name": name.rsplit("__", 1)[0],
        "trial_name": name,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:10:00Z",
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": {"exception_type": error} if error else None,
        "agent_result": {
            "n_input_tokens": 100,
            "n_output_tokens": 20,
            "metadata": {"n_episodes": 2, "api_request_times_msec": list(request_times)},
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "message": '{"task_complete": true}',
                "metrics": {"completion_tokens": 18},
                "observation": {"results": [{"content": "New Terminal Output: ok"}]},
            }
        ]
    }
    (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory))


def test_load_local_reads_nested_reward_exception_latency_and_atif(tmp_path):
    _write_trial(tmp_path, "solved__a", reward=1, request_times=(120_000, 30_000))
    _write_trial(tmp_path, "timed-out__b", error="AgentTimeoutError")

    traces = load_local(tmp_path)
    report = compare(traces[:1], traces)

    assert [message["role"] for message in traces[0].messages] == ["assistant", "tool"]
    assert report["current"]["mean_reward_all_trials"] == 0.5
    assert report["current"]["agent_timeout_rate"] == 0.5
    assert report["current"]["api_request_seconds"]["over_120_seconds"] == 1
    assert report["current"]["mean_duration_seconds"] == pytest.approx(600)
    assert report["current"]["completion_tokens_by_call"]["max"] == 18
    assert report["current"]["behavior"]["no_action_messages"] == 1


def test_compare_reports_matched_task_regression():
    before = [Trace("same", 1, None, ())]
    after = [Trace("same", 0, None, ())]

    report = compare(before, after)

    assert report["delta"]["mean_reward_all_trials"] == -1
    assert report["matched_tasks"]["regressed_tasks"] == 1
    assert report["matched_tasks"]["largest_regressions"] == [("same", -1.0)]


def test_config_mismatches_reports_only_changed_semantic_fields():
    historical = {"parser": "xml", "max_output_tokens": 16384, "enable_thinking": True}
    current = {"parser": "json", "max_output_tokens": 8192, "enable_thinking": True, "agent_version": "2.0.0"}

    assert config_mismatches(historical, current) == {
        "max_output_tokens": {"historical": 16384, "current": 8192},
        "parser": {"historical": "xml", "current": "json"},
    }


def test_timeout_counterfactual_substitutes_current_timeout_rate():
    historical = {"accuracy": 0.5, "mean_reward_without_error": 0.8, "mean_reward_with_error": 0.1}
    current = {"agent_timeout_rate": 0.5, "mean_reward_all_trials": 0.3}

    result = timeout_counterfactual(historical, current)

    assert result["predicted_score_at_current_timeout_rate"] == pytest.approx(0.45)
    assert result["fraction_of_actual_drop_predicted"] == pytest.approx(0.25)
