# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.evaluation.harbor_results import summarize_harbor_trials, write_harbor_result_summary


def test_harbor_result_summary_counts_completed_trials_rewards_and_exception_files(tmp_path):
    successful = tmp_path / "trial-one"
    failed = tmp_path / "trial-two"
    successful.mkdir()
    failed.mkdir()
    (successful / "result.json").write_text(json.dumps({"reward": 1.0}))
    (failed / "result.json").write_text(json.dumps({"reward": 0.0, "exception": "AgentTimeoutError: timed out"}))
    (failed / "exception.txt").write_text("VerifierTimeoutError: timed out\n")

    summary = summarize_harbor_trials(tmp_path)
    written = write_harbor_result_summary(tmp_path, tmp_path / "result.json")

    assert summary.completed_trials == 2
    assert summary.mean_reward == 0.5
    assert summary.exception_counts == {"AgentTimeoutError": 1, "VerifierTimeoutError": 1}
    assert written == summary
    assert json.loads((tmp_path / "result.json").read_text())["completed_trials"] == 2
