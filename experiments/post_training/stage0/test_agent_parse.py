# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.stage0.agent_parse import parse_report


def _step(source: str, observation: str = "") -> dict:
    return {"source": source, "observation": {"results": [{"content": observation}]}}


def test_parse_share_counts_agent_turns_whose_next_observation_reports_a_parse_error():
    trajectory = {
        "steps": [
            _step("user"),
            _step("agent", "Previous response had parsing errors:\nERROR: No valid JSON found"),
            _step("agent", "Previous response had warnings:\nWARNINGS: - Extra text detected"),
            _step("agent", "New Terminal Output: ok"),
        ]
    }
    report = parse_report([trajectory])
    assert (report["agent_turns"], report["parse_failures"]) == (3, 1)
    assert not report["passed"]
