# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest

prepare = runpy.run_path(str(Path(__file__).with_name("prepare.py")))


def test_partition_preserves_model_failures_and_recovers_transport_failures():
    rows = [
        {"task_name": "pass", "reward": 1, "exception_type": None},
        {"task_name": "fail", "reward": 0, "exception_type": None},
        {"task_name": "timeout", "reward": 0, "exception_type": "AgentTimeoutError"},
        {
            "task_name": "transport",
            "reward": 0,
            "exception_type": "NonZeroAgentExitCodeError",
            "agent_logs": [{"terminal_error_classes": ["httpx.ReadTimeout"]}],
        },
        {
            "task_name": "retry",
            "reward": 1,
            "exception_type": "AgentTimeoutError",
            "agent_logs": [{"terminal_error_classes": [], "retry_error_classes": ["ReadTimeout"]}],
        },
        {"task_name": "verifier", "reward": None, "exception_type": "VerifierTimeoutError"},
    ]
    names = {row["task_name"] for row in rows} | {"missing"}
    retained, recovery = prepare["partition"]({"trials": rows}, names)
    assert {row["task_name"] for row in retained} == {"pass", "fail", "timeout"}
    assert recovery == ["missing", "retry", "transport", "verifier"]


def test_transport_keyword_alone_cannot_discard_a_scored_trial():
    row = {
        "task_name": "ambiguous",
        "reward": 0,
        "exception_type": "AgentTimeoutError",
        "transport_failure_observed": True,
        "agent_logs": [{"terminal_error_classes": []}],
    }
    with pytest.raises(ValueError, match="manual review"):
        prepare["partition"]({"trials": [row]}, {"ambiguous"})
