# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest

combine = runpy.run_path(str(Path(__file__).with_name("combine.py")))["combine"]


def test_recovery_coverage_uses_full_cohort_and_excludes_transport_errors():
    retained = [{"task_name": str(i), "reward": 0, "exception_type": None} for i in range(437)]
    manifest = {"retained_trials": retained, "recovery_tasks": [str(i) for i in range(437, 500)]}
    recovered = [{"task_name": str(i), "reward": 1, "exception_type": None} for i in range(437, 450)]
    transport = {
        "task_name": "450",
        "reward": 0,
        "exception_type": "NonZeroAgentExitCodeError",
        "agent_logs": [{"terminal_error_classes": ["httpx.ReadTimeout"]}],
    }
    result = combine(manifest, retained, [*recovered, transport])
    assert result["n_scored"] == 450
    assert result["n_unscored"] == 1
    assert result["mean_reward"] == pytest.approx(13 / 450)
    assert result["sealable"]
    assert len(result["pending_tasks"]) == 49
    assert not combine(manifest, retained, [*recovered[:-1], transport])["sealable"]
    with pytest.raises(ValueError, match="duplicate"):
        combine(manifest, retained, [*recovered, recovered[0]])
