"""Read-only protocol launcher checks; fixture generation remains remote."""

from pathlib import Path

import pytest

from scripts.run_reset_protocol_probe import _docker_policy, run


def test_protocol_launcher_refuses_local_execution(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CAPABILITY_REMOTE_RESET_PROBE", raising=False)
    with pytest.raises(RuntimeError, match="remote-only"):
        run(tmp_path / "out", tmp_path / "builder.py")
    assert not (tmp_path / "out").exists()


def test_fixture_policy_is_frozen_from_observed_names():
    policy = _docker_policy({
        "process_comm_counts": {"python3": 1, "init": 1},
        "environment_names": ["PATH", "HOME"],
    })
    assert policy["process_policy"] == {
        "allowed_comm": ["init", "python3"], "max_count": 6,
    }
    assert policy["environment_name_policy"] == {
        "allowed_names": ["HOME", "PATH"],
        "required_names": ["HOME", "PATH"],
        "forbidden_names": [],
    }
    assert policy["mutation"]["command"].endswith("/app/reset-probe-marker.txt")
