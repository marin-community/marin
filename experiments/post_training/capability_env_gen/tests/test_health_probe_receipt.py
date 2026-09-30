import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("observation", [True, False, "lookup_error"])
def test_health_observation_and_cleanup_both_survive_in_receipt(
    tmp_path, monkeypatch, observation
):
    monkeypatch.setitem(sys.modules, "daytona", SimpleNamespace(
        CreateSandboxFromSnapshotParams=lambda **kwargs: kwargs,
    ))
    spec = importlib.util.spec_from_file_location(
        "health_probe_under_test",
        Path(__file__).resolve().parents[1] / "scripts/probe_daytona_health.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deleted = []
    sandbox = SimpleNamespace(id="sandbox", delete=lambda: deleted.append("sandbox"))

    def observe(sandbox_id):
        assert sandbox_id == "sandbox"
        if observation == "lookup_error":
            raise RuntimeError("provider observation failed")
        return SimpleNamespace(network_block_all=observation)

    client = SimpleNamespace(
        snapshot=SimpleNamespace(get=lambda _: SimpleNamespace(
            id="snapshot-id", name="snapshot", state="active",
        )),
        create=lambda parameters, timeout: sandbox,
        get=observe,
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "fake-test-key")
    monkeypatch.setattr(module, "_dt", lambda: SimpleNamespace(client=lambda: client))
    monkeypatch.setattr(module, "version", lambda _: "test-sdk")
    monkeypatch.setattr(module, "wait_for_sandbox_deletion", lambda *_: (
        "not_found", [{"elapsed_seconds": 0.0, "state": "not_found"}],
    ))
    output = tmp_path / "health.json"
    result = module.run("snapshot", output)
    receipt = json.loads(output.read_text())
    assert result == (0 if observation is True else 1)
    assert receipt["state"] == ("passed" if observation is True else "failed")
    assert receipt["deleted"] is True
    assert receipt["lookup_after_delete"] == "not_found"
    assert receipt["deletion_lookups"][-1]["state"] == "not_found"
    assert deleted == ["sandbox"]
    if observation == "lookup_error":
        assert receipt["error_type"] == "RuntimeError"
        assert receipt["network_block_all_observed"] is None
    else:
        assert receipt["network_block_all_observed"] is observation
