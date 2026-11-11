"""Startup diagnostics must not turn a polling interval into a failure deadline."""

import pytest

from skyrl_train.inference_engines import ray_wrapped_inference_engine as startup


@pytest.mark.parametrize("eventually_ready", [True, False])
def test_startup_wait_preserves_pending_actors_across_progress_polls(monkeypatch, eventually_ready):
    clock = [0.0]
    replies = ["ready", "slow"]
    killed = []
    diagnostics = []

    def wait(pending, *, num_returns, timeout, fetch_local):
        if pending == replies:
            return ["ready"], ["slow"]
        clock[0] += timeout
        if eventually_ready and clock[0] >= 60:
            return ["slow"], []
        return [], pending

    monkeypatch.setattr(startup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(startup.ray, "wait", wait)
    monkeypatch.setattr(startup.ray, "get", lambda refs: refs)
    monkeypatch.setattr(startup.ray, "kill", killed.append)
    monkeypatch.setattr(startup.logger, "info", lambda fmt, *args: diagnostics.append(fmt.format(*args)))
    if eventually_ready:
        assert startup.wait_for_inference_engine_startup(replies, ["actor0", "actor1"], timeout_seconds=90) == replies
        assert killed == []
    else:
        with pytest.raises(TimeoutError, match=r"pending engine actors: \[1\]"):
            startup.wait_for_inference_engine_startup(replies, ["actor0", "actor1"], timeout_seconds=90)
        assert killed == ["actor0", "actor1"]
        assert clock[0] == 90
    assert any("1/2 ready; pending actor indices [1]" in message for message in diagnostics)
