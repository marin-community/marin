# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from levanter.tracker.histogram import SummaryStats
from levanter.tracker.trackio import TrackioTracker


trackio = pytest.importorskip("trackio")


def _capture_logs(monkeypatch):
    """Patch ``trackio.log`` to record forwarded payloads instead of persisting them."""
    calls = []
    monkeypatch.setattr(trackio, "log", lambda payload, step=None: calls.append((payload, step)))
    return calls


def _assert_json_native(value):
    """Trackio persists to SQLite, so every forwarded leaf must be a Python scalar/container."""
    if isinstance(value, dict):
        for v in value.values():
            _assert_json_native(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _assert_json_native(v)
    else:
        assert isinstance(value, (int, float, str, bool, type(None))), f"un-coerced leaf {value!r}: {type(value)}"


def test_log_coerces_values_and_preserves_step(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = trackio.init(project="test-log", embed=False)
    tracker = TrackioTracker(run)
    calls = _capture_logs(monkeypatch)

    tracker.log({"float": 2.0}, step=0)
    tracker.log({"str": "test"}, step=1)
    tracker.log({"scalar_jax_array": jnp.array(3.0)}, step=2)
    tracker.log({"scalar_np_array": np.array(4.0)}, step=3)

    by_step = {step: payload for payload, step in calls}
    assert by_step[0] == {"float": 2.0}
    assert isinstance(by_step[1]["str"], str)
    # jax/np scalars must be coerced to native Python floats, not left as arrays
    assert by_step[2] == {"scalar_jax_array": 3.0}
    assert isinstance(by_step[2]["scalar_jax_array"], float)
    assert by_step[3] == {"scalar_np_array": 4.0}
    assert isinstance(by_step[3]["scalar_np_array"], float)
    for payload, _ in calls:
        _assert_json_native(payload)

    trackio.finish()


def test_log_expands_summary_stats(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = trackio.init(project="test-log-hist", embed=False)
    tracker = TrackioTracker(run)
    calls = _capture_logs(monkeypatch)

    tracker.log({"histogram": SummaryStats.from_array(jnp.array([1.0, 2.0, 3.0]))}, step=0)
    tracker.log(
        {"summary_only": SummaryStats.from_array(jnp.array([1.0, 2.0, 3.0]), include_histogram=False)},
        step=1,
    )

    hist = calls[0][0]["histogram"]
    assert hist["min"] == 1.0
    assert hist["max"] == 3.0
    assert hist["mean"] == pytest.approx(2.0)
    assert hist["num"] == 3
    # histogram bucket payload survives the numpy -> list coercion
    assert isinstance(hist["histogram"]["counts"], list)
    assert isinstance(hist["histogram"]["limits"], list)
    assert sum(hist["histogram"]["counts"]) == 3

    summary_only = calls[1][0]["summary_only"]
    assert "histogram" not in summary_only
    assert summary_only["mean"] == pytest.approx(2.0)

    for payload, _ in calls:
        _assert_json_native(payload)

    trackio.finish()


def test_log_summary_prefixes_keys(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = trackio.init(project="test-log-summary", embed=False)
    tracker = TrackioTracker(run)
    calls = _capture_logs(monkeypatch)

    tracker.log_summary({"float": 2.0})
    tracker.log_summary({"scalar_jax_array": jnp.array(3.0)})

    # log_summary namespaces metrics under "summary/" and forwards without a step
    assert calls[0] == ({"summary/float": 2.0}, None)
    assert calls[1][0] == {"summary/scalar_jax_array": 3.0}
    assert isinstance(calls[1][0]["summary/scalar_jax_array"], float)

    trackio.finish()
