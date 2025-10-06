# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from levanter.tracker.histogram import Histogram
from levanter.tracker.trackio import TrackioTracker


trackio = pytest.importorskip("trackio")


def test_log_summary(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = trackio.init(project="test-log-summary")
    tracker = TrackioTracker(run)
    tracker.log_summary({"float": 2.0})
    tracker.log_summary({"str": "test"})
    tracker.log_summary({"scalar_jax_array": jnp.array(3.0)})
    tracker.log_summary({"scalar_np_array": np.array(3.0)})
    trackio.finish()


def test_log(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = trackio.init(project="test-log")
    tracker = TrackioTracker(run)
    tracker.log({"float": 2.0}, step=0)
    tracker.log({"str": "test"}, step=0)
    tracker.log({"scalar_jax_array": jnp.array(3.0)}, step=0)
    tracker.log({"scalar_np_array": np.array(3.0)}, step=0)
    tracker.log({"histogram": Histogram.from_array(jnp.array([1.0, 2.0, 3.0]))}, step=0)
    trackio.finish()
