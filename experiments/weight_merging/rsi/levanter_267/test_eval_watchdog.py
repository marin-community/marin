# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

from experiments.weight_merging.rsi.levanter_267 import eval_watchdog


def test_dead_endpoint_stops_client(monkeypatch):
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: False)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.01,
            failure_seconds=0.03,
        )


def test_recovered_endpoint_allows_client_completion(monkeypatch):
    health = iter([False, False, True])
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: next(health, True))
    assert (
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(.1)"],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.01,
            failure_seconds=0.03,
        )
        == 0
    )


def test_runtime_bound_stops_client(monkeypatch):
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: True)
    with pytest.raises(TimeoutError, match="runtime bound"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            "unused",
            timeout=0.05,
            interval=0.01,
        )
