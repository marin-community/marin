# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import marin.execution.step_runner as step_runner_module
import pytest


@pytest.fixture(autouse=True)
def fake_step_runner_clock(monkeypatch):
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _interval: None)
    monkeypatch.setattr(step_runner_module, "time", clock)
