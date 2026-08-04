# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import marin.execution.step_runner as step_runner_module
import pytest


@pytest.fixture(autouse=True)
def skip_step_runner_poll_wait(monkeypatch):
    monkeypatch.setattr(step_runner_module, "_wait_for_job_status", lambda: None)
