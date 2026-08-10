# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import marin.execution.step_runner as step_runner_module
import pytest
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec


@pytest.fixture(autouse=True)
def fake_step_runner_clock(monkeypatch):
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _interval: None)
    monkeypatch.setattr(step_runner_module, "time", clock)


def recording_step(name: str, out: str, executed: list[str], deps: list[StepSpec] | None = None) -> StepSpec:
    """A StepSpec whose fn appends ``name`` to ``executed`` when (and only when) it runs."""

    def _fn(output_path: str) -> Artifact:
        executed.append(name)
        return Artifact(path=output_path)

    return StepSpec(name=name, override_output_path=out, deps=deps or [], fn=_fn)
