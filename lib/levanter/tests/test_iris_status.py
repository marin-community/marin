# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from levanter.callbacks import _iris_status
from levanter.callbacks._iris_status import _format_status, iris_status_reporter


@dataclasses.dataclass
class _FakeStep:
    step: int
    loss: float
    step_duration: float


class _RecordingClient:
    """Stand-in for the Iris client that records status-text pushes."""

    def __init__(self):
        self.calls: list[tuple] = []

    def report_task_status_text(self, task_id, attempt_id, detail_md, summary_md):
        self.calls.append((task_id, attempt_id, detail_md, summary_md))


@dataclasses.dataclass
class _FakeCtx:
    client: object


@dataclasses.dataclass
class _FakeJobInfo:
    task_id: str
    attempt_id: int


@pytest.fixture
def recording_client(monkeypatch):
    """Place the callback inside a fake Iris job and record its pushes."""
    client = _RecordingClient()
    monkeypatch.setattr(_iris_status, "get_iris_ctx", lambda: _FakeCtx(client=client))
    monkeypatch.setattr(_iris_status, "get_job_info", lambda: _FakeJobInfo(task_id="job/test/0", attempt_id=3))
    # Keep the W&B lookup out of the picture; it has no global tracker in tests.
    monkeypatch.setattr(_iris_status, "_wandb_run_url", lambda: None)
    return client


def test_reports_step_loss_and_throughput(recording_client):
    hook = iris_status_reporter(tokens_per_example=4096, batch_schedule=256, total_steps=1000)
    hook(_FakeStep(step=100, loss=2.5, step_duration=0.5), force=True)

    assert len(recording_client.calls) == 1
    task_id, attempt_id, detail_md, summary_md = recording_client.calls[0]
    assert task_id == "job/test/0"
    assert attempt_id == 3
    # 256 examples / 0.5s = 512 ex/s; 4096 tok/example -> ~2.1M tok/s.
    assert "2,097,152 tok/s" in detail_md
    assert "512.0 ex/s" in detail_md
    assert "2.5000" in detail_md  # loss
    assert "100/1,000" in detail_md  # step / total
    assert summary_md.startswith("step 100/1,000")
    assert "loss 2.500" in summary_md


def test_noop_outside_iris_job(monkeypatch):
    monkeypatch.setattr(_iris_status, "get_iris_ctx", lambda: None)
    monkeypatch.setattr(_iris_status, "get_job_info", lambda: None)
    hook = iris_status_reporter(tokens_per_example=4096, batch_schedule=256, total_steps=1000)
    # Must not raise even when forced.
    hook(_FakeStep(step=10, loss=1.0, step_duration=0.5), force=True)


def test_rate_limited_between_forced_pushes(recording_client):
    hook = iris_status_reporter(tokens_per_example=4096, batch_schedule=256, total_steps=1000, interval_seconds=1000.0)
    step = _FakeStep(step=10, loss=1.0, step_duration=0.5)

    hook(step)  # first unforced push fires (limiter has never run)
    hook(step)  # within the interval -> suppressed
    assert len(recording_client.calls) == 1

    hook(step, force=True)  # force bypasses the limiter
    assert len(recording_client.calls) == 2


def test_zero_step_duration_omits_throughput(recording_client):
    hook = iris_status_reporter(tokens_per_example=4096, batch_schedule=256, total_steps=1000)
    hook(_FakeStep(step=1, loss=3.0, step_duration=0.0), force=True)

    _, _, detail_md, summary_md = recording_client.calls[0]
    assert "tok/s" not in detail_md
    assert "tok/s" not in summary_md
    assert "3.0000" in detail_md  # loss still reported


def test_format_status_includes_wandb_link_and_eta():
    detail_md, summary_md = _format_status(
        step=500,
        total_steps=1000,
        loss=1.25,
        step_duration=2.0,
        tokens_per_second=10000.0,
        examples_per_second=20.0,
        mfu=42.0,
        wandb_url="https://wandb.ai/org/proj/runs/abc",
    )
    assert "[W&B run](https://wandb.ai/org/proj/runs/abc)" in detail_md
    assert "MFU**: 42.0%" in detail_md
    assert "ETA" in detail_md  # 500 steps remaining * 2s
    assert "(50.0%)" in detail_md
    assert "10.0k tok/s" in summary_md


def test_format_status_without_total_steps_has_no_progress_or_eta():
    detail_md, summary_md = _format_status(
        step=42,
        total_steps=None,
        loss=1.0,
        step_duration=1.0,
        tokens_per_second=None,
        examples_per_second=None,
        mfu=None,
        wandb_url=None,
    )
    assert "%" not in detail_md
    assert "ETA" not in detail_md
    assert "W&B" not in detail_md
    assert summary_md == "step 42 · loss 1.000"
