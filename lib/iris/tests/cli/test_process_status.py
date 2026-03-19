# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.process_status — --target option parsing."""

from click.testing import CliRunner

from iris.cli.process_status import profile


def test_profile_default_target_label(monkeypatch):
    _calls = []

    class _FakeClient:
        def profile_task(self, req):
            _calls.append(req.target)
            raise RuntimeError("stop")

    monkeypatch.setattr("iris.cli.process_status.require_controller_url", lambda ctx: "http://fake")
    monkeypatch.setattr("iris.cli.process_status.ControllerServiceClientSync", lambda url: _FakeClient())

    runner = CliRunner()
    runner.invoke(profile, ["cpu"], obj={})
    assert _calls == ["/system/process"]


def test_profile_task_target(monkeypatch):
    _calls = []

    class _FakeClient:
        def profile_task(self, req):
            _calls.append(req.target)
            raise RuntimeError("stop")

    monkeypatch.setattr("iris.cli.process_status.require_controller_url", lambda ctx: "http://fake")
    monkeypatch.setattr("iris.cli.process_status.ControllerServiceClientSync", lambda url: _FakeClient())

    runner = CliRunner()
    runner.invoke(profile, ["cpu", "--target", "/alice/my-job/0"], obj={})
    assert _calls == ["/alice/my-job/0"]


def test_profile_worker_target(monkeypatch):
    _calls = []

    class _FakeClient:
        def profile_task(self, req):
            _calls.append(req.target)
            raise RuntimeError("stop")

    monkeypatch.setattr("iris.cli.process_status.require_controller_url", lambda ctx: "http://fake")
    monkeypatch.setattr("iris.cli.process_status.ControllerServiceClientSync", lambda url: _FakeClient())

    runner = CliRunner()
    runner.invoke(profile, ["cpu", "--target", "/system/worker/abc123"], obj={})
    assert _calls == ["/system/worker/abc123"]
