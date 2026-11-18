# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The internal-image pull probe reports the backend's credential state."""

import asyncio

import pytest
from shellbox.machine import UnsupportedMachineSpec

from experiments.post_training.task_curation.backend_live_probes import (
    CANARY_IMAGE,
    PullProbeRecord,
    probe_internal_image_pull,
)


class FakeMachine:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.specs = []
        self.machine = None

    async def create(self, spec):
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        self.machine = FakeMachine()
        return self.machine


def test_pull_probe_reports_supported_when_the_canary_pulls(monkeypatch):
    factory = FakeFactory()
    monkeypatch.setattr(
        "experiments.post_training.task_curation.backend_live_probes._factory", lambda backend, storage_mb: factory
    )
    record = asyncio.run(probe_internal_image_pull("iris"))
    assert record == PullProbeRecord(
        backend="iris",
        state="supported",
        detail="the backend pulled the internal canary image and started the machine",
        credential_env=None,
    )
    assert factory.specs[0].source.reference == CANARY_IMAGE
    assert factory.machine.closed


def test_pull_probe_reports_unsupported_when_the_pull_is_refused(monkeypatch):
    factory = FakeFactory(error=UnsupportedMachineSpec("pull access denied for marin-community/iris-task"))
    monkeypatch.setattr(
        "experiments.post_training.task_curation.backend_live_probes._factory", lambda backend, storage_mb: factory
    )
    record = asyncio.run(probe_internal_image_pull("daytona"))
    assert record.state == "unsupported"
    assert "pull access denied" in record.detail
    assert record.credential_env == "DAYTONA_API_KEY"


def test_pull_probe_reports_unsupported_on_transport_failure(monkeypatch):
    factory = FakeFactory(error=TimeoutError("the image pull timed out"))
    monkeypatch.setattr(
        "experiments.post_training.task_curation.backend_live_probes._factory", lambda backend, storage_mb: factory
    )
    record = asyncio.run(probe_internal_image_pull("iris"))
    assert record.state == "unsupported"
    assert "could not pull" in record.detail


def test_pull_probe_refuses_local_backends():
    with pytest.raises(ValueError, match="remote backends only"):
        asyncio.run(probe_internal_image_pull("docker"))
