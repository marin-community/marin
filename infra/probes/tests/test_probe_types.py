# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the core probe dataclasses and validation."""

from __future__ import annotations

import pytest
from probes.probe import (
    ErrorClass,
    ProbeOutcome,
    ProbeResult,
    ProbeSpec,
)


class _StubProbe:
    def run(self, deadline_seconds: float) -> ProbeResult:
        return ProbeResult.success()


def test_probe_result_success_helper():
    r = ProbeResult.success(extras={"foo": 1})
    assert r.outcome is ProbeOutcome.SUCCESS
    assert r.error_class is None
    assert r.extras == {"foo": 1}


def test_probe_result_remote_error_carries_target_id():
    r = ProbeResult.remote_error(ErrorClass.RPC_ERROR, "boom", target_id="job-1")
    assert r.outcome is ProbeOutcome.REMOTE_ERROR
    assert r.error_class is ErrorClass.RPC_ERROR
    assert r.target_id == "job-1"


def test_probe_spec_rejects_short_cadence():
    with pytest.raises(ValueError, match="cadence_seconds"):
        ProbeSpec(
            name="x",
            kind="X",
            location=None,
            cadence_seconds=5,
            deadline_seconds=1.0,
            probe=_StubProbe(),
        )


def test_probe_spec_rejects_deadline_ge_cadence():
    with pytest.raises(ValueError, match="deadline_seconds"):
        ProbeSpec(
            name="x",
            kind="X",
            location=None,
            cadence_seconds=30,
            deadline_seconds=30.0,
            probe=_StubProbe(),
        )


def test_probe_spec_accepts_valid_values():
    spec = ProbeSpec(
        name="x",
        kind="X",
        location="us-central1-a",
        cadence_seconds=60,
        deadline_seconds=5.0,
        probe=_StubProbe(),
    )
    assert spec.location == "us-central1-a"
