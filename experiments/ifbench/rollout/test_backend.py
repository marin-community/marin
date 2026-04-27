# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the RolloutBackend protocol surface.

The protocol is structurally typed so we test by registering a stub
implementation and verifying it's usable as a `RolloutBackend`.
"""

from __future__ import annotations

import pathlib

import pytest

from experiments.ifbench.rollout.backend import (
    BatchHandle,
    BatchStatus,
    Rollout,
    RolloutBackend,
    RolloutRequest,
    SamplingConfig,
)


class _StubBackend:
    name = "stub"

    def submit_batch(self, model_id, requests, jsonl_dir):
        reqs = list(requests)
        return BatchHandle(
            backend=self.name,
            batch_id="stub-1",
            model_id=model_id,
            submitted_at_iso="2026-04-26T00:00:00Z",
            expected_request_count=len(reqs),
        )

    def poll(self, handle):
        return BatchStatus.COMPLETED

    def download(self, handle):
        return iter(
            [
                Rollout(
                    prompt_id="p1",
                    model_id=handle.model_id,
                    backend=self.name,
                    response_text="hi",
                    finish_reason="stop",
                    input_tokens=4,
                    output_tokens=1,
                    thinking_tokens=None,
                    seed=0,
                    sampling_config_hash="abc",
                )
            ]
        )


def test_stub_satisfies_protocol() -> None:
    """Any duck-typed object with the right shape should pass isinstance()."""
    stub: RolloutBackend = _StubBackend()
    assert isinstance(stub, RolloutBackend)


def test_sampling_config_hash_stable() -> None:
    a = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
    b = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
    c = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024, thinking_level="high")
    assert a.hash_short() == b.hash_short()
    assert a.hash_short() != c.hash_short()


def test_round_trip(tmp_path: pathlib.Path) -> None:
    """Mock end-to-end: submit → poll → download yields the right shape."""
    backend: RolloutBackend = _StubBackend()
    sampling = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
    requests = [
        RolloutRequest(
            prompt_id="p1",
            model_id="m1",
            messages=[{"role": "user", "content": "hi"}],
            sampling=sampling,
            seed=0,
        )
    ]
    handle = backend.submit_batch("m1", requests, tmp_path)
    assert handle.expected_request_count == 1
    assert backend.poll(handle) is BatchStatus.COMPLETED
    rollouts = list(backend.download(handle))
    assert len(rollouts) == 1
    assert rollouts[0].prompt_id == "p1"
    assert rollouts[0].backend == "stub"


@pytest.mark.parametrize("status", list(BatchStatus))
def test_status_is_string(status: BatchStatus) -> None:
    """StrEnum so backends can serialise status to logs trivially."""
    assert isinstance(status.value, str)
    assert isinstance(str(status), str)
