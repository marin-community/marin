# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``collective_overlap_limit`` overrides the ragged / inline-watch / default concurrent-collective limit."""

import os

import pytest

from experiments.grug.fast_track.train import XLA_COLLECTIVE_OVERLAP_FLAG, RaggedTransport, _apply_runtime_defaults


@pytest.mark.parametrize(("override", "expected"), [(None, 1), (3, 3)])
def test_ragged_limit_defaults_to_one_and_can_be_overridden(monkeypatch, override, expected):
    # A private copy: the defaults also set process-wide env (LD_PRELOAD, allocator) other tests must not inherit.
    monkeypatch.setattr(os, "environ", {**os.environ, "XLA_FLAGS": ""})
    _apply_runtime_defaults(
        inline_watch_enabled=False, ragged_transport=RaggedTransport.ONE_SHOT, collective_overlap_limit=override
    )
    assert f"{XLA_COLLECTIVE_OVERLAP_FLAG}={expected}" in os.environ["XLA_FLAGS"].split()
