# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fault-injection harness.

These cover the firing decision and the STICKY injector, which is the only kind
safe to fire inside the test process. SIGKILL/SIGSTOP/SPIN would kill, freeze, or
wedge the runner, so they are exercised only through ``should_fire``.
"""

from __future__ import annotations

import pytest

from levanter.recovery.faults import (
    FaultConfig,
    InjectedStickyFault,
    maybe_inject,
    should_fire,
)
from levanter.recovery.types import FaultKind


@pytest.mark.parametrize(
    ("kind", "step", "process_index", "expected"),
    [
        (FaultKind.STICKY, 5, 2, True),  # exact match fires
        (FaultKind.STICKY, 6, 2, False),  # wrong step
        (FaultKind.STICKY, 5, 1, False),  # wrong process
        (FaultKind.SIGKILL, 5, 2, True),  # selection works for lethal kinds too
        (FaultKind.NONE, 5, 2, False),  # no kind configured
    ],
)
def test_should_fire_matches_only_the_configured_step_and_process(kind, step, process_index, expected):
    cfg = FaultConfig(kind=kind, step=5, process=2)

    assert should_fire(cfg, step=step, process_index=process_index) is expected


def test_maybe_inject_sticky_raises_with_the_cross_module_cuda_marker():
    cfg = FaultConfig(kind=FaultKind.STICKY, step=3, process=0)

    with pytest.raises(InjectedStickyFault) as excinfo:
        maybe_inject(3, process_index=0, config=cfg)

    # Contract: detection.classify_exception keys off this substring to classify
    # the fault as a poisoned CUDA context.
    assert "CUDA_ERROR_ILLEGAL_INSTRUCTION" in str(excinfo.value)


def test_maybe_inject_is_inert_on_a_non_matching_step():
    cfg = FaultConfig(kind=FaultKind.STICKY, step=3, process=0)

    # The per-step hook must run untouched on every step but the configured one.
    assert maybe_inject(2, process_index=0, config=cfg) is None
