# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fault-injection harness.

These cover the JAX-free decision path and the STICKY injector, which is the only
kind safe to fire inside the test process. SIGKILL/SIGSTOP/SPIN would kill, freeze,
or wedge the runner, so their selection is asserted through ``should_fire`` only.
"""

from __future__ import annotations

import pytest

from levanter.recovery.faults import (
    ENV_FAULT_KIND,
    ENV_FAULT_PROCESS,
    ENV_FAULT_STEP,
    FaultConfig,
    InjectedStickyFault,
    maybe_inject,
    should_fire,
)
from levanter.recovery.types import FaultKind


def test_fault_config_from_env_unset_disables_injection():
    cfg = FaultConfig.from_env({})

    # An unconfigured subprocess must never fire: kind NONE and a step that no
    # real step index matches.
    assert cfg.kind is FaultKind.NONE
    assert cfg.step == -1
    assert should_fire(cfg, step=0, process_index=0) is False


def test_fault_config_from_env_parses_typed_values():
    cfg = FaultConfig.from_env(
        {
            ENV_FAULT_KIND: "spin",
            ENV_FAULT_STEP: "5",
            ENV_FAULT_PROCESS: "2",
        }
    )

    assert cfg == FaultConfig(kind=FaultKind.SPIN, step=5, process=2)


def test_fault_config_from_env_rejects_unknown_kind():
    # Injection is a deliberate ablation knob, so a typo must fail fast rather
    # than silently disable the drill.
    with pytest.raises(ValueError, match="bogus"):
        FaultConfig.from_env({ENV_FAULT_KIND: "bogus"})


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
def test_should_fire_truth_table(kind, step, process_index, expected):
    cfg = FaultConfig(kind=kind, step=5, process=2)

    assert should_fire(cfg, step=step, process_index=process_index) is expected


def test_maybe_inject_sticky_at_matching_step_raises_with_cuda_marker():
    cfg = FaultConfig(kind=FaultKind.STICKY, step=3, process=0)

    with pytest.raises(InjectedStickyFault) as excinfo:
        maybe_inject(3, process_index=0, config=cfg)

    # Cross-module contract: detection.classify_exception matches this substring
    # to classify the fault as a poisoned CUDA context.
    assert "CUDA_ERROR_ILLEGAL_INSTRUCTION" in str(excinfo.value)


def test_maybe_inject_non_matching_step_is_noop():
    cfg = FaultConfig(kind=FaultKind.STICKY, step=3, process=0)

    # The per-step hook must be inert on every step but the configured one, so
    # training runs untouched until the fault is due.
    assert maybe_inject(2, process_index=0, config=cfg) is None
