# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from levanter.callbacks._core import StepInfo
from levanter.trainer import TrainerHooks, _hook_should_fire


@dataclass
class _FakeState:
    """Minimal stand-in for TrainerState exposing the step attribute StepInfo uses."""

    step: int


def _info_at(completed_step: int) -> StepInfo:
    # StepInfo.step is state.step - 1 (the step that just completed).
    return StepInfo(state=_FakeState(step=completed_step + 1), loss=0.0, step_duration=0.0)


def test_hook_should_fire_skips_step_zero_for_periodic_hooks():
    # `every=1` hooks must still run at step 0 (pbar, log_step_info, etc.).
    assert _hook_should_fire(step=0, every=1, force=False) is True
    # Periodic hooks (`every>1`) should skip step 0; they otherwise fire at multiples.
    assert _hook_should_fire(step=0, every=10, force=False) is False
    assert _hook_should_fire(step=10, every=10, force=False) is True
    assert _hook_should_fire(step=5, every=10, force=False) is False
    # `force` always wins.
    assert _hook_should_fire(step=0, every=10, force=True) is True
    assert _hook_should_fire(step=5, every=10, force=True) is True


def test_run_hooks_skips_periodic_at_step_zero():
    hooks = TrainerHooks()
    per_step: list[int] = []
    periodic: list[int] = []

    hooks.add_hook(lambda info: per_step.append(info.step), every=1)
    hooks.add_hook(lambda info: periodic.append(info.step), every=10)

    hooks.run_hooks(_info_at(0))
    # `every=1` fires; `every=10` does not (would otherwise spuriously match 0 % 10).
    assert per_step == [0]
    assert periodic == []

    # Skip ahead to step 10 — both should fire.
    hooks.run_hooks(_info_at(10))
    assert per_step == [0, 10]
    assert periodic == [10]


def test_run_hooks_force_overrides_step_zero_skip():
    hooks = TrainerHooks()
    fired: list[int] = []

    hooks.add_hook(lambda info: fired.append(info.step), every=100)

    # End-of-training drain calls run_hooks with force=True; the step-0 skip must not block it.
    hooks.run_hooks(_info_at(0), force=True)
    assert fired == [0]
