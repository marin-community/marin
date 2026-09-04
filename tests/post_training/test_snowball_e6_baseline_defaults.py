# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The CLI and the programmatic entry point must agree on every flag default.

`build_workflow` takes keyword defaults and the click command wraps it. When the two disagree, a
programmatic caller silently gets different behaviour from the command line -- and the flags here
select which kernels the forward runs, so the skew is not cosmetic. It has now happened twice
(`bf16_grad_reduce`, then `grouped_mm`) and been caught by review both times, never by a test.
"""

from __future__ import annotations

import inspect

from experiments.post_training import snowball_e6_baseline


def _click_defaults(command) -> dict[str, object]:
    """Each boolean flag's default, keyed by the parameter name it binds to."""
    return {param.name: param.default for param in command.params if param.default is not None or param.is_flag}


def test_every_click_flag_default_matches_the_programmatic_default() -> None:
    command = snowball_e6_baseline.main
    signature = inspect.signature(snowball_e6_baseline.build_workflow)
    click_defaults = _click_defaults(command)

    checked = 0
    mismatched: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.default is inspect.Parameter.empty or name not in click_defaults:
            continue
        if not isinstance(parameter.default, bool):
            continue
        checked += 1
        if click_defaults[name] != parameter.default:
            mismatched.append(f"{name}: click={click_defaults[name]!r} build_workflow={parameter.default!r}")

    assert checked >= 5, f"only {checked} boolean flags compared; the walk is not finding them"
    assert not mismatched, "CLI and programmatic defaults disagree:\n  " + "\n  ".join(mismatched)


def test_the_measured_throughput_flags_default_on() -> None:
    """These three are the configuration the result was measured at, so they are the default.

    grouped_mm was default-OFF between 2026-09-03 and 2026-09-04 because it broke the exact PPO ratio
    invariant on its own (F25/F26, and F31 showed the same defect house-wide). The grouped combine now
    reduces each token's rows in a fixed order and the E6 smoke on the real checkpoint reports
    log_ratio_abs_max 0 and exact-unit 1 on every step with it on, so it is back on.
    """
    signature = inspect.signature(snowball_e6_baseline.build_workflow)
    click_defaults = {param.name: param.default for param in snowball_e6_baseline.main.params}
    for name in ("grouped_mm", "flash_attn", "bf16_grad_reduce"):
        assert signature.parameters[name].default is True, name
        assert click_defaults[name] is True, name
