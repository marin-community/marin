# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run


@pytest.mark.parametrize(
    ("size", "num_steps", "expected_simulated_epoching"),
    [("d2048", None, True), ("d6144", 1, True), ("d6144", None, False)],
)
def test_scaling_ladder_disables_simulated_epoching_above_flop_limit(size, num_steps, expected_simulated_epoching):
    step = build_ladder_run(run_id=f"test-{size}", size=size, num_steps=num_steps, version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)

    data = step.build_config(ctx).data

    assert (data.target_budget is not None) is expected_simulated_epoching
    assert (data.experiment_budget is not None) is expected_simulated_epoching
