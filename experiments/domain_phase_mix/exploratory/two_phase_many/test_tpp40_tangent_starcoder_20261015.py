# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_starcoder_20261015 as runner


@pytest.mark.parametrize("family", runner.models.MODEL_NAMES)
def test_starcoder_adapter_uses_only_eligible_labels_and_realized_two_bucket_exposures(family: str) -> None:
    rng = np.random.default_rng(281)
    phase0, phase1 = rng.dirichlet(np.ones(2), 80), rng.dirichlet(np.ones(2), 80)
    phase1[-5:] = phase0[-5:]
    alpha = 3040 / 3814
    c0, c1 = alpha * np.array([1.0, 26.457868]), (1 - alpha) * np.array([1.0, 26.457868])
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    early, late = phase0 * c0, phase1 * c1
    matrix, offset = runner.models.design_and_offset(early, late, alpha, shape, "tangent_benefit_odds", {"kappa": 4.0})
    response = runner.models.frozen.temporal.predict_head(
        runner.models.mariner.Head(0.1, np.array([0.5, 0.8, 0.03, 0.02]), 0.6), matrix, offset
    )
    panel = runner.benchmark.data.BenchPanel(
        name="two_bucket_behavior_test",
        buckets=("web", "code"),
        phase0=phase0,
        phase1=phase1,
        alpha=alpha,
        c0=c0,
        c1=c1,
        targets={"test_bpb": response},
        rows=pd.DataFrame({"tied": np.arange(80) >= 75}),
        input_hashes={},
        metadata={"realized_phase0_epochs": early, "realized_phase1_epochs": late},
    )
    train, query, inner = np.arange(60), np.arange(60, 80), np.arange(60) % 3
    parameters = ({},) if family == "aggregate_mariner" else ({"kappa": 0.0}, {"kappa": 4.0})
    grid = runner.models.TestGrid(parameters, (0.01,), (1.0, 2.5, 6.0))
    fitted = runner.fit_partition(panel, "test_bpb", train, inner, family, None, test_shapes=(shape,), test_grid=grid)
    poisoned = response.copy()
    poisoned[query] = np.resize(np.array([np.nan, np.inf, -1e100, 1e100]), len(query))
    altered_panel = replace(panel, targets={"test_bpb": poisoned})
    replay = runner.fit_partition(
        altered_panel, "test_bpb", train, inner, family, None, test_shapes=(shape,), test_grid=grid
    )
    assert replay.to_json() == fitted.to_json()
    np.testing.assert_array_equal(replay.predict(early[query], late[query]), fitted.predict(early[query], late[query]))
    changed = response.copy()
    changed[train] += np.linspace(0.01, 0.1, len(train))
    changed_fit = runner.fit_partition(
        replace(panel, targets={"test_bpb": changed}),
        "test_bpb",
        train,
        inner,
        family,
        None,
        test_shapes=(shape,),
        test_grid=grid,
    )
    assert np.max(abs(changed_fit.predict(early[query], late[query]) - fitted.predict(early[query], late[query]))) > 1e-3
