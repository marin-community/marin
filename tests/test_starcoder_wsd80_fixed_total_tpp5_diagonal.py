# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np

from experiments.domain_phase_mix import launch_starcoder_wsd80_fixed_total_tpp5_diagonal as launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_fixed_total_tpp5_diagonal_20260904 as design,
)


def test_design_reproduces_frozen_manifest() -> None:
    assert design.build_manifest() == json.loads(design.OUTPUT_PATH.read_text())


def test_ladder_holds_total_parameter_tpp_and_matches_compute_rungs() -> None:
    payload = design.build_manifest()
    cells = payload["cells"]

    assert payload["expected_run_count"] == 60
    assert payload["canary_run_name"] == design.CANARY_RUN_NAME
    assert len(cells) == 4
    assert design.source_compute_targets() == design.TARGET_COMPUTE_FLOPS
    np.testing.assert_allclose(
        [cell["materialized_tokens"] / cell["total_parameters"] for cell in cells],
        design.TARGET_TOTAL_PARAMETER_TPP,
        rtol=design.MAX_RELATIVE_TPP_MISMATCH,
    )
    np.testing.assert_allclose(
        [cell["compute_flops"] for cell in cells],
        [cell["target_compute_flops"] for cell in cells],
        rtol=design.MAX_RELATIVE_COMPUTE_MISMATCH,
    )
    assert all(
        left["total_parameters"] < right["total_parameters"]
        and left["materialized_tokens"] < right["materialized_tokens"]
        for left, right in pairwise(cells)
    )
    assert all(cell["num_layers"] > cell["natural_num_layers"] for cell in cells)

    runs = payload["runs"]
    assert {run["data_seed"] for run in runs} == {design.REFERENCE_SEED}
    assert {run["simulated_epoch_subset_seed"] for run in runs} == {design.REFERENCE_SEED}
    assert all(run["phase_0_starcoder"] == run["phase_1_starcoder"] for run in runs)
    assert {
        run["coordinate_id"]: sum(candidate["coordinate_id"] == run["coordinate_id"] for candidate in runs)
        for run in runs
    } == {coordinate["coordinate_id"]: 4 for coordinate in payload["coordinates"]}


def test_launcher_audits_current_runtime_against_frozen_design() -> None:
    cells, runs = launcher.load_design()

    assert len(cells) == 4
    assert len(runs) == 60
    assert launcher.DEFAULT_MAX_CONCURRENT == launcher.EXPECTED_RUN_COUNT
