# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe.benchmark_ep_ring_data_axis import (
    _BASELINE_MFU,
    _BASELINE_STEP_SECONDS,
    _parser,
    _projection,
    _replica_groups,
    _validate_args,
)


def test_ep_ring_data_axis_defaults_match_target_geometry() -> None:
    args = _parser().parse_args([])

    _validate_args(args)

    assert args.microbatch_size == 32
    assert args.sequence_length == 4096
    assert args.hidden_dim == 2560
    assert args.intermediate_dim == 1280
    assert args.num_experts == 64
    assert args.top_k == 4
    assert args.microbatches_per_step == 256
    assert args.layers_per_stage == 6
    assert args.treatment_data_axis_size == 2


def test_ep_ring_data_axis_projection_amortizes_step_boundary_work() -> None:
    projection = _projection(
        control_vag_ms=24.0,
        treatment_vag_ms=19.0,
        sync_ms=2.0,
        materialize_ms=3.0,
        microbatches_per_step=256,
        layers_per_stage=6,
        baseline_step_seconds=_BASELINE_STEP_SECONDS,
        baseline_mfu=_BASELINE_MFU,
    )

    assert projection["amortized_step_boundary_overhead_ms"] == pytest.approx(5.0 / 256)
    expected_step_seconds = _BASELINE_STEP_SECONDS - 6 * 256 * (24.0 - 19.0 - 5.0 / 256) / 1000
    assert projection["projected_step_seconds"] == pytest.approx(expected_step_seconds)
    assert projection["projected_mfu"] == pytest.approx(_BASELINE_MFU * _BASELINE_STEP_SECONDS / expected_step_seconds)


def test_ep_ring_data_axis_replica_groups_distinguish_data_and_expert_axes() -> None:
    assert _replica_groups(2, 4, axis="expert") == "{{0,1,2,3},{4,5,6,7}}"
    assert _replica_groups(2, 4, axis="data") == "{{0,4},{1,5},{2,6},{3,7}}"
