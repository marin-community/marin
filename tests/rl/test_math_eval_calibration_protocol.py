# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.math_eval.calibration_protocol import (
    BATTERIES,
    CalibrationProtocol,
    bounded_bytes,
    calibration_panels,
    calibration_protocol_receipt,
    validate_calibration_battery,
)


def test_calibration_protocol_order_and_response_budget_are_explicit():
    panels = calibration_panels()
    assert len(panels) == len({panel.identity for panel in panels}) == 9
    assert [(p.panel, p.samples, p.repetition) for p in panels] == [
        *(("dev_greedy", 1, repeat) for repeat in range(1, 6)),
        ("heldout_greedy", 1, 1),
        ("heldout_stochastic", 1, 1),
        ("heldout_stochastic", 4, 1),
        ("heldout_stochastic", 8, 1),
    ]
    assert sum(BATTERIES[p.split]["rows"] * p.samples for p in panels) == 29406
    receipts = [calibration_protocol_receipt(p) for p in panels]
    assert all(r["max_response_tokens"] == 1024 and r["context_tokens"] == 2048 for r in receipts)
    assert [(r["temperature"], r["top_p"]) for r in receipts] == [(0.0, 1.0)] * 6 + [(0.6, 0.95)] * 3
    assert all(r["engine_global_seed"] == 17 and r["request_sampling_seed"] is None for r in receipts)


@pytest.mark.parametrize(
    "args",
    [
        ("dev_greedy", 4, 1),
        ("dev_greedy", 1, 0),
        ("dev_greedy", 1, 6),
        ("heldout_greedy", 8, 1),
        ("heldout_stochastic", 2, 1),
        ("heldout_stochastic", 1, 2),
        ("unknown", 1, 1),
    ],
)
def test_calibration_refuses_unregistered_panels(args):
    with pytest.raises(ValueError):
        CalibrationProtocol(*args)


def test_calibration_metadata_bound_refuses_oversized_returned_bytes(tmp_path):

    path = tmp_path / "metadata"
    path.write_bytes(b"abcd")
    assert bounded_bytes(path, limit=4) == b"abcd"
    with pytest.raises(ValueError, match="byte bound"):
        bounded_bytes(path, limit=3)


def test_calibration_frozen_battery_rejects_changed_bytes_before_parsing():

    with pytest.raises(ValueError, match="battery bytes"):
        validate_calibration_battery(b"changed", [], {}, {}, protocol=CalibrationProtocol("heldout_greedy", 1))
