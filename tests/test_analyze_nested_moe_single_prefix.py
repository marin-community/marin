# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.training.analyze_nested_moe_fixed import HistoryPoint
from scripts.training.analyze_nested_moe_single_prefix import (
    _rolling_time_to_equivalent,
    _time_to_equivalent,
)


def test_time_to_equivalent_is_anchored_at_observed_terminal_losses() -> None:
    control = [
        HistoryPoint(step=1_000, value=4.0),
        HistoryPoint(step=2_000, value=3.6),
        HistoryPoint(step=4_000, value=3.2),
    ]
    treatment = [
        HistoryPoint(step=1_000, value=4.1),
        HistoryPoint(step=2_000, value=3.7),
        HistoryPoint(step=4_000, value=3.3),
    ]

    result = _time_to_equivalent(
        control,
        treatment,
        control_step_seconds=0.5,
        treatment_step_seconds=0.5,
    )

    assert result is not None
    assert result["control_loss"] == 3.2
    assert result["treatment_loss"] == 3.3
    assert result["extra_token_fraction"] > 0
    assert result["extra_time_fraction"] == pytest.approx(result["extra_token_fraction"])

    rolling = _rolling_time_to_equivalent(
        control,
        treatment,
        control_step_seconds=0.5,
        treatment_step_seconds=0.5,
    )

    assert len(rolling) == 1
    assert rolling[0].step == 4_000
    assert rolling[0].value > 0
