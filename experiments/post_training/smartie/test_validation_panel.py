# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the fixed structural validation panel."""

import pytest

from experiments.post_training.smartie.validation_panel import selected_structural_ids


def test_structural_selection_balances_groups_and_ignores_input_order():
    groups = {
        "json/close": [f"json-{index}" for index in range(8)],
        "token/close": [f"token-{index}" for index in range(8)],
        "tool/close": [f"tool-{index}" for index in range(8)],
    }
    first = selected_structural_ids(groups, 10)
    reversed_groups = {key: list(reversed(values)) for key, values in reversed(list(groups.items()))}
    second = selected_structural_ids(reversed_groups, 10)
    assert first == second
    assert [sum(task_id.startswith(prefix) for task_id in first) for prefix in ("json", "token", "tool")] == [
        4,
        3,
        3,
    ]


def test_structural_selection_rejects_insufficient_group():
    with pytest.raises(ValueError, match="Insufficient structural rows for json/close"):
        selected_structural_ids({"json/close": ["one"], "tool/close": ["two", "three", "four"]}, 4)
