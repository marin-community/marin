# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from scripts.training.analyze_nested_moe_fixed import GLOBAL_STEP, STEP_DURATION, TRAIN_LOSS, histories


class SparseHistoryRun:
    def __init__(self) -> None:
        self.requested_keys: list[tuple[str, ...]] = []
        self.rows = {
            TRAIN_LOSS: [{GLOBAL_STEP: 7, TRAIN_LOSS: 3.5}],
            STEP_DURATION: [{GLOBAL_STEP: 7, STEP_DURATION: 0.4}],
        }

    def scan_history(self, *, keys: list[str], page_size: int) -> Iterable[dict[str, Any]]:
        assert page_size == 10_000
        self.requested_keys.append(tuple(keys))
        metric = keys[1]
        return self.rows.get(metric, [])


def test_histories_preserves_metrics_logged_on_different_rows() -> None:
    run = SparseHistoryRun()

    result = histories(run, include_nested=False)  # type: ignore[arg-type]

    assert [(point.step, point.value) for point in result["train_loss"]] == [(7, 3.5)]
    assert [(point.step, point.value) for point in result["step_duration"]] == [(7, 0.4)]
    assert all(len(keys) == 2 for keys in run.requested_keys)
