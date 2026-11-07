"""Check 2PL input construction does not silently impute unscored pairs."""

from typing import Any

import numpy as np
import pytest

from experiments.post_training.capability_driven_envs.biotasks_library.fit_irt_v3 import build_data, interval


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in range(12):
        for item in range(230):
            status, score = (
                ("ungraded", None)
                if (model, item) == (0, 0)
                else ("missing_infrastructure", None)
                if (model, item) == (1, 0)
                else ("submission_failure", 0)
                if (model, item) == (2, 0)
                else ("graded", int((model + item) % 2 == 0))
            )
            rows.append(
                {"model": f"m-{model:02d}", "task_id": f"task-{item:03d}", "grade_status": status, "score": score}
            )
    return rows


def test_primary_includes_malformed_final_zeros_but_omits_unscored_pairs() -> None:
    data, models, items = build_data(_rows(), include_submission_failures=True)
    assert (len(models), len(items), data["N"]) == (12, 230, 2758)
    assert data["respondent"][0] == 1
    assert data["item"][0] == 2
    assert (data["respondent"][458], data["item"][458], data["y"][458]) == (3, 1, 0)
    graded, _, _ = build_data(_rows(), include_submission_failures=False)
    assert graded["N"] == 2757


def test_rejects_duplicate_and_inconsistent_score() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="Duplicate"):
        build_data([rows[0], rows[0], *rows[2:]], include_submission_failures=True)
    rows[0] = {**rows[0], "score": 0}
    with pytest.raises(ValueError, match="Inconsistent"):
        build_data(rows, include_submission_failures=True)


def test_posterior_interval() -> None:
    assert interval(np.asarray([1.0] * 20)) == {"low": 1.0, "median": 1.0, "high": 1.0}
