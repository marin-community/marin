# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation for corrected 300M DCLM Core metric matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import recompute_dclm_bigbench_generation_scores as rescore


MACRO_COLUMN = "lm_eval/dclm_core/centered_accuracy_macro"
TASK_COUNT_COLUMN = "lm_eval/dclm_core/task_count"
MISSING_TASK_COUNT_COLUMN = "lm_eval/dclm_core/missing_task_count"
FINAL_CORRECTED_DCLM_MATRIX_FILENAME = (
    "300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"
)
INTERMEDIATE_BIGBENCH_RESCORDED_MATRIX_FILENAME = "300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored.csv"
CORRECTED_DCLM_MATRIX_FILENAME = FINAL_CORRECTED_DCLM_MATRIX_FILENAME
STALE_DCLM_MATRIX_FILENAMES = frozenset(
    {
        "300m_dclm_core_eval_results_merged.csv",
        "300m_dclm_core_eval_results_full_after_retry8.csv",
        INTERMEDIATE_BIGBENCH_RESCORDED_MATRIX_FILENAME,
    }
)
RERUN_REQUIRED_ALIASES = frozenset(rescore.RERUN_REQUIRED_TASKS)
REPEAT_COPY_RAW_COLUMN = "lm_eval/dclm_core/bb_repeat_copy_logic_10shot/raw_score"
REPEAT_COPY_CENTERED_COLUMN = "lm_eval/dclm_core/bb_repeat_copy_logic_10shot/centered_accuracy"


def dclm_centered_columns(frame: pd.DataFrame) -> list[str]:
    """Return per-task DCLM centered-accuracy columns."""
    return sorted(
        column
        for column in frame.columns
        if column.startswith("lm_eval/dclm_core/")
        and column.endswith("/centered_accuracy")
        and column != MACRO_COLUMN
    )


def assert_dclm_macro_consistent(frame: pd.DataFrame, *, atol: float = 1e-12) -> None:
    """Assert the DCLM macro equals the mean of its per-task centered columns."""
    centered_columns = dclm_centered_columns(frame)
    if len(centered_columns) != rescore.DCLM_TOTAL_TASKS:
        raise ValueError(f"Expected {rescore.DCLM_TOTAL_TASKS} centered columns, found {len(centered_columns)}")
    numeric = frame[centered_columns].apply(pd.to_numeric, errors="coerce")
    expected = numeric.mean(axis=1).to_numpy(dtype=float)
    actual = pd.to_numeric(frame[MACRO_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True):
        max_abs = float(np.nanmax(np.abs(actual - expected)))
        raise ValueError(f"DCLM macro is inconsistent with component mean; max abs diff={max_abs}")


def validate_corrected_dclm_matrix(
    frame: pd.DataFrame,
    matrix_path: Path,
    *,
    allow_intermediate_repeat_copy: bool = False,
) -> None:
    """Validate that a DCLM matrix is the corrected post-rescore matrix.

    This intentionally rejects known stale filenames and also detects stale
    content when the four rescored BigBench generation components are still at
    the old constant-zero floor. The canonical 300M DCLM matrix also includes
    the 128-token repeat-copy rerun overlay. Builders that produce the
    intermediate BigBench-only overlay may opt in with
    ``allow_intermediate_repeat_copy``; model-fitting code should not.
    """
    if matrix_path.name in STALE_DCLM_MATRIX_FILENAMES and not (
        allow_intermediate_repeat_copy and matrix_path.name == INTERMEDIATE_BIGBENCH_RESCORDED_MATRIX_FILENAME
    ):
        raise ValueError(
            f"{matrix_path} is a stale DCLM matrix. Use {CORRECTED_DCLM_MATRIX_FILENAME} "
            "or rebuild the final overlay from BigBench rescore and repeat-copy 128-token results."
        )
    if MACRO_COLUMN not in frame.columns:
        raise ValueError(f"DCLM matrix is missing macro column {MACRO_COLUMN!r}")

    assert_dclm_macro_consistent(frame)
    _assert_complete_task_counts(frame)
    _assert_rescored_bigbench_signal(frame)
    if not allow_intermediate_repeat_copy:
        _assert_repeat_copy_overlay_signal(frame)


def _assert_complete_task_counts(frame: pd.DataFrame) -> None:
    if TASK_COUNT_COLUMN not in frame.columns or MISSING_TASK_COUNT_COLUMN not in frame.columns:
        return

    macro = pd.to_numeric(frame[MACRO_COLUMN], errors="coerce")
    observed = macro.notna()
    if not observed.any():
        return
    task_count = pd.to_numeric(frame.loc[observed, TASK_COUNT_COLUMN], errors="coerce")
    missing_count = pd.to_numeric(frame.loc[observed, MISSING_TASK_COUNT_COLUMN], errors="coerce")
    bad = (task_count != float(rescore.DCLM_TOTAL_TASKS)) | (missing_count != 0.0)
    if bad.any():
        raise ValueError(
            "DCLM matrix has incomplete task counts for rows with a macro score; "
            f"bad row count={int(bad.sum())}"
        )


def _assert_rescored_bigbench_signal(frame: pd.DataFrame) -> None:
    for task in rescore.RESCORABLE_TASKS:
        for column in (task.raw_score_column, task.centered_column):
            if column not in frame.columns:
                raise ValueError(f"DCLM matrix is missing corrected BigBench column {column!r}")
        raw_values = pd.to_numeric(frame[task.raw_score_column], errors="coerce")
        observed = raw_values.dropna()
        if observed.nunique(dropna=True) <= 1 or float(observed.max()) <= 0.0:
            raise ValueError(
                f"DCLM matrix appears unrescored for {task.alias}: "
                f"{task.raw_score_column!r} has no positive variation"
            )


def _assert_repeat_copy_overlay_signal(frame: pd.DataFrame) -> None:
    for column in (REPEAT_COPY_RAW_COLUMN, REPEAT_COPY_CENTERED_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"DCLM matrix is missing repeat-copy overlay column {column!r}")
    raw_values = pd.to_numeric(frame[REPEAT_COPY_RAW_COLUMN], errors="coerce")
    observed = raw_values.dropna()
    if observed.nunique(dropna=True) <= 1 or float(observed.max()) <= 0.0:
        raise ValueError(
            "DCLM matrix appears to be missing the repeat-copy 128-token overlay: "
            f"{REPEAT_COPY_RAW_COLUMN!r} has no positive variation"
        )
