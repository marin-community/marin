# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import build_dclm_bigbench_rescored_full_matrix as overlay
from experiments.domain_phase_mix.exploratory.two_phase_many import dclm_matrix_guard
from experiments.domain_phase_mix.exploratory.two_phase_many import fit_dclm_core_dsp_300m as dclm_fit
from experiments.domain_phase_mix.exploratory.two_phase_many import recompute_dclm_bigbench_generation_scores as rescore


def test_filtered_exact_match_left_strips_and_ignores_punctuation_without_changing_internal_spaces() -> None:
    assert rescore.filtered_exact_match({"target": "17", "resps": [[" 17"]]}) == 1.0
    assert rescore.filtered_exact_match({"target": "hello world", "resps": [[" hello, world!"]]}) == 1.0
    assert rescore.filtered_exact_match({"target": "hello world", "resps": [[" hello  world"]]}) == 0.0


def test_generation_eval_key_selects_generation_component_from_merged_rows() -> None:
    assert rescore._generation_eval_key("dclm300m_001_generation_run;dclm300m_001_mcq_run") == (
        "dclm300m_001_generation_run"
    )
    assert rescore._generation_eval_key("dclm300m_001_mcq_run") is None


def test_eval_key_from_sample_path_strips_executor_hash() -> None:
    path = (
        "marin-us-east5/prefix/evaluation/lm_evaluation_harness/"
        "dclm300m_001_generation_signal_300m_6b_baseline_proportional_full22_all_20260608-78a697/"
        "bb_operators_10shot_10shot/model/samples_bigbench_operators_generate_until_x.jsonl"
    )

    assert rescore._eval_key_from_sample_path(path) == (
        "dclm300m_001_generation_signal_300m_6b_baseline_proportional_full22_all_20260608"
    )


def test_recompute_dclm_macro_uses_corrected_per_task_centered_columns() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "lm_eval/dclm_core/bb_qa_wikidata_10shot/centered_accuracy": 0.25,
                "lm_eval/dclm_core/bb_operators_10shot/centered_accuracy": 0.75,
                "lm_eval/dclm_core/centered_accuracy_macro": 0.0,
                "lm_eval/dclm_core/task_count": 0.0,
                "lm_eval/dclm_core/missing_task_count": 22.0,
            }
        ]
    )

    rescore.recompute_dclm_macro(frame)

    assert frame.loc[0, "lm_eval/dclm_core/centered_accuracy_macro"] == pytest.approx(0.5)
    assert frame.loc[0, "lm_eval/dclm_core/task_count"] == 2.0
    assert frame.loc[0, "lm_eval/dclm_core/missing_task_count"] == 20.0


def test_apply_rescores_replaces_task_and_derived_scores(monkeypatch) -> None:
    task = rescore.RESCORABLE_TASKS[0]
    frame = pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_proportional",
                "eval_key": "dclm300m_001_generation_run;dclm300m_001_mcq_run",
                task.metric_column: 0.0,
                task.raw_score_column: 0.0,
                task.centered_column: 0.0,
                "lm_eval/dclm_core/boolq_10shot/centered_accuracy": 0.5,
            }
        ]
    )

    class FakeFS:
        def open(self, path: str, mode: str):  # noqa: ANN001
            assert path == "bucket/sample.jsonl"
            return open(__file__)

    def fake_read_jsonl(_fs, path: str) -> list[dict[str, object]]:
        assert path == "bucket/sample.jsonl"
        return [{"target": "17", "resps": [[" 17"]]}, {"target": "18", "resps": [[" 0"]]}]

    monkeypatch.setattr(rescore.fsspec, "filesystem", lambda _name: FakeFS())
    monkeypatch.setattr(rescore, "_read_jsonl", fake_read_jsonl)

    corrected, audit = rescore.apply_rescores(
        frame,
        {(task.alias, "dclm300m_001_generation_run"): ["bucket/sample.jsonl"]},
        (task,),
    )

    assert corrected.loc[0, task.metric_column] == 0.5
    assert corrected.loc[0, task.raw_score_column] == 0.5
    assert corrected.loc[0, task.centered_column] == 0.5
    assert corrected.loc[0, "lm_eval/dclm_core/centered_accuracy_macro"] == pytest.approx(0.5)
    assert audit.loc[0, "status"] == "rescored"
    assert audit.loc[0, "sample_count"] == 2


def test_overlay_bigbench_rescores_preserves_full_columns_and_recomputes_macro() -> None:
    task = rescore.RESCORABLE_TASKS[0]
    records = []
    for run_name in ("run_a", "run_b"):
        record = {
            "run_name": run_name,
            "lm_eval/dclm_core/boolq_10shot/centered_accuracy": 0.25,
            "lm_eval/dclm_core/centered_accuracy_macro": 0.0,
            "extra_full_column": "keep",
        }
        for rescore_task in rescore.RESCORABLE_TASKS:
            record[rescore_task.metric_column] = 0.0
            record[rescore_task.raw_score_column] = 0.0
            record[rescore_task.centered_column] = 0.0
        records.append(record)
    base = pd.DataFrame.from_records(records)
    for index in range(rescore.DCLM_TOTAL_TASKS - 1 - len(rescore.RESCORABLE_TASKS)):
        base[f"lm_eval/dclm_core/filler_{index}/centered_accuracy"] = 0.5
    rescored = base.copy()
    for task_index, rescore_task in enumerate(rescore.RESCORABLE_TASKS):
        rescored.loc[:, rescore_task.metric_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
        rescored.loc[:, rescore_task.raw_score_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
        rescored.loc[:, rescore_task.centered_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
    rescored.loc[0, task.metric_column] = 0.75
    rescored.loc[0, task.raw_score_column] = 0.75
    rescored.loc[0, task.centered_column] = 0.75

    corrected = overlay.overlay_bigbench_rescores(
        base,
        rescored,
        Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored.csv"),
    )

    assert corrected.columns.tolist() == base.columns.tolist()
    assert corrected.loc[0, task.metric_column] == 0.75
    assert corrected.loc[0, task.raw_score_column] == 0.75
    assert corrected.loc[0, task.centered_column] == 0.75
    assert corrected.loc[0, "extra_full_column"] == "keep"
    filler_count = rescore.DCLM_TOTAL_TASKS - 1 - len(rescore.RESCORABLE_TASKS)
    expected_macro = (0.75 + 0.15 + 0.2 + 0.25 + 0.25 + filler_count * 0.5) / rescore.DCLM_TOTAL_TASKS
    assert corrected.loc[0, "lm_eval/dclm_core/centered_accuracy_macro"] == pytest.approx(expected_macro)


def test_macro_consistency_assertion_rejects_stale_macro() -> None:
    frame = pd.DataFrame.from_records([{"lm_eval/dclm_core/centered_accuracy_macro": 0.0}])
    for index in range(rescore.DCLM_TOTAL_TASKS):
        frame[f"lm_eval/dclm_core/filler_{index}/centered_accuracy"] = 1.0

    with pytest.raises(ValueError, match="DCLM macro is inconsistent"):
        overlay.assert_dclm_macro_consistent(frame)


def _valid_guard_frame(*, repeat_copy_values: tuple[float, float] = (0.0, 0.03125)) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(
        [
            {
                "run_name": "run_a",
                "lm_eval/dclm_core/centered_accuracy_macro": 0.0,
                "lm_eval/dclm_core/task_count": float(rescore.DCLM_TOTAL_TASKS),
                "lm_eval/dclm_core/missing_task_count": 0.0,
            },
            {
                "run_name": "run_b",
                "lm_eval/dclm_core/centered_accuracy_macro": 0.0,
                "lm_eval/dclm_core/task_count": float(rescore.DCLM_TOTAL_TASKS),
                "lm_eval/dclm_core/missing_task_count": 0.0,
            },
        ]
    )
    for index in range(rescore.DCLM_TOTAL_TASKS - len(rescore.RESCORABLE_TASKS) - 1):
        frame[f"lm_eval/dclm_core/filler_{index}/centered_accuracy"] = 0.25
    for task_index, task in enumerate(rescore.RESCORABLE_TASKS):
        frame[task.centered_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
        frame[task.raw_score_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
        frame[task.metric_column] = [0.1 + 0.05 * task_index, 0.2 + 0.05 * task_index]
    frame["lm_eval/dclm_core/bb_repeat_copy_logic_10shot/centered_accuracy"] = list(repeat_copy_values)
    frame["lm_eval/dclm_core/bb_repeat_copy_logic_10shot/raw_score"] = list(repeat_copy_values)
    rescore.recompute_dclm_macro(frame)
    return frame


def test_corrected_matrix_guard_rejects_known_stale_filenames() -> None:
    frame = _valid_guard_frame()

    with pytest.raises(ValueError, match="stale DCLM matrix"):
        dclm_matrix_guard.validate_corrected_dclm_matrix(
            frame,
            Path("300m_dclm_core_eval_results_merged.csv"),
        )


def test_corrected_matrix_guard_rejects_unrescored_bigbench_columns() -> None:
    frame = _valid_guard_frame()
    task = rescore.RESCORABLE_TASKS[0]
    frame[task.raw_score_column] = 0.0
    frame[task.centered_column] = 0.0
    rescore.recompute_dclm_macro(frame)

    with pytest.raises(ValueError, match=task.alias):
        dclm_matrix_guard.validate_corrected_dclm_matrix(
            frame,
            Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"),
        )


def test_corrected_matrix_guard_rejects_intermediate_bigbench_only_matrix_by_default() -> None:
    frame = _valid_guard_frame()

    with pytest.raises(ValueError, match="stale DCLM matrix"):
        dclm_matrix_guard.validate_corrected_dclm_matrix(
            frame,
            Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored.csv"),
        )


def test_corrected_matrix_guard_allows_intermediate_bigbench_only_builder_output() -> None:
    frame = _valid_guard_frame(repeat_copy_values=(0.0, 0.0))

    dclm_matrix_guard.validate_corrected_dclm_matrix(
        frame,
        Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored.csv"),
        allow_intermediate_repeat_copy=True,
    )


def test_corrected_matrix_guard_rejects_missing_repeat_copy_overlay() -> None:
    frame = _valid_guard_frame(repeat_copy_values=(0.0, 0.0))

    with pytest.raises(ValueError, match="repeat-copy 128-token overlay"):
        dclm_matrix_guard.validate_corrected_dclm_matrix(
            frame,
            Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"),
        )


def test_corrected_matrix_guard_allows_final_repeat_copy_overlay() -> None:
    frame = _valid_guard_frame()

    dclm_matrix_guard.validate_corrected_dclm_matrix(
        frame,
        Path("300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"),
    )


def test_dclm_fit_loader_rejects_explicit_stale_matrix_override(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw_metric_matrix_300m.csv"
    stale_dclm_csv = tmp_path / "300m_dclm_core_eval_results_full_after_retry8.csv"
    pd.DataFrame.from_records([{"run_name": "run_a", "row_kind": "signal", "status": "completed"}]).to_csv(
        raw_csv,
        index=False,
    )
    _valid_guard_frame().to_csv(stale_dclm_csv, index=False)

    with pytest.raises(ValueError, match="stale DCLM matrix"):
        dclm_fit.load_fit_frame(raw_csv, stale_dclm_csv, dclm_fit.TARGET_COLUMN)
