# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path

import pytest

from experiments.post_training.pivotrl.mixed_rlvr_figures import (
    Z80,
    Deltas,
    grid,
    load_deltas,
    missing_note,
    save,
    scatter,
    scatter_marks,
)

# The header ``mixed_rlvr_analysis plot`` writes; the figures module reads its CSV.
HEADER = "benchmark,arm,step,n,reference_accuracy,arm_accuracy,delta_points,ci80_low,ci80_high"


def write_deltas_csv(path: Path, rows: list[tuple[str, str, int, float]]) -> Path:
    lines = [HEADER]
    for bench, arm, step, delta in rows:
        lines.append(f"{bench},{arm},{step},500,26.0,{26 + delta},{delta},{delta - 2.4},{delta + 2.4}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_scatter_marks_pair_each_checkpoint_and_separate_coincident_marks_of_one_arm():
    deltas = Deltas(
        {
            ("swebench-verified", "a-seed43", 4): 4.2,
            ("math500", "a-seed43", 4): -0.8,
            ("swebench-verified", "a-seed43", 8): 3.8,
            ("math500", "a-seed43", 8): -0.8,
            ("swebench-verified", "b", 8): 4.0,
            ("math500", "b", 8): -0.8,
            ("swebench-verified", "c", 8): -1.2,  # C8 has no Math500 score
        },
        {"swebench-verified": 2.0, "math500": 0.9},
    )
    marks = scatter_marks(deltas, "math500")
    assert [(mark.arm, mark.step) for mark in marks] == [("a", 4), ("a", 8), ("b", 8)]
    a_marks = {mark.step: mark for mark in marks if mark.arm == "a"}
    assert a_marks[4].y - a_marks[8].y == pytest.approx(0.6)
    assert next(mark for mark in marks if mark.arm == "b").y == -0.8
    assert missing_note(deltas, "math500") == "not evaluated: C8"
    assert missing_note(deltas, "swebench-verified") == ""


def test_both_figures_draw_the_checkpoints_from_the_analysis_csv(tmp_path):
    csv_path = write_deltas_csv(
        tmp_path / "deltas.csv",
        [
            ("swebench-verified", "a", 4, 2.8),
            ("math500", "a", 4, 0.0),
            ("mmlu-pro", "a", 4, 0.9),
            ("gpqa-diamond", "a", 4, 3.9),
            ("swebench-verified", "b-seed43", 8, 6.8),
            ("math500", "b-seed43", 8, -0.6),
            ("gpqa-diamond", "b-seed43", 8, 9.6),
        ],
    )
    deltas = load_deltas(csv_path)
    assert deltas.errors["swebench-verified"] == pytest.approx(2.4 / Z80)
    json_path = tmp_path / "deltas.json"
    json_path.write_text(json.dumps(list(csv.DictReader(csv_path.open()))))
    assert load_deltas(json_path) == deltas
    figure = scatter(deltas, ["math500", "mmlu-pro", "gpqa-diamond"])
    math_panel, mmlu_panel, gpqa_panel = figure.axes
    assert [(t.get_text(), t.get_position()) for t in math_panel.texts if t.get_text() in {"4", "8"}] == [
        ("4", (2.8, 0.0)),
        ("8", (6.8, -0.6)),
    ]
    assert [t.get_text() for t in mmlu_panel.texts if t.get_text() in {"4", "8"}] == ["4"]
    assert "not evaluated: B8 s43" in [t.get_text() for t in mmlu_panel.texts]
    assert [t.get_text() for t in gpqa_panel.texts if t.get_text() in {"4", "8"}] == ["4", "8"]

    cells = grid(deltas)
    assert [t.get_text() for t in cells.axes[0].get_yticklabels()] == ["A  step  4  s42", "B  step  8  s43"]
    cell_text = [t.get_text() for t in cells.axes[0].texts]
    assert cell_text.count("—") == 12 - 7
    assert "+6.8" in cell_text and "-0.6" in cell_text

    save(figure, tmp_path / "scatter")
    assert {path.name for path in tmp_path.iterdir()} >= {"scatter.png", "scatter.pdf", "scatter.svg"}
