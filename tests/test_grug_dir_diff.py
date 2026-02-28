# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from scripts.grug_dir_diff import build_directory_diff_report, collect_files


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_directory_diff_report_classifies_files(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    out = tmp_path / "report"

    _write(left / "same.py", "print('same')\n")
    _write(right / "same.py", "print('same')\n")

    _write(left / "changed.py", "a = 1\nb = 2\n")
    _write(right / "changed.py", "a = 1\nb = 3\nc = 4\n")

    _write(left / "removed.py", "to_remove = True\n")
    _write(right / "added.py", "new_file = True\n")

    index_path, entries = build_directory_diff_report(
        left_dir=left,
        right_dir=right,
        output_dir=out,
        extensions=(".py",),
        include_all_files=False,
        show_unchanged=False,
        context_lines=2,
    )

    assert index_path.exists()

    by_rel = {entry.rel_path: entry for entry in entries}
    assert by_rel["same.py"].status == "unchanged"
    assert by_rel["changed.py"].status == "changed"
    assert by_rel["removed.py"].status == "removed"
    assert by_rel["added.py"].status == "added"

    report_html = index_path.read_text(encoding="utf-8")
    assert "changed.py" in report_html
    assert "removed.py" in report_html
    assert "added.py" in report_html
    assert "<h2>same.py</h2>" not in report_html


def test_collect_files_honors_all_files_flag(tmp_path: Path):
    root = tmp_path / "code"
    _write(root / "a.py", "x = 1\n")
    _write(root / "notes.txt", "hello\n")

    py_only = collect_files(root, extensions=(".py",), include_all_files=False)
    assert set(py_only) == {"a.py"}

    all_files = collect_files(root, extensions=(".py",), include_all_files=True)
    assert set(all_files) == {"a.py", "notes.txt"}


def test_build_directory_diff_report_uses_display_labels(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    out = tmp_path / "report"

    _write(left / "changed.py", "a = 1\n")
    _write(right / "changed.py", "a = 2\n")

    index_path, _entries = build_directory_diff_report(
        left_dir=left,
        right_dir=right,
        output_dir=out,
        left_label="experiments/grug/base",
        right_label="experiments/grug/moe",
        extensions=(".py",),
        include_all_files=False,
        show_unchanged=False,
        context_lines=2,
    )

    html = index_path.read_text(encoding="utf-8")
    assert "<strong>Left:</strong> experiments/grug/base" in html
    assert "<strong>Right:</strong> experiments/grug/moe" in html
    assert "experiments/grug/base/changed.py" in html
    assert "experiments/grug/moe/changed.py" in html
