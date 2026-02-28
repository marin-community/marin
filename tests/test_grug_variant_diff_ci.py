# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from scripts.grug_variant_diff_ci import directory_distance, find_closest_variant


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_directory_distance_is_zero_for_identical_dirs(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left / "model.py", "x = 1\n")
    _write(right / "model.py", "x = 1\n")

    distance = directory_distance(left_dir=left, right_dir=right, extensions=(".py",))

    assert distance == 0


def test_find_closest_variant_picks_smallest_line_delta(tmp_path: Path):
    new_variant = tmp_path / "new"
    base = tmp_path / "base"
    alt = tmp_path / "alt"

    _write(new_variant / "model.py", "a = 1\nb = 2\n")
    _write(base / "model.py", "a = 1\nb = 2\n")
    _write(alt / "model.py", "a = 1\nb = 99\nc = 4\n")

    match = find_closest_variant(
        variant_dir=new_variant,
        candidate_dirs={"base": base, "alt": alt},
        extensions=(".py",),
    )

    assert match.closest_variant == "base"
    assert match.distance_score == 0
