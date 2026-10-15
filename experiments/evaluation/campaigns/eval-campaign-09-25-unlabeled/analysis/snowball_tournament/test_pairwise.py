# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for tracker-derived Snowball/Grug tournament membership."""

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_new_grug_rows_enter_tournament_and_joint_leaders_share_rank(tmp_path: Path) -> None:
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | math500 | swebench-verified | tau3-pi |\n"
        "| --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy |\n"
        "| open-athena/Snowball-Step92 | 0.900 (s3://bucket/regraded/step-math.json) | "
        "0.300 (s3://bucket/step-swe/results) | 0.500 (s3://bucket/step-tau/results) |\n"
        "| open-athena/Grug-Antidoom-Step12 | 0.900 (s3://bucket/anti-math/results) | "
        "0.300 (s3://bucket/anti-swe/results) | 0.500 (s3://bucket/anti-tau/results) |\n"
        "| open-athena/Grug-dr-doom | 0.800 (s3://bucket/doom-math/results) | "
        "0.200 (s3://bucket/doom-swe/results) | RUNNING |\n"
        "| Qwen/baseline | 0.990 (s3://bucket/qwen-math/results) | "
        "0.990 (s3://bucket/qwen-swe/results) | 0.990 (s3://bucket/qwen-tau/results) |\n"
    )
    output = tmp_path / "tournament"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("pairwise.py")),
            "--tracker",
            str(tracker),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    selection = json.loads((output / "snowball_selection.json").read_text())
    assert selection["candidates"] == [
        "open-athena/Snowball-Step92",
        "open-athena/Grug-Antidoom-Step12",
        "open-athena/Grug-dr-doom",
    ]
    assert selection["joint_leaders"] == ["open-athena/Grug-Antidoom-Step12", "open-athena/Snowball-Step92"]
    with (output / "snowball_pairwise_summary.csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert [row["mean_pairwise_win_rate"] for row in summary] == ["0.75", "0.75", "0.0"]
    report = (output / "snowball_pairwise_win_rates.md").read_text()
    assert "| 1 | S1 |" in report
    assert "| 1 | S2 |" in report
    assert "| 3 | S3 |" in report
