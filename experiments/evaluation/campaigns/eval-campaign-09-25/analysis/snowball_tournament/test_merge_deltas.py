# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for merge-delta figure provenance."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_merge_deltas_preserves_exact_tracker_used_for_plots(tmp_path: Path) -> None:
    scores = {
        "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92": 0.2,
        "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02": 0.3,
        "open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10": 0.4,
        "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12": 0.35,
        "open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09": 0.25,
    }
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | math500 |\n| --- | --- |\n| Metric | accuracy |\n"
        + "".join(
            f"| {model} | {score:.3f} (s3://bucket/{index}/results) |\n"
            for index, (model, score) in enumerate(scores.items())
        )
    )
    output = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("merge_deltas.py")),
            "--tracker",
            str(tracker),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    snapshot = output / "TRACKER.snapshot.md"
    assert snapshot.read_bytes() == tracker.read_bytes()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["tracker_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert (
        metadata["models"]["open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09"]["base_label"] == "Snowball"
    )
    with (output / "merge_deltas.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert {row["benchmark"] for row in rows} == {"math500"}
    assert {row["merge_label"]: float(row["delta_pp"]) for row in rows} == pytest.approx(
        {"DARE-Linear-02": -5.0, "DARE-TIES-02": 10.0, "Shared100": 5.0}
    )
    assert all((output / f"{stem}.png").is_file() for stem in ("dare-linear-02", "dare-ties-02", "shared100"))
