# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig


def test_starcoder_generic_selector_benchmark_smoke(tmp_path: Path):
    script = Path("experiments/domain_phase_mix/exploratory/starcoder_generic_selector_benchmark.py")
    output_dir = tmp_path / "benchmark_outputs"

    command = [
        sys.executable,
        str(script),
        "--output-dir",
        str(output_dir),
        "--datasets",
        "two_phase_starcoder",
        "--subset-sizes",
        "4,10",
        "--workers",
        "1",
        "--random-bootstrap-seeds",
        "1",
        "--retrospective-dopt-seeds",
        "1",
        "--prospective-seeds",
        "1",
        "--prospective-pool-size",
        "32",
        "--dsre-fit-seeds",
        "1",
        "--opt-search-points",
        "128",
        "--opt-restarts",
        "2",
        "--opt-maxiter",
        "40",
        "--row-limit",
        "12",
        "--skip-two-phase-oracle-ensure",
        "--skip-logbook",
    ]
    subprocess.run(command, check=True)

    expected_files = [
        output_dir / "selection_records.csv",
        output_dir / "model_scores.csv",
        output_dir / "curve_points.csv",
        output_dir / "selector_summary.csv",
        output_dir / "predicted_optima.csv",
        output_dir / "predicted_optima.jsonl",
        output_dir / "plots" / "two_phase_starcoder_panel.png",
        output_dir / "plots" / "two_phase_starcoder_panel_logx.png",
    ]
    for path in expected_files:
        assert path.exists(), path

    selection_records = pd.read_csv(output_dir / "selection_records.csv")
    model_scores = pd.read_csv(output_dir / "model_scores.csv")
    predicted_optima = pd.read_csv(output_dir / "predicted_optima.csv")

    assert not selection_records.empty
    assert set(selection_records["mode"]) == {"retrospective", "prospective"}
    assert not model_scores.empty
    assert {"DS-RE-CEQ", "CES-Overfit", "BayesLogQuad(w-e)"} <= set(model_scores["evaluation_model"])
    assert not predicted_optima.empty

    with (output_dir / "predicted_optima.jsonl").open() as handle:
        first = json.loads(handle.readline())
    weight_config = WeightConfig.from_dict(first["weight_config"])
    assert abs(sum(weight_config.phase_weights["phase_0"].values()) - 1.0) < 1e-9
