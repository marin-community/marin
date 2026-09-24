# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.bio_tasks.generators.real_clinical import generate_clinical
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.solvers.real_clinical import solve_clinical


@pytest.mark.parametrize("treatment", ["all", "1", "2"])
@pytest.mark.parametrize("biomarker", ["bili", "albumin", "protime"])
def test_cox_coefficients_uncertainty_and_events_match_independent_statsmodels(tmp_path, treatment, biomarker):
    instance = generate_clinical(0, "real-clinical-adjusted-cox")
    for name, content in instance.inputs.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "query.json").write_text(json.dumps({"treatment": treatment, "biomarker": biomarker}))
    answer = solve_clinical(tmp_path, "real-clinical-adjusted-cox")
    expected = json.loads(source_text("survival:PBC", "pbc-cox-reference.json.gz"))["models"][f"{treatment}/{biomarker}"]
    for index, row in enumerate(answer):
        assert row["coefficient"] == pytest.approx(expected["coefficients"][index], abs=1e-9)
        assert row["standard_error"] == pytest.approx(expected["standard_errors"][index], abs=1e-9)
        assert row["pvalue"] == pytest.approx(expected["pvalues"][index], abs=1e-9)
        assert row["patients"] == expected["patients"]
        assert row["events"] == expected["events"]


def test_survival_joins_ids_excludes_nonrandomized_and_orders_tied_events(tmp_path):
    (tmp_path / "patients.tsv").write_text("id\ttrt\na\t1\nb\t1\nc\t1\nd\t1\ne\t\n")
    (tmp_path / "outcomes.tsv").write_text("id\ttime\tstatus\nd\t4\t0\ne\t1\t2\nb\t2\t0\na\t2\t2\nc\t3\t1\n")
    (tmp_path / "query.json").write_text(json.dumps({"treatment": "all", "horizons_days": [2, 3, 4]}))
    assert solve_clinical(tmp_path, "real-clinical-kaplan-meier") == [
        {"id": "2", "patients": 4, "at_risk": 4, "events": 1, "censored": 1, "survival": 0.75},
        {"id": "3", "patients": 4, "at_risk": 2, "events": 2, "censored": 1, "survival": 0.375},
        {"id": "4", "patients": 4, "at_risk": 1, "events": 2, "censored": 2, "survival": 0.375},
    ]


def test_paired_visits_selects_nearest_before_dropping_missing_values(tmp_path):
    (tmp_path / "patients.tsv").write_text("id\ttrt\na\t1\nb\t1\nc\t1\nd\t\n")
    (tmp_path / "visits.tsv").write_text(
        "id\tday\tbili\na\t0\t10\na\t10\t\na\t12\t100\n"
        "b\t0\t20\nb\t12\t30\nb\t8\t23\nc\t10\t10\nc\t0\t8\nd\t0\t1\nd\t10\t1000\n"
    )
    (tmp_path / "query.json").write_text(
        json.dumps({"treatment": "all", "target_day": 10, "window_days": 5, "biomarkers": ["bili"]})
    )
    assert solve_clinical(tmp_path, "real-clinical-paired-visits") == [
        {"id": "bili", "pairs": 2, "mean_change": 2.5, "median_change": 2.5}
    ]
