# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned-package oracle for two univariable fits on an observed clinical trial."""

import csv
import json
import math
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.real_clinical_binary import (
    fit_logistic_package,
    probability,
    write_table,
)

MODEL_PREDICTORS = {"age_only": "age_years", "bmi_only": "bmi_kg_m2"}
REQUIRED_FIELDS = ("age_years", "bmi_kg_m2", "throat_pain_30min")


def solve_clinical_univariable(inputs: Path, output: Path) -> list[dict]:
    """Fit age- and BMI-only binomial logits on identical complete cases."""
    with (inputs / "trial.tsv").open(newline="") as handle:
        trial = list(csv.DictReader(handle, delimiter="\t"))
    policy = json.loads((inputs / "analysis.json").read_text())
    if set(policy["models"]) != set(MODEL_PREDICTORS):
        raise ValueError("Univariable model policy changed")
    output.mkdir(parents=True, exist_ok=True)

    cohort_rows = []
    complete = []
    for row in trial:
        missing = [field for field in REQUIRED_FIELDS if not row[field]]
        if missing:
            cohort_rows.append(
                {
                    "id": row["id"],
                    "included": 0,
                    "no_pain_response": "NA",
                    "exclusion_reason": ",".join("missing_" + field for field in missing),
                }
            )
            continue
        event = int(float(row["throat_pain_30min"]) == 0)
        cohort_rows.append({"id": row["id"], "included": 1, "no_pain_response": event, "exclusion_reason": ""})
        complete.append((row, event))
    write_table(output / "cohort.tsv", cohort_rows)
    response = [event for _, event in complete]

    coefficients = []
    fits = []
    predictions = []
    model_results = {}
    for model, predictor in MODEL_PREDICTORS.items():
        design = [[1.0, float(row[predictor])] for row, _ in complete]
        beta, standard_errors, likelihood = fit_logistic_package(design, response)
        aic = 2 * len(beta) - 2 * likelihood
        model_results[model] = {"beta": beta, "aic": aic}
        for term, coefficient, error in zip(("intercept", predictor), beta, standard_errors, strict=True):
            z_value = coefficient / error
            coefficients.append(
                {
                    "id": f"{model}/{term}",
                    "model": model,
                    "term": term,
                    "coefficient": coefficient,
                    "standard_error": error,
                    "z": z_value,
                    "pvalue": math.erfc(abs(z_value) / math.sqrt(2)),
                    "odds_ratio": math.exp(coefficient),
                }
            )
        fits.append(
            {
                "id": model,
                "patients": len(complete),
                "responses": sum(response),
                "parameters": 2,
                "log_likelihood": likelihood,
                "aic": aic,
                "converged": 1,
            }
        )
        for profile in policy["profiles"][model]:
            value = float(profile[predictor])
            predictions.append(
                {
                    "id": f"{model}/{profile['id']}",
                    "model": model,
                    "profile": profile["id"],
                    "predictor": predictor,
                    "predictor_value": value,
                    "probability_no_pain": probability(beta[0] + beta[1] * value),
                }
            )
    write_table(output / "coefficients.tsv", coefficients)
    write_table(output / "model_fit.tsv", fits)
    write_table(output / "predictions.tsv", predictions)
    return [
        {
            "id": "study",
            "patients": len(trial),
            "complete_cases": len(complete),
            "no_pain_responses": sum(response),
            "age_only_aic": model_results["age_only"]["aic"],
            "bmi_only_aic": model_results["bmi_only"]["aic"],
            "age_only_age_coefficient": model_results["age_only"]["beta"][1],
            "bmi_only_bmi_coefficient": model_results["bmi_only"]["beta"][1],
        }
    ]


SOLVERS = {"real-clinical-univariable-logit-audit": solve_clinical_univariable}
