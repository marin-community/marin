# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent small-matrix logistic oracle for the observed licorice trial."""

import csv
import json
import math
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import table

TERMS = {
    "age": ("intercept", "licorice", "female", "age_per_10y"),
    "bmi": ("intercept", "licorice", "female", "bmi_per_5kg_m2"),
    "joint": ("intercept", "licorice", "female", "age_per_10y", "bmi_per_5kg_m2"),
}
PROFILES = (
    ("reference", 60, 25, 0, 0),
    ("younger", 40, 25, 0, 0),
    ("older", 80, 25, 0, 0),
    ("lower_bmi", 60, 20, 0, 0),
    ("higher_bmi", 60, 35, 0, 0),
    ("female", 60, 25, 1, 0),
    ("licorice", 60, 25, 0, 1),
    ("older_higher_bmi_licorice", 80, 35, 1, 1),
)


def features(row: dict[str, str], terms: tuple[str, ...]) -> list[float]:
    values = {
        "intercept": 1.0,
        "licorice": float(row["gargle_arm"]),
        "female": float(row["recorded_gender"]),
        "age_per_10y": (float(row["age_years"]) - 60) / 10,
        "bmi_per_5kg_m2": (float(row["bmi_kg_m2"]) - 25) / 5,
    }
    return [values[term] for term in terms]


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve the five-or-smaller information system with partial pivoting."""
    size = len(vector)
    rows = [[*row, value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ValueError("Logistic information matrix is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for index in range(size):
            if index == column:
                continue
            factor = rows[index][column]
            rows[index] = [old - factor * reduced for old, reduced in zip(rows[index], rows[column], strict=True)]
    return [row[-1] for row in rows]


def probability(eta: float) -> float:
    if eta >= 0:
        return 1 / (1 + math.exp(-eta))
    exponential = math.exp(eta)
    return exponential / (1 + exponential)


def log_likelihood(design: list[list[float]], response: list[int], beta: list[float]) -> float:
    total = 0.0
    for row, event in zip(design, response, strict=True):
        eta = math.fsum(value * coefficient for value, coefficient in zip(row, beta, strict=True))
        total += event * eta - (eta + math.log1p(math.exp(-eta)) if eta > 0 else math.log1p(math.exp(eta)))
    return total


def derivatives(
    design: list[list[float]], response: list[int], beta: list[float]
) -> tuple[list[float], list[list[float]]]:
    size = len(beta)
    score = [0.0] * size
    information = [[0.0] * size for _ in range(size)]
    for row, event in zip(design, response, strict=True):
        eta = math.fsum(value * coefficient for value, coefficient in zip(row, beta, strict=True))
        chance = probability(eta)
        weight = chance * (1 - chance)
        for first in range(size):
            score[first] += row[first] * (event - chance)
            for second in range(size):
                information[first][second] += weight * row[first] * row[second]
    return score, information


def fit_logistic(design: list[list[float]], response: list[int]) -> tuple[list[float], list[float], float]:
    """Newton MLE with a likelihood ascent check, independent of statsmodels."""
    fraction = sum(response) / len(response)
    beta = [math.log(fraction / (1 - fraction))] + [0.0] * (len(design[0]) - 1)
    for _ in range(100):
        score, information = derivatives(design, response, beta)
        change = solve_linear(information, score)
        baseline = log_likelihood(design, response, beta)
        for attempt in range(30):
            scale = 2.0**-attempt
            candidate = [value + scale * step for value, step in zip(beta, change, strict=True)]
            if log_likelihood(design, response, candidate) >= baseline - 1e-12:
                beta = candidate
                break
        else:
            raise ValueError("Logistic likelihood did not increase")
        if max(abs(scale * step) for step in change) < 1e-11:
            break
    else:
        raise ValueError("Logistic MLE did not converge")
    score, information = derivatives(design, response, beta)
    if max(map(abs, score)) > 1e-7:
        raise ValueError("Logistic score did not converge")
    variance = []
    for column in range(len(beta)):
        unit = [float(index == column) for index in range(len(beta))]
        variance.append(solve_linear(information, unit)[column])
    return beta, [math.sqrt(value) for value in variance], log_likelihood(design, response, beta)


def fit_logistic_package(design: list[list[float]], response: list[int]) -> tuple[list[float], list[float], float]:
    """Use the pinned statsmodels GLM and check it against independent Newton MLE."""
    import numpy as np  # noqa: PLC0415
    import statsmodels.api as sm  # noqa: PLC0415

    fitted = sm.GLM(
        np.asarray(response, dtype=float),
        np.asarray(design, dtype=float),
        family=sm.families.Binomial(),
    ).fit(maxiter=100, tol=1e-12)
    if not fitted.converged:
        raise ValueError("Native binomial-logit fit did not converge")
    beta = [float(value) for value in fitted.params]
    standard_errors = [float(value) for value in fitted.bse]
    likelihood = float(fitted.llf)
    checked_beta, checked_errors, checked_likelihood = fit_logistic(design, response)
    for native, checked in zip(
        beta + standard_errors + [likelihood], checked_beta + checked_errors + [checked_likelihood], strict=True
    ):
        if not math.isclose(native, checked, abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError("Independent logistic likelihood or information check differs")
    return beta, standard_errors, likelihood


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def solve_clinical_binary(inputs: Path, output: Path) -> list[dict]:
    """Fit all three prespecified models and retain the full artifact audit."""
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads((inputs / "analysis.json").read_text())
    if policy["model_terms"] != {name: list(terms) for name, terms in TERMS.items()}:
        raise ValueError("Unexpected clinical model specification")
    if policy["profiles"] != [list(profile) for profile in PROFILES]:
        raise ValueError("Unexpected clinical prediction profiles")
    rows = table(inputs / "trial.tsv", delimiter="\t")
    if len(rows) != 235 or len({row["id"] for row in rows}) != 235:
        raise ValueError("Changed participant table")
    required = ("age_years", "bmi_kg_m2", "recorded_gender", "gargle_arm", "throat_pain_30min")
    complete = [row for row in rows if all(row[key] for key in required)]
    cohort = [
        {
            "id": row["id"],
            "included": int(row in complete),
            "no_pain_response": int(float(row["throat_pain_30min"]) == 0) if row in complete else "NA",
            "exclusion_reason": "" if row in complete else "missing_30min_outcome",
        }
        for row in rows
    ]
    response = [int(float(row["throat_pain_30min"]) == 0) for row in complete]
    if len(complete) != 233 or sum(response) != 169:
        raise ValueError("Unexpected complete-case event counts")
    write_table(output / "cohort.tsv", cohort)

    models = {}
    coefficients = []
    fit_rows = []
    for name, terms in TERMS.items():
        design = [features(row, terms) for row in complete]
        beta, standard_errors, likelihood = fit_logistic_package(design, response)
        aic = 2 * len(terms) - 2 * likelihood
        models[name] = {"terms": terms, "beta": beta, "log_likelihood": likelihood, "aic": aic}
        fit_rows.append(
            {
                "id": name,
                "patients": len(complete),
                "responses": sum(response),
                "parameters": len(terms),
                "log_likelihood": likelihood,
                "aic": aic,
                "converged": 1,
            }
        )
        for term, coefficient, error in zip(terms, beta, standard_errors, strict=True):
            z_value = coefficient / error
            coefficients.append(
                {
                    "id": f"{name}/{term}",
                    "model": name,
                    "term": term,
                    "coefficient": coefficient,
                    "standard_error": error,
                    "z": z_value,
                    "pvalue": math.erfc(abs(z_value) / math.sqrt(2)),
                    "odds_ratio": math.exp(coefficient),
                }
            )
    write_table(output / "coefficients.tsv", coefficients)
    write_table(output / "model_fit.tsv", fit_rows)

    comparisons = []
    for reduced in ("age", "bmi"):
        ratio = max(0.0, 2 * (models["joint"]["log_likelihood"] - models[reduced]["log_likelihood"]))
        comparisons.append(
            {
                "id": f"{reduced}_to_joint",
                "reduced": reduced,
                "full": "joint",
                "likelihood_ratio": ratio,
                "degrees_freedom": 1,
                "pvalue": math.erfc(math.sqrt(ratio / 2)),
                "delta_aic": models["joint"]["aic"] - models[reduced]["aic"],
            }
        )
    write_table(output / "comparisons.tsv", comparisons)

    predictions = []
    for name, model in models.items():
        for identifier, age, bmi, female, arm in PROFILES:
            profile = {
                "age_years": str(age),
                "bmi_kg_m2": str(bmi),
                "recorded_gender": str(female),
                "gargle_arm": str(arm),
            }
            values = features(profile, model["terms"])
            chance = probability(math.fsum(a * b for a, b in zip(values, model["beta"], strict=True)))
            predictions.append(
                {
                    "id": f"{name}/{identifier}",
                    "model": name,
                    "profile": identifier,
                    "age_years": age,
                    "bmi_kg_m2": bmi,
                    "recorded_gender": female,
                    "gargle_arm": arm,
                    "probability_no_pain": chance,
                }
            )
    write_table(output / "predictions.tsv", predictions)
    joint = next(row for row in coefficients if row["id"] == "joint/licorice")
    return [
        {
            "id": "study",
            "patients": len(rows),
            "complete_cases": len(complete),
            "no_pain_responses": sum(response),
            "best_model_by_aic": min(models, key=lambda name: (models[name]["aic"], name)),
            "joint_licorice_odds_ratio": joint["odds_ratio"],
            "age_to_joint_pvalue": comparisons[0]["pvalue"],
            "bmi_to_joint_pvalue": comparisons[1]["pvalue"],
        }
    ]


SOLVERS = {"real-clinical-binary-response": solve_clinical_binary}
