# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.3.3", "pyreadr==0.5.6", "scipy==1.17.0", "statsmodels==0.14.6"]
# ///

"""Preserve the observed licorice-gargle RCT and fit fixed binary-outcome references."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import chi2, norm

RDA_SHA256 = "aa0fe8957848031e6bd9df8d327ad31b19633bdf4b918f937bba5346acc3b9d4"
MIRROR_SHA256 = "2cc1078d1478095fd252e4d0cb9aa5ab5d71c9c41fdccb50af60a2756c937f4a"
MODEL_TERMS = {
    "age": ("intercept", "licorice", "female", "age_per_10y"),
    "bmi": ("intercept", "licorice", "female", "bmi_per_5kg_m2"),
    "joint": ("intercept", "licorice", "female", "age_per_10y", "bmi_per_5kg_m2"),
}
SOURCE_COLUMNS = (
    "preOp_gender",
    "preOp_asa",
    "preOp_calcBMI",
    "preOp_age",
    "preOp_mallampati",
    "preOp_smoking",
    "preOp_pain",
    "treat",
    "intraOp_surgerySize",
    "extubation_cough",
    "pacu30min_cough",
    "pacu30min_throatPain",
    "pacu30min_swallowPain",
    "pacu90min_cough",
    "pacu90min_throatPain",
    "postOp4hour_cough",
    "postOp4hour_throatPain",
    "pod1am_cough",
    "pod1am_throatPain",
)
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


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_frame(source: Path, mirror: Path) -> pd.DataFrame:
    """Read the original package RDA and verify the separate CSV conversion."""
    if sha256(source) != RDA_SHA256 or sha256(mirror) != MIRROR_SHA256:
        raise ValueError("Changed source RDA or independently published CSV mirror")
    objects = pyreadr.read_r(str(source))
    if set(objects) != {"licorice_gargle"}:
        raise ValueError(f"Unexpected RDA objects: {sorted(objects)}")
    frame = objects["licorice_gargle"]
    if frame.shape != (235, 19) or tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError("Unexpected trial shape or column order")
    copied = pd.read_csv(mirror).drop(columns=["rownames"])
    if tuple(copied.columns) != SOURCE_COLUMNS or copied.shape != frame.shape:
        raise ValueError("Independent mirror has different columns or rows")
    for column in SOURCE_COLUMNS:
        original = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        converted = pd.to_numeric(copied[column], errors="raise").to_numpy(dtype=float)
        if not np.array_equal(np.isnan(original), np.isnan(converted)):
            raise ValueError(f"Mirror missingness differs for {column}")
        if not np.allclose(original, converted, atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f"Mirror values differ for {column}")
    return frame


def observation_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Keep all trial participants and source missingness with stable row keys."""
    rows = []
    columns = {
        "age_years": "preOp_age",
        "bmi_kg_m2": "preOp_calcBMI",
        "recorded_gender": "preOp_gender",
        "gargle_arm": "treat",
        "throat_pain_30min": "pacu30min_throatPain",
    }
    for index, (_, source) in enumerate(frame.iterrows(), start=1):
        row = {"id": f"P{index:04d}"}
        for name, column in columns.items():
            value = source[column]
            row[name] = "" if pd.isna(value) else format(float(value), ".17g")
        rows.append(row)
    return rows


def study_cohort(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], np.ndarray]:
    """Apply one complete-case policy shared across all prespecified models."""
    required = ("age_years", "bmi_kg_m2", "recorded_gender", "gargle_arm", "throat_pain_30min")
    complete = [row for row in rows if all(row[key] for key in required)]
    if len(rows) != 235 or len(complete) != 233:
        raise ValueError("Unexpected baseline or outcome missingness")
    for row in complete:
        if int(float(row["recorded_gender"])) not in (0, 1) or int(float(row["gargle_arm"])) not in (0, 1):
            raise ValueError("Unexpected recorded gender or trial arm")
        if not (18 <= float(row["age_years"]) <= 86 and 0 < float(row["bmi_kg_m2"]) < 100):
            raise ValueError("Invalid observed age or BMI")
        pain = float(row["throat_pain_30min"])
        if pain != int(pain) or not 0 <= pain <= 10:
            raise ValueError("Unexpected throat-pain score")
    response = np.array([float(row["throat_pain_30min"]) == 0 for row in complete], dtype=float)
    if int(response.sum()) != 169:
        raise ValueError("Unexpected observed response count")
    return complete, response


def cohort_audit(rows: list[dict[str, str]]) -> list[dict]:
    audited = []
    for row in rows:
        present = all(
            row[key] for key in ("age_years", "bmi_kg_m2", "recorded_gender", "gargle_arm", "throat_pain_30min")
        )
        audited.append(
            {
                "id": row["id"],
                "included": int(present),
                "no_pain_response": int(float(row["throat_pain_30min"]) == 0) if present else "NA",
                "exclusion_reason": "" if present else "missing_30min_outcome",
            }
        )
    return audited


def design(rows: list[dict[str, str]], terms: tuple[str, ...]) -> np.ndarray:
    values = {
        "intercept": np.ones(len(rows)),
        "licorice": np.array([float(row["gargle_arm"]) for row in rows]),
        "female": np.array([float(row["recorded_gender"]) for row in rows]),
        "age_per_10y": np.array([(float(row["age_years"]) - 60) / 10 for row in rows]),
        "bmi_per_5kg_m2": np.array([(float(row["bmi_kg_m2"]) - 25) / 5 for row in rows]),
    }
    return np.column_stack([values[term] for term in terms])


def fit_models(rows: list[dict[str, str]], response: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    """Fit nonrobust binomial-logit MLEs and independently check fit arithmetic."""
    models = {}
    coefficient_rows = []
    fit_rows = []
    for name, terms in MODEL_TERMS.items():
        matrix = design(rows, terms)
        fitted = sm.GLM(response, matrix, family=sm.families.Binomial()).fit(maxiter=100, tol=1e-12)
        if not fitted.converged or not np.all(np.isfinite(fitted.params)):
            raise ValueError(f"Logistic MLE failed for {name}")
        beta = np.asarray(fitted.params, dtype=float)
        probability = expit(matrix @ beta)
        log_likelihood = float(np.sum(response * np.log(probability) + (1 - response) * np.log1p(-probability)))
        information = matrix.T @ (matrix * (probability * (1 - probability))[:, None])
        covariance = np.linalg.inv(information)
        standard_errors = np.sqrt(np.diag(covariance))
        if not np.allclose(standard_errors, fitted.bse, atol=1e-8, rtol=1e-8):
            raise ValueError(f"Observed information differs from package fit for {name}")
        if not math.isclose(log_likelihood, float(fitted.llf), abs_tol=1e-8):
            raise ValueError(f"Independent log likelihood differs for {name}")
        score = matrix.T @ (response - probability)
        if np.max(np.abs(score)) > 1e-7:
            raise ValueError(f"Fit score is not stationary for {name}")
        aic = 2 * len(terms) - 2 * log_likelihood
        if not math.isclose(aic, float(fitted.aic), abs_tol=1e-8):
            raise ValueError(f"Independent AIC differs for {name}")
        models[name] = {
            "terms": terms,
            "beta": beta.tolist(),
            "se": standard_errors.tolist(),
            "log_likelihood": log_likelihood,
            "aic": aic,
            "converged": True,
        }
        fit_rows.append(
            {
                "id": name,
                "patients": len(rows),
                "responses": int(response.sum()),
                "parameters": len(terms),
                "log_likelihood": log_likelihood,
                "aic": aic,
                "converged": 1,
            }
        )
        for term, coefficient, error in zip(terms, beta, standard_errors, strict=True):
            z_value = coefficient / error
            coefficient_rows.append(
                {
                    "id": f"{name}/{term}",
                    "model": name,
                    "term": term,
                    "coefficient": coefficient,
                    "standard_error": error,
                    "z": z_value,
                    "pvalue": 2 * norm.sf(abs(z_value)),
                    "odds_ratio": math.exp(coefficient),
                }
            )
    return models, coefficient_rows, fit_rows


def comparisons(models: dict) -> list[dict]:
    rows = []
    for reduced in ("age", "bmi"):
        statistic = 2 * (models["joint"]["log_likelihood"] - models[reduced]["log_likelihood"])
        if statistic < -1e-8:
            raise ValueError("Nested model has worse likelihood")
        rows.append(
            {
                "id": f"{reduced}_to_joint",
                "reduced": reduced,
                "full": "joint",
                "likelihood_ratio": max(0.0, statistic),
                "degrees_freedom": 1,
                "pvalue": chi2.sf(max(0.0, statistic), 1),
                "delta_aic": models["joint"]["aic"] - models[reduced]["aic"],
            }
        )
    return rows


def predictions(models: dict) -> list[dict]:
    rows = []
    for name, model in models.items():
        for identifier, age, bmi, female, arm in PROFILES:
            profile = [
                {"age_years": str(age), "bmi_kg_m2": str(bmi), "recorded_gender": str(female), "gargle_arm": str(arm)}
            ]
            value = float(expit(design(profile, model["terms"]) @ np.array(model["beta"]))[0])
            rows.append(
                {
                    "id": f"{name}/{identifier}",
                    "model": name,
                    "profile": identifier,
                    "age_years": age,
                    "bmi_kg_m2": bmi,
                    "recorded_gender": female,
                    "gargle_arm": arm,
                    "probability_no_pain": value,
                }
            )
    return rows


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rda", type=Path, required=True)
    parser.add_argument("--mirror-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = source_frame(args.source_rda, args.mirror_csv)
    observations = observation_rows(frame)
    cohort, response = study_cohort(observations)
    models, coefficient_rows, fit_rows = fit_models(cohort, response)
    tables = {
        "trial.tsv": observations,
        "cohort.tsv": cohort_audit(observations),
        "coefficients.tsv": coefficient_rows,
        "model_fit.tsv": fit_rows,
        "comparisons.tsv": comparisons(models),
        "predictions.tsv": predictions(models),
    }
    for name, rows in tables.items():
        write_table(args.output / name, rows)
    reference = {
        "source_package": "medicaldata 0.2.0",
        "rda_sha256": RDA_SHA256,
        "mirror_sha256": MIRROR_SHA256,
        "outcome": "pacu30min_throatPain == 0 among complete cases",
        "complete_cases": len(cohort),
        "no_pain_responses": int(response.sum()),
        "observed_age_range": [
            min(float(row["age_years"]) for row in observations),
            max(float(row["age_years"]) for row in observations),
        ],
        "observed_bmi_range": [
            min(float(row["bmi_kg_m2"]) for row in observations),
            max(float(row["bmi_kg_m2"]) for row in observations),
        ],
        "model_terms": MODEL_TERMS,
        "profiles": PROFILES,
        "models": models,
        "artifact_sha256": {name: sha256(args.output / name) for name in tables},
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyreadr", "scipy", "statsmodels")
        },
    }
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "patients": len(observations),
                "complete_cases": len(cohort),
                "no_pain_responses": int(response.sum()),
                "models": list(models),
            }
        )
    )


if __name__ == "__main__":
    main()
