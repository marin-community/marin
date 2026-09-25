# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.3.3", "pyreadr==0.5.6", "scipy==1.17.0", "statsmodels==0.14.6"]
# ///

"""Fit frozen univariable logits to the original licorice trial observations."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

RDA_MEMBER = "medicaldata/data/licorice_gargle.rda"
SOURCE_FIELDS = {
    "age_years": "preOp_age",
    "bmi_kg_m2": "preOp_calcBMI",
    "recorded_gender": "preOp_gender",
    "gargle_arm": "treat",
    "throat_pain_30min": "pacu30min_throatPain",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def original_frame(archive: Path, protocol: dict) -> pd.DataFrame:
    """Read the authoritative RDA member from the byte-checked CRAN archive."""
    if sha256(archive) != protocol["source_archive_sha256"]:
        raise ValueError("Changed original CRAN source archive")
    with tarfile.open(archive, "r:gz") as package:
        member = package.getmember(RDA_MEMBER)
        handle = package.extractfile(member)
        if handle is None or member.size > 2 * 1024 * 1024:
            raise ValueError("Missing or oversized original RDA")
        contents = handle.read(member.size + 1)
    if hashlib.sha256(contents).hexdigest() != protocol["original_rda_sha256"]:
        raise ValueError("Changed original licorice RDA")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "licorice_gargle.rda"
        path.write_bytes(contents)
        objects = pyreadr.read_r(str(path))
    if set(objects) != {"licorice_gargle"}:
        raise ValueError("Unexpected source RDA objects")
    frame = objects["licorice_gargle"]
    if frame.shape != (235, 19):
        raise ValueError("Original clinical study size changed")
    return frame


def observed_rows(trial_asset: Path, frame: pd.DataFrame, protocol: dict) -> list[dict[str, str]]:
    """Verify every solver-visible value against its original trial row."""
    trial = gzip.decompress(trial_asset.read_bytes())
    if hashlib.sha256(trial).hexdigest() != protocol["solver_input_content_sha256"]:
        raise ValueError("Changed solver-visible trial rows")
    rows = list(csv.DictReader(io.StringIO(trial.decode()), delimiter="\t"))
    if len(rows) != len(frame) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Trial row count or IDs changed")
    for index, row in enumerate(rows):
        if row["id"] != f"P{index + 1:04d}":
            raise ValueError("Source-order patient ID changed")
        original = frame.iloc[index]
        for field, source_field in SOURCE_FIELDS.items():
            source = original[source_field]
            value = row[field]
            if pd.isna(source):
                if value:
                    raise ValueError(f"Input missingness differs for {field}")
            elif not value or float(value) != float(source):
                raise ValueError(f"Input value differs from RDA for {field}")
    return rows


def cohort_rows(rows: list[dict[str, str]], protocol: dict) -> tuple[list[dict], list[dict], list[int]]:
    required = protocol["common_cohort"]["required_columns"]
    audit = []
    complete = []
    response = []
    for row in rows:
        missing = [field for field in required if not row[field]]
        if missing:
            audit.append(
                {
                    "id": row["id"],
                    "included": 0,
                    "no_pain_response": "NA",
                    "exclusion_reason": ",".join("missing_" + field for field in missing),
                }
            )
            continue
        pain = float(row["throat_pain_30min"])
        if pain != int(pain) or not 0 <= pain <= 10:
            raise ValueError("Unexpected observed throat-pain score")
        event = int(pain == 0)
        audit.append({"id": row["id"], "included": 1, "no_pain_response": event, "exclusion_reason": ""})
        complete.append(row)
        response.append(event)
    if not complete or not 0 < sum(response) < len(response):
        raise ValueError("Observed complete-case response is not estimable")
    return audit, complete, response


def newton_reference(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Independently solve the two-parameter score equations using NumPy."""
    fraction = float(np.mean(response))
    beta = np.array([math.log(fraction / (1 - fraction)), 0.0])
    for _ in range(100):
        chance = expit(design @ beta)
        score = design.T @ (response - chance)
        information = design.T @ (design * (chance * (1 - chance))[:, None])
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-11:
            break
    else:
        raise ValueError("Independent Newton fit did not converge")
    chance = expit(design @ beta)
    likelihood = float(np.sum(response * np.log(chance) + (1 - response) * np.log1p(-chance)))
    information = design.T @ (design * (chance * (1 - chance))[:, None])
    standard_errors = np.sqrt(np.diag(np.linalg.inv(information)))
    if float(np.max(np.abs(design.T @ (response - chance)))) > 1e-7:
        raise ValueError("Independent Newton score is not stationary")
    return beta, standard_errors, likelihood


def fit_models(complete: list[dict], response: list[int], protocol: dict) -> tuple[dict, list[dict], list[dict]]:
    """Check native GLM coefficients with direct likelihood and independent Newton fit."""
    models = {}
    coefficients = []
    fits = []
    for model, predictor in (("age_only", "age_years"), ("bmi_only", "bmi_kg_m2")):
        if protocol["models"][model]["terms"] != ["intercept", predictor]:
            raise ValueError("Changed frozen univariable design")
        design = np.column_stack((np.ones(len(complete)), [float(row[predictor]) for row in complete]))
        observed = np.asarray(response, dtype=float)
        fitted = sm.GLM(observed, design, family=sm.families.Binomial()).fit(maxiter=100, tol=1e-12)
        if not fitted.converged or not np.all(np.isfinite(fitted.params)):
            raise ValueError(f"Native {model} logistic fit failed")
        beta = np.asarray(fitted.params, dtype=float)
        chance = expit(design @ beta)
        likelihood = float(np.sum(observed * np.log(chance) + (1 - observed) * np.log1p(-chance)))
        information = design.T @ (design * (chance * (1 - chance))[:, None])
        standard_errors = np.sqrt(np.diag(np.linalg.inv(information)))
        checked_beta, checked_errors, checked_likelihood = newton_reference(design, observed)
        native = [*beta, *standard_errors, likelihood]
        checked = [*checked_beta, *checked_errors, checked_likelihood]
        maximum_difference = max(abs(float(a) - float(b)) for a, b in zip(native, checked, strict=True))
        if maximum_difference > 1e-8:
            raise ValueError(f"Independent {model} Newton fit differs")
        if not np.allclose(standard_errors, fitted.bse, atol=1e-8, rtol=1e-8):
            raise ValueError(f"Native {model} information differs")
        if not math.isclose(likelihood, float(fitted.llf), abs_tol=1e-8):
            raise ValueError(f"Native {model} log likelihood differs")
        score_max = float(np.max(np.abs(design.T @ (observed - chance))))
        if score_max > 1e-7:
            raise ValueError(f"Native {model} score not stationary")
        aic = 4 - 2 * likelihood
        if not math.isclose(aic, float(fitted.aic), abs_tol=1e-8):
            raise ValueError(f"Native {model} AIC differs")
        models[model] = {
            "predictor": predictor,
            "beta": beta.tolist(),
            "aic": aic,
            "maximum_independent_difference": maximum_difference,
            "score_max": score_max,
        }
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
        for term, coefficient, error in zip(("intercept", predictor), beta, standard_errors, strict=True):
            z_value = coefficient / error
            coefficients.append(
                {
                    "id": f"{model}/{term}",
                    "model": model,
                    "term": term,
                    "coefficient": float(coefficient),
                    "standard_error": float(error),
                    "z": float(z_value),
                    "pvalue": float(2 * norm.sf(abs(z_value))),
                    "odds_ratio": math.exp(coefficient),
                }
            )
    return models, coefficients, fits


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--trial-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["recipe_id"] != "real-clinical-univariable-logit-audit":
        raise ValueError("Wrong frozen univariable protocol")
    frame = original_frame(args.source_archive, protocol)
    rows = observed_rows(args.trial_asset, frame, protocol)
    cohort, complete, response = cohort_rows(rows, protocol)
    models, coefficients, fits = fit_models(complete, response, protocol)
    predictions = []
    for model, fitted in models.items():
        predictor = fitted["predictor"]
        for profile in protocol["profiles"][model]:
            value = float(profile[predictor])
            predictions.append(
                {
                    "id": f"{model}/{profile['id']}",
                    "model": model,
                    "profile": profile["id"],
                    "predictor": predictor,
                    "predictor_value": value,
                    "probability_no_pain": float(expit(fitted["beta"][0] + fitted["beta"][1] * value)),
                }
            )
    tables = {
        "cohort.tsv": cohort,
        "coefficients.tsv": coefficients,
        "model_fit.tsv": fits,
        "predictions.tsv": predictions,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in tables.items():
        write_table(args.output / name, data)
    answer = {
        "patients": len(rows),
        "complete_cases": len(complete),
        "no_pain_responses": sum(response),
        "age_only_aic": models["age_only"]["aic"],
        "bmi_only_aic": models["bmi_only"]["aic"],
        "age_only_age_coefficient": models["age_only"]["beta"][1],
        "bmi_only_bmi_coefficient": models["bmi_only"]["beta"][1],
    }
    reference = {
        "source_package": "medicaldata 0.2.0",
        "rda_sha256": protocol["original_rda_sha256"],
        "protocol_sha256": sha256(args.protocol),
        "trial_sha256": protocol["solver_input_content_sha256"],
        "model_terms": {name: model["terms"] for name, model in protocol["models"].items()},
        "profiles": protocol["profiles"],
        "answer": answer,
        "tables": {
            name: {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in data}
            for name, data in tables.items()
        },
        "artifact_sha256": {name: sha256(args.output / name) for name in tables},
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyreadr", "scipy", "statsmodels")
        },
    }
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    checks = {
        "status": "passed",
        "protocol_sha256": sha256(args.protocol),
        "source_archive_sha256": sha256(args.source_archive),
        "trial_content_sha256": protocol["solver_input_content_sha256"],
        "cohort_rows": len(cohort),
        "complete_case_rows": len(complete),
        "coefficient_rows": len(coefficients),
        "fit_rows": len(fits),
        "prediction_rows": len(predictions),
        "maximum_independent_difference": max(model["maximum_independent_difference"] for model in models.values()),
        "maximum_score_component": max(model["score_max"] for model in models.values()),
        "package_versions": reference["package_versions"],
    }
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "patients": len(rows), "models": list(models)}))


if __name__ == "__main__":
    main()
