# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed trial: univariable BMI and age logistic-model audit."""

import hashlib
import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

NAME = "real-clinical-univariable-logit-audit"
SOURCE = "medicaldata:licorice_gargle"
REFERENCE_ASSET = "licorice-gargle-univariable-reference.json.gz"
TRIAL_ASSET = "licorice-gargle-trial.tsv.gz"


def integer(description: str, unit: str, nullable: bool = False) -> Column:
    return Column(kind="integer", description=description, unit=unit, nullable=nullable)


def number(description: str, unit: str) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-8, rtol=1e-7)


def text(description: str, unit: str) -> Column:
    return Column(kind="text", description=description, unit=unit)


TABLE_COLUMNS = {
    "cohort.tsv": {
        "included": integer("one if age, BMI and 30-minute pain score are observed", "indicator"),
        "no_pain_response": integer("one if observed throat-pain score is zero", "indicator", True),
        "exclusion_reason": text("ordered missing_<column> tokens, empty for included rows", "reason"),
    },
    "coefficients.tsv": {
        "model": text("age_only or bmi_only", "model"),
        "term": text("intercept or raw-unit predictor", "term"),
        "coefficient": number("unpenalized binomial-logit maximum-likelihood estimate", "log odds"),
        "standard_error": number("inverse observed-information standard error", "log odds"),
        "z": number("coefficient divided by standard error", "Wald statistic"),
        "pvalue": number("two-sided standard normal Wald probability", "probability"),
        "odds_ratio": number("exponential of the coefficient", "odds ratio"),
    },
    "model_fit.tsv": {
        "patients": integer("identical complete-case cohort size", "patients"),
        "responses": integer("observed complete-case no-pain events", "patients"),
        "parameters": integer("intercept and one predictor", "parameters"),
        "log_likelihood": number("maximized Bernoulli log likelihood", "log likelihood"),
        "aic": number("twice parameter count minus twice maximized log likelihood", "AIC"),
        "converged": integer("one for converged maximum likelihood", "indicator"),
    },
    "predictions.tsv": {
        "model": text("age_only or bmi_only", "model"),
        "profile": text("named fixed single-predictor profile", "profile"),
        "predictor": text("age_years or bmi_kg_m2", "predictor"),
        "predictor_value": number("raw predictor value from analysis.json", "years or kg/m^2"),
        "probability_no_pain": number("model-implied no-pain event probability", "probability"),
    },
}


def generate_clinical_univariable(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE, REFERENCE_ASSET))
    trial = source_text(SOURCE, TRIAL_ASSET)
    analysis = {
        "models": reference["model_terms"],
        "profiles": reference["profiles"],
        "outcome": "throat_pain_30min == 0",
        "complete_case_columns": ["age_years", "bmi_kg_m2", "throat_pain_30min"],
        "source_rda_sha256": reference["rda_sha256"],
    }
    if reference["trial_sha256"] != hashlib.sha256(trial.encode()).hexdigest():
        raise ValueError("Observed trial input differs from private univariable reference")
    # Native TSV uses NA; the nullable integer contract represents that value as None.
    reference["tables"]["cohort.tsv"] = {
        row_id: {**row, "no_pain_response": None if row["no_pain_response"] == "NA" else row["no_pain_response"]}
        for row_id, row in reference["tables"]["cohort.tsv"].items()
    }
    tables = {
        name: TableContract(columns=columns, expected=reference["tables"][name], max_bytes=64 * 1024)
        for name, columns in TABLE_COLUMNS.items()
    }
    return Instance(
        "Analyze all 235 observed participants in the licorice-gargle randomized trial using /app/inputs. "
        "trial.tsv retains source row order, with stable P0001-P0235 IDs, age_years, bmi_kg_m2, recorded_gender, "
        "gargle_arm and throat_pain_30min. Empty input fields are missing. Define a binary no-pain response "
        "as a recorded 30-minute throat-pain score of exactly zero; positive scores are pain. Use the "
        "identical complete-case cohort for both models, requiring only age_years, bmi_kg_m2 and "
        "throat_pain_30min. Exclude any row missing one of these fields without imputation. Write cohort.tsv "
        "for all 235 participants with included=0/1, no_pain_response=0/1 or NA when excluded, and "
        "exclusion_reason empty for included rows or comma-separated missing_<column> tokens in the "
        "declared required-column order for excluded rows. "
        "Fit exactly two unpenalized binomial-logit maximum-likelihood models: age_only has an intercept "
        "and the raw age_years value; bmi_only has an intercept and the raw bmi_kg_m2 value. Each is a "
        "single-predictor model: do not include gargle_arm, recorded_gender or any other covariate; do "
        "not center or scale the predictors. The age slope is log-odds change per one year and the BMI "
        "slope per one kg/m^2. Use inverse observed-information standard errors, normal two-sided Wald "
        "p-values, and exp(coefficient) odds ratios. Write coefficients.tsv with exactly four id=model/term "
        "rows. Write model_fit.tsv with id=model, the common cohort size and event count, two parameters, "
        "maximized Bernoulli log_likelihood, AIC=2k-2log_likelihood, and converged=1. The models are "
        "nonnested; do not perform a likelihood-ratio test between them. Use the six fixed profiles in "
        "analysis.json to write predictions.tsv with id=model/profile, predictor, raw predictor_value "
        "and fitted no-pain probability. Keep full numerical precision in the TSVs. Return one answer.json "
        "record with id=study: patients, complete_cases, no_pain_responses, age_only_aic, bmi_only_aic, "
        "age_only_age_coefficient and bmi_only_bmi_coefficient. This clinical endpoint is absence of a "
        "postoperative symptom at 30 minutes, not cancer treatment efficacy or chronic disease remission; "
        "the univariable coefficients are associations under this specification.",
        {"trial.tsv": trial, "analysis.json": json.dumps(analysis, indent=2) + "\n"},
        Contract(
            columns={
                "patients": integer("all observed trial participants", "patients"),
                "complete_cases": integer("same cohort for both univariable fits", "patients"),
                "no_pain_responses": integer("complete cases with score zero", "patients"),
                "age_only_aic": number("AIC for intercept-plus-age model", "AIC"),
                "bmi_only_aic": number("AIC for intercept-plus-BMI model", "AIC"),
                "age_only_age_coefficient": number("age-only log-odds slope per one year", "log odds/year"),
                "bmi_only_bmi_coefficient": number("BMI-only log-odds slope per one kg/m^2", "log odds/(kg/m^2)"),
            },
            expected={"study": reference["answer"]},
            tables=tables,
        ),
        {
            "wrong_bmi_only_aic": [
                {"id": "study", **reference["answer"], "bmi_only_aic": reference["answer"]["bmi_only_aic"] + 1}
            ],
            "wrong_age_slope": [
                {
                    "id": "study",
                    **reference["answer"],
                    "age_only_age_coefficient": reference["answer"]["age_only_age_coefficient"] + 1,
                }
            ],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "All original medicaldata 0.2.0 trial rows remain in source order. The source RDA and "
            "separate CSV conversion agree on all original fields; two univariable binomial-logit "
            "fits and all artifacts receive pinned-package and independent Newton checks."
        ),
    )


RECIPES = (
    Recipe(
        id=NAME,
        version="1",
        skills=(
            "clinical cohort missingness",
            "single-predictor logistic regression",
            "AIC",
            "per-unit age and BMI coefficients",
            "fixed-profile probabilities",
        ),
        formats=("clinical TSV", "JSON"),
        sources=("https://cran.r-project.org/package=medicaldata",),
        generate=generate_clinical_univariable,
        oracle_timeout=180,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
