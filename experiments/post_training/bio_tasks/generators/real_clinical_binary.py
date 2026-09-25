# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed licorice-gargle trial: fixed binary response and logistic-model audit."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

NAME = "real-clinical-binary-response"
SOURCE = "medicaldata:licorice_gargle"
REFERENCE_ASSET = "licorice-gargle-binary-reference.json.gz"
TRIAL_ASSET = "licorice-gargle-trial.tsv.gz"


def integer(description: str, unit: str, nullable: bool = False) -> Column:
    return Column(kind="integer", description=description, unit=unit, nullable=nullable)


def number(description: str, unit: str) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-8, rtol=1e-7)


def text(description: str, unit: str) -> Column:
    return Column(kind="text", description=description, unit=unit)


TABLE_COLUMNS = {
    "cohort.tsv": {
        "included": integer("one for a participant with all five required observations", "indicator"),
        "no_pain_response": integer("one for zero reported throat pain at 30 minutes", "indicator", True),
        "exclusion_reason": text("empty if included, otherwise missing_30min_outcome", "reason"),
    },
    "coefficients.tsv": {
        "model": text("age, bmi, or joint prespecified model", "model"),
        "term": text("named model coefficient", "term"),
        "coefficient": number("unpenalized binomial-logit maximum likelihood estimate", "log odds"),
        "standard_error": number("model-based inverse observed-information standard error", "log odds"),
        "z": number("coefficient divided by standard error", "Wald statistic"),
        "pvalue": number("two-sided normal Wald probability", "probability"),
        "odds_ratio": number("exponential of the coefficient", "odds ratio"),
    },
    "model_fit.tsv": {
        "patients": integer("common complete-case participants", "patients"),
        "responses": integer("complete cases without throat pain at 30 minutes", "patients"),
        "parameters": integer("number of fitted coefficients including intercept", "parameters"),
        "log_likelihood": number("maximized Bernoulli log likelihood", "log likelihood"),
        "aic": number("2k minus twice maximized log likelihood", "AIC"),
        "converged": integer("one for converged maximum likelihood", "indicator"),
    },
    "comparisons.tsv": {
        "reduced": text("age or bmi reduced model", "model"),
        "full": text("joint nested model", "model"),
        "likelihood_ratio": number("twice full minus reduced log likelihood", "chi-square statistic"),
        "degrees_freedom": integer("one added covariate", "degrees of freedom"),
        "pvalue": number("upper-tail chi-square probability with one degree of freedom", "probability"),
        "delta_aic": number("joint AIC minus reduced AIC", "AIC difference"),
    },
    "predictions.tsv": {
        "model": text("age, bmi, or joint model", "model"),
        "profile": text("one of the eight fixed covariate profiles", "profile"),
        "age_years": integer("profile age", "years"),
        "bmi_kg_m2": integer("profile body mass index", "kg/m^2"),
        "recorded_gender": integer("source-coded gender; zero male, one female", "indicator"),
        "gargle_arm": integer("zero sugar-water, one licorice", "indicator"),
        "probability_no_pain": number("model-implied probability of no pain at 30 minutes", "probability"),
    },
}


def generate_clinical_binary(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE, REFERENCE_ASSET))
    trial = source_text(SOURCE, TRIAL_ASSET)
    policy = {
        "model_terms": reference["model_terms"],
        "profiles": reference["profiles"],
        "outcome": "throat_pain_30min == 0",
        "complete_case_columns": ["age_years", "bmi_kg_m2", "recorded_gender", "gargle_arm", "throat_pain_30min"],
        "source_rda_sha256": reference["rda_sha256"],
    }
    tables = {
        name: TableContract(columns=columns, expected=reference["tables"][name], max_bytes=64 * 1024)
        for name, columns in TABLE_COLUMNS.items()
    }
    return Instance(
        "Analyze the observed 235-participant licorice-gargle randomized trial in /app/inputs. "
        "trial.tsv keeps every participant in source row order, with stable P0001-P0235 IDs. It records "
        "age_years, bmi_kg_m2, recorded_gender (0=male, 1=female as coded by the source), gargle_arm "
        "(0=5 g sugar-water, 1=0.5 g licorice) and throat_pain_30min (0-10 recorded severity at rest "
        "30 minutes after PACU arrival). Empty input fields are missing. Define a binary no-pain response "
        "as an observed throat-pain score of exactly zero; positive scores are pain. Use the same complete "
        "cases for all models, excluding participants with any missing required field without imputation. "
        "Write cohort.tsv for all 235 participants with included=0/1, no_pain_response=0/1 or NA when "
        "excluded, and exclusion_reason empty or missing_30min_outcome. "
        "Fit three prespecified unpenalized binomial-logit maximum-likelihood models on the common cohort: "
        "age has intercept, licorice, female and age_per_10y=(age_years-60)/10; bmi has intercept, licorice, "
        "female and bmi_per_5kg_m2=(bmi_kg_m2-25)/5; joint has all five terms. No interactions, "
        "regularization or robust/sandwich standard errors. Use inverse observed-information standard errors "
        "and two-sided normal Wald probabilities. Write every coefficient to coefficients.tsv with id=model/term, "
        "coefficient, standard_error, z, pvalue and exp(coefficient) odds_ratio. Write model_fit.tsv with "
        "one row per model, id=model, complete-case patients/responses, parameters, maximized Bernoulli "
        "log_likelihood, AIC=2k-2log_likelihood, and converged=1. Compare age and bmi reduced models "
        "separately with their nested joint model: write comparisons.tsv with id=age_to_joint or bmi_to_joint, "
        "likelihood_ratio=2(LL_joint-LL_reduced), one degree of freedom, chi-square upper-tail pvalue and "
        "delta_aic=AIC_joint-AIC_reduced. Do not use a likelihood-ratio test directly between age and bmi. "
        "For each of the eight profiles in analysis.json, use every model to predict the no-pain probability "
        "and write predictions.tsv with id=model/profile and all profile covariates. Covariates absent from "
        "a reduced model do not enter its linear predictor. Keep full numerical precision in all TSVs. "
        "Return one answer.json record with id=study: patients, complete_cases, no_pain_responses, "
        "best_model_by_aic (smallest AIC, alphabetical name on ties), joint_licorice_odds_ratio, and the "
        "two age_to_joint and bmi_to_joint pvalues. Model coefficients describe adjusted associations "
        "under this specification; the binary endpoint is absence of a postoperative symptom, not chronic "
        "disease remission or evidence of treatment-response heterogeneity.",
        {"trial.tsv": trial, "analysis.json": json.dumps(policy, indent=2) + "\n"},
        Contract(
            columns={
                "patients": integer("all observed trial participants", "patients"),
                "complete_cases": integer("participants with all required observations", "patients"),
                "no_pain_responses": integer("complete cases with zero reported throat pain", "patients"),
                "best_model_by_aic": text("lowest-AIC model among age, bmi and joint", "model"),
                "joint_licorice_odds_ratio": number("joint-model licorice coefficient exponentiated", "odds ratio"),
                "age_to_joint_pvalue": number("nested age-to-joint likelihood-ratio pvalue", "probability"),
                "bmi_to_joint_pvalue": number("nested bmi-to-joint likelihood-ratio pvalue", "probability"),
            },
            expected={"study": reference["answer"]},
            tables=tables,
        ),
        {
            "changed_complete_case_count": [
                {"id": "study", **reference["answer"], "complete_cases": reference["answer"]["complete_cases"] + 2}
            ],
            "wrong_model_selection": [{"id": "study", **reference["answer"], "best_model_by_aic": "joint"}],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "All 235 rows of the original medicaldata 0.2.0 licorice-gargle RDA were preserved in source "
            "order; a separately published CSV conversion matched all 19 original fields and missingness. "
            "Five observed fields form the task input. A pinned statsmodels reference and independent "
            "pure-Python Newton fit establish every row-keyed artifact."
        ),
    )


RECIPES = (
    Recipe(
        id=NAME,
        version="1",
        skills=(
            "clinical cohort missingness",
            "binary logistic models",
            "AIC model comparison",
            "nested likelihood-ratio tests",
            "fixed-profile predictions",
        ),
        formats=("clinical TSV", "JSON"),
        sources=("https://cran.r-project.org/package=medicaldata",),
        generate=generate_clinical_binary,
        oracle_timeout=180,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
