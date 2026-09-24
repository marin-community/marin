# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cohort, survival and repeated-measure tasks on published PBC observations."""

import csv
import io
import json
import math
import random
from functools import cache, partial

import numpy as np
from scipy.stats import CensoredData, ecdf

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.real_data import source_text, tsv_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, Recipe

SOURCE_ID = "survival:PBC"
SOURCE_URL = "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbc.html"
NAMES = ("real-clinical-kaplan-meier", "real-clinical-adjusted-cox", "real-clinical-paired-visits")


@cache
def observations(name: str) -> tuple[dict, ...]:
    return tuple(csv.DictReader(io.StringIO(source_text(SOURCE_ID, f"pbc-{name}.tsv.gz")), delimiter="\t"))


def integer(description: str) -> Column:
    return Column(kind="integer", description=description, unit="patients")


def number(description: str, unit: str) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-8, rtol=1e-7)


def generate_clinical(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    treatment = rng.choice(["all", "1", "2"])
    query = {"treatment": treatment, "analysis": operation}
    rows = observations("pbc")
    selected = [r for r in rows if r["trt"] and (treatment == "all" or r["trt"] == treatment)]
    patients = [{k: v for k, v in row.items() if k not in {"time", "status"}} for row in rows]
    outcomes = [{k: row[k] for k in ("id", "time", "status")} for row in rows]
    rng.shuffle(outcomes)
    inputs = {"patients.tsv": tsv_text(patients), "outcomes.tsv": tsv_text(outcomes)}
    prompt = (
        "Analyze the published Mayo PBC observations in /app/inputs. patients.tsv contains 418 baseline records, "
        "including non-randomized participants. "
        "Exclude participants with missing trt; trt=1 is D-penicillamine and trt=2 is placebo. "
        "Then select query.json's treatment ('all' retains both randomized groups). Empty fields are missing, "
        "not zero. Preserve observed values and do not impute. "
    )
    if operation.endswith("kaplan-meier"):
        horizons = sorted(rng.sample([180, 365, 730, 1095, 1460, 1825, 2555, 3650], 4))
        query["horizons_days"] = horizons
        times = np.array([int(r["time"]) for r in selected])
        events = np.array([int(r["status"]) != 0 for r in selected])
        curve = ecdf(CensoredData(uncensored=times[events], right=times[~events])).sf
        expected = {
            str(day): {
                "patients": len(selected),
                "at_risk": int((times >= day).sum()),
                "events": int(((times <= day) & events).sum()),
                "censored": int(((times <= day) & ~events).sum()),
                "survival": float(curve.evaluate(day)),
            }
            for day in horizons
        }
        columns = {
            key: integer(description)
            for key, description in {
                "patients": "eligible randomized participants",
                "at_risk": "participants observed through the horizon, including events at that day",
                "events": "composite events on or before the horizon",
                "censored": "censoring observations on or before the horizon",
            }.items()
        }
        columns["survival"] = number("right-continuous Kaplan-Meier event-free estimate", "probability")
        prompt += (
            "Join outcomes.tsv by id, never row position. time is days; status=0 means censored, "
            "1 transplant, 2 death. Use a composite endpoint: "
            "death OR transplant is an event. Estimate event-free survival by Kaplan-Meier; at tied times "
            "count events against the risk set before removing same-day censored participants. For each "
            "horizon in query.json, use its decimal day as id and report the eligible patient count, "
            "at_risk immediately before that day, cumulative events and censoring through that day, "
            "and survival after any events that day."
        )
        wrong = [
            {"id": key, **value, "survival": 1 - value["events"] / value["patients"]} for key, value in expected.items()
        ]
        mutations = {"ignored_censoring": wrong}
    elif operation.endswith("adjusted-cox"):
        biomarker = rng.choice(["bili", "albumin", "protime"])
        query["biomarker"] = biomarker
        reference = json.loads(source_text(SOURCE_ID, "pbc-cox-reference.json.gz"))["models"][f"{treatment}/{biomarker}"]
        expected = {
            term: {
                "coefficient": coefficient,
                "standard_error": se,
                "hazard_ratio": math.exp(coefficient),
                "pvalue": pvalue,
                "patients": reference["patients"],
                "events": reference["events"],
            }
            for term, coefficient, se, pvalue in zip(
                ["biomarker", "age_decades", "male"],
                reference["coefficients"],
                reference["standard_errors"],
                reference["pvalues"],
                strict=True,
            )
        }
        columns = {
            "coefficient": number("unpenalized Cox coefficient", "log hazard ratio"),
            "standard_error": number("model-based inverse-information standard error", "log hazard ratio"),
            "hazard_ratio": number("exponential of coefficient", "ratio"),
            "pvalue": number("two-sided normal Wald probability", "probability"),
            "patients": integer("complete-case eligible participants"),
            "events": integer("observed deaths or transplants in complete cases"),
        }
        prompt += (
            "Join outcomes.tsv by id, never row position. Fit an unpenalized Cox proportional-hazards "
            "model with Breslow ties, no intercept, and "
            "model-based standard errors. time is follow-up in days; status=1 or 2 is the composite "
            "death/transplant event and status=0 is right censoring. Use only complete cases for time, "
            "status, age, sex and the selected biomarker. Covariates are: biomarker = natural log of bili "
            "or protime, but untransformed albumin; age_decades = age/10; male = 1 for sex=m and 0 for sex=f. "
            "Fit all three covariates together and report one record for each named term, with coefficient, "
            "standard_error, hazard_ratio, two-sided normal Wald pvalue, patients and events. Do not use "
            "robust/sandwich variance or penalization. This estimates adjusted associations under the stated "
            "model, not treatment benefit or a clinical prediction recommendation."
        )
        mutations = {
            "reversed_effect": [
                {"id": key, **value, "coefficient": -value["coefficient"], "hazard_ratio": 1 / value["hazard_ratio"]}
                for key, value in expected.items()
            ],
            "censoring_counted_as_event": [
                {"id": key, **value, "events": value["patients"]} for key, value in expected.items()
            ],
        }
    else:
        target, width = rng.choice([(180, 90), (365, 120), (730, 180)])
        query.update(target_day=target, window_days=width)
        inputs = {
            "patients.tsv": tsv_text([{key: row[key] for key in ("id", "trt", "sex")} for row in rows]),
            "visits.tsv": source_text(SOURCE_ID, "pbc-pbcseq.tsv.gz"),
        }
        visits = observations("pbcseq")
        eligible = {row["id"] for row in selected}
        panel = rng.sample(["bili", "albumin", "chol", "alk.phos", "ast", "platelet", "protime"], 4)
        query["biomarkers"] = panel
        expected = {}
        for marker in panel:
            differences = []
            for identifier in sorted(eligible):
                subject = [row for row in visits if row["id"] == identifier]
                baseline = next((row for row in subject if row["day"] == "0"), None)
                candidates = [row for row in subject if int(row["day"]) > 0 and abs(int(row["day"]) - target) <= width]
                if baseline is None or not candidates:
                    continue
                nearest = min(candidates, key=lambda row: (abs(int(row["day"]) - target), int(row["day"])))
                if baseline[marker] and nearest[marker]:
                    differences.append(float(nearest[marker]) - float(baseline[marker]))
            expected[marker] = {
                "pairs": len(differences),
                "mean_change": float(np.mean(differences)),
                "median_change": float(np.median(differences)),
            }
        columns = {
            "pairs": integer("subjects with both measurements at the selected visits"),
            "mean_change": number("mean follow-up minus baseline", "source measurement units"),
            "median_change": number("median follow-up minus baseline", "source measurement units"),
        }
        prompt += (
            "For this task patients.tsv supplies only cohort membership; visits.tsv contains 1,945 real "
            "longitudinal records. Use each subject's day=0 row in visits.tsv as baseline: this table contains "
            "corrected measurements and must not be mixed with the separate baseline study table. Select "
            "the one positive-day visit nearest target_day within the inclusive window_days tolerance, "
            "breaking distance ties toward the earlier day. Select the visit BEFORE checking biomarker "
            "missingness; do not substitute a more distant complete visit. For each requested biomarker, "
            "retain subjects measured at both selected visits and compute paired follow-up minus baseline "
            "in the original dataset units. Report pairs, mean_change and median_change with biomarker "
            "name as id. Keep patients equally weighted, regardless of their number of visits. Missingness "
            "may be informative; these are descriptive complete-pair summaries, not causal treatment effects."
        )
        mutations = {
            "reversed_paired_change": [
                {"id": key, **value, "mean_change": -value["mean_change"], "median_change": -value["median_change"]}
                for key, value in expected.items()
            ]
        }
    inputs["query.json"] = json.dumps(query) + "\n"
    return Instance(
        prompt,
        inputs,
        Contract(columns=columns, expected=expected),
        mutations,
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE_ID,),
        derivation="Published observations unchanged; identifier joins, randomized-cohort selection and "
        "explicit analysis queries. No generated patients, measurement imputation or outcome relabeling.",
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        (
            "patient-identity",
            "cohort-eligibility",
            "missingness",
            "censoring" if "paired" not in name else "paired-visits",
        ),
        ("clinical-tsv", "longitudinal-tsv" if "paired" in name else "survival-tsv"),
        (SOURCE_URL,),
        partial(generate_clinical, operation=name),
    )
    for name in NAMES
)
