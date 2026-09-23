# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small statistical tasks with exact or planted-design reference results."""

import math
import random
from fractions import Fraction
from functools import partial
from itertools import combinations

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_statistics(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "enrichment-universe":
        universe = [f"g{i}" for i in range(10)]
        query = set(rng.sample(universe, 3))
        terms = {"specific": set(query), "outside": {"outside1", "outside2"}}
        terms.update({f"term{i}": set(rng.sample(universe, rng.randint(2, 7))) | {"outside1"} for i in range(4)})
        inputs = {
            "universe.txt": "\n".join(universe) + "\n",
            "query.txt": "\n".join(sorted(query)) + "\noutside1\n",
            "terms.gmt": "\n".join("\t".join([name, "na", *sorted(genes)]) for name, genes in terms.items()) + "\n",
        }
        draws = [set(draw) for draw in combinations(universe, 3)]
        pvalues = {}
        for name, genes in terms.items():
            eligible = genes & set(universe)
            if len(eligible) < 2:
                continue
            overlap = len(query & eligible)
            pvalues[name] = Fraction(sum(len(draw & eligible) >= overlap for draw in draws), len(draws))
            expected[name] = {"overlap": overlap, "term_size": len(eligible), "pvalue": float(pvalues[name])}
        ordered = sorted(pvalues, key=pvalues.get)
        for i, name in enumerate(ordered):
            adjusted = min(
                Fraction(1), *(pvalues[other] * len(ordered) / rank for rank, other in enumerate(ordered, 1) if rank > i)
            )
            expected[name]["padj"] = float(adjusted)
        columns = {
            "overlap": Column(kind="integer", unit="genes", description="query genes in term and universe"),
            "term_size": Column(kind="integer", unit="genes", description="term size after universe restriction"),
            "pvalue": Column(
                kind="number",
                unit="probability",
                description="one-sided hypergeometric enrichment tail",
                atol=1e-10,
                rtol=1e-8,
            ),
            "padj": Column(
                kind="number",
                unit="probability",
                description="BH correction across all eligible terms",
                atol=1e-10,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Perform overrepresentation analysis of query.txt in terms.gmt using only genes in "
            "universe.txt. Treat lists as sets; restrict both query and terms to the universe. Test "
            "every term with at least two universe genes, including zero-overlap terms. Compute P(X >= "
            "observed overlap) under sampling without replacement and Benjamini-Hochberg adjusted "
            "p-values across all tested terms. Use term name as id."
        )
        mutation = {"reported_uncorrected_pvalues": [{"id": k, **v, "padj": v["pvalue"]} for k, v in expected.items()]}
    elif operation == "design-estimability":
        rows = []
        kinds = rng.sample(["full", "confounded", "constant"], 3)
        for i, kind in enumerate(kinds):
            replicates = rng.randint(2, 4)
            pairs = (
                [(0, 0), (0, 1), (1, 0), (1, 1)]
                if kind == "full"
                else [(0, 0), (1, 1)] if kind == "confounded" else [(0, 0), (0, 1)]
            )
            for treatment, batch in pairs:
                rows.extend(
                    [
                        {"cohort": f"c{i}", "patient": f"p{len(rows)+j}", "treatment": treatment, "batch": batch}
                        for j in range(replicates)
                    ]
                )
            expected[f"c{i}"] = {
                "rank": 3 if kind == "full" else 2,
                "estimable": int(kind == "full"),
                "n_patients": len(pairs) * replicates,
            }
        rng.shuffle(rows)
        inputs = {"design.csv": csv_text(rows)}
        columns = {
            "rank": Column(kind="integer", unit="design columns", description="rank of [intercept,treatment,batch]"),
            "estimable": Column(
                kind="integer", unit="decision", description="1 if treatment coefficient is uniquely estimable"
            ),
            "n_patients": Column(kind="integer", unit="patients", description="cohort row count"),
        }
        prompt = (
            "For each cohort in design.csv, assess the specified linear design with intercept, numeric "
            "treatment and numeric batch columns. Report matrix rank and whether the treatment "
            "coefficient is estimable from that design. Do not drop batch or change the contrast to "
            "force estimability. No outcome data are needed. Use cohort as id."
        )
        mutation = {"dropped_batch_to_force_fit": [{"id": k, **v, "estimable": 1} for k, v in expected.items()]}
    elif operation == "paired-treatment-effect":
        rows = []
        changes = []
        for patient in range(5):
            base = rng.randint(10, 30)
            change = rng.randint(2, 8)
            if patient < 4:
                changes.append(change)
            for visit, value in [("baseline", base), ("post", base + change)]:
                if patient == 4 and visit == "post":
                    continue
                noise = rng.randint(1, 3)
                for replicate, measurement in enumerate([value - noise, value + noise]):
                    rows.append({"patient": f"p{patient}", "visit": visit, "replicate": replicate, "value": measurement})
        rng.shuffle(rows)
        inputs = {"measurements.csv": csv_text(rows)}
        expected = {"cohort": {"mean_delta": sum(changes) / len(changes), "n_pairs": len(changes)}}
        columns = {
            "mean_delta": Column(
                kind="number",
                unit="measurement units",
                description="mean patient post-minus-baseline difference",
                atol=1e-10,
                rtol=1e-8,
            ),
            "n_pairs": Column(kind="integer", unit="patients", description="patients with both visits"),
        }
        prompt = (
            "Estimate the mean paired treatment change in measurements.csv. Average technical "
            "replicates within patient and visit first; include only patients with both baseline and "
            "post. Average post-minus-baseline differences with equal patient weight. Return id=cohort."
        )
        mutation = {
            "reversed_treatment_direction": [
                {"id": "cohort", "mean_delta": -expected["cohort"]["mean_delta"], "n_pairs": 4}
            ]
        }
    elif operation == "adjusted-linear-effect":
        intercept, treatment_effect, batch_effect = rng.randint(5, 12), rng.randint(2, 6), rng.randint(-6, -2)
        rows = []
        for treatment, batch, repeats in [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 2)]:
            for _ in range(repeats):
                noise = rng.randint(1, 3)
                for residual in [-noise, noise]:
                    rows.append(
                        {
                            "patient": f"p{len(rows)}",
                            "treatment": treatment,
                            "batch": batch,
                            "outcome": intercept + treatment_effect * treatment + batch_effect * batch + residual,
                        }
                    )
        rng.shuffle(rows)
        inputs = {"patients.csv": csv_text(rows)}
        expected = {
            "cohort": {
                "treatment_effect": float(treatment_effect),
                "batch_effect": float(batch_effect),
                "n_patients": len(rows),
            }
        }
        columns = {
            "treatment_effect": Column(
                kind="number",
                unit="outcome units",
                description="OLS treatment coefficient adjusted for batch",
                atol=1e-8,
                rtol=1e-8,
            ),
            "batch_effect": Column(
                kind="number", unit="outcome units", description="OLS numeric batch coefficient", atol=1e-8, rtol=1e-8
            ),
            "n_patients": Column(kind="integer", unit="patients", description="included rows"),
        }
        prompt = (
            "Fit unweighted ordinary least squares to patients.csv: outcome ~ intercept + treatment + "
            "batch, with numeric 0/1 predictors. Report treatment and batch coefficients and included "
            "row count. Use all supplied patients and id=cohort."
        )
        mutation = {
            "ignored_confounding": [
                {"id": "cohort", **expected["cohort"], "treatment_effect": float(treatment_effect) + batch_effect / 3}
            ]
        }
    elif operation == "odds-ratio-contingency":
        rows = []
        for i in range(3):
            a, b, c, d = [rng.randint(2, 12) for _ in range(4)]
            if i == 2:
                a = 0
            rows.append(
                {"study": f"s{i}", "exposed_case": a, "exposed_control": b, "unexposed_case": c, "unexposed_control": d}
            )
            correction = Fraction(1, 2) if min(a, b, c, d) == 0 else 0
            ratio = (a + correction) * (d + correction) / ((b + correction) * (c + correction))
            expected[f"s{i}"] = {"odds_ratio": float(ratio), "corrected": int(bool(correction)), "n": a + b + c + d}
        inputs = {"counts.csv": csv_text(rows)}
        columns = {
            "odds_ratio": Column(
                kind="number", unit="ratio", description="exposed versus unexposed disease odds", atol=1e-10, rtol=1e-8
            ),
            "corrected": Column(kind="integer", unit="decision", description="1 when 0.5 was added to all four cells"),
            "n": Column(kind="integer", unit="patients", description="original total before correction"),
        }
        prompt = (
            "Compute exposed-versus-unexposed disease odds ratios for counts.csv. If any of the four "
            "cells is zero, add 0.5 to every cell for the odds ratio only. Report whether correction "
            "was used and the original patient total. Use study as id."
        )
        mutation = {
            "reversed_exposure_contrast": [
                {"id": k, **v, "odds_ratio": 1 / v["odds_ratio"]} for k, v in expected.items()
            ]
        }
    elif operation == "kaplan-meier":
        schedule = [(rng.randint(1, 3), rng.randint(1, 3), 1), (5, rng.randint(1, 2), 2), (9, 1, 1)]
        risk = sum(events + censors for _, events, censors in schedule)
        survival = Fraction(1)
        rows = []
        for time, events, censors in schedule:
            survival *= Fraction(risk - events, risk)
            expected[str(time)] = {"at_risk": risk, "events": events, "survival": float(survival)}
            for event, count in [(1, events), (0, censors)]:
                rows.extend([{"patient": f"p{len(rows)+i}", "time": time, "event": event} for i in range(count)])
            risk -= events + censors
        rng.shuffle(rows)
        inputs = {"survival.csv": csv_text(rows)}
        columns = {
            "at_risk": Column(kind="integer", unit="patients", description="risk set immediately before time"),
            "events": Column(kind="integer", unit="deaths", description="event=1 records at time"),
            "survival": Column(
                kind="number",
                unit="probability",
                description="Kaplan-Meier survival immediately after deaths",
                atol=1e-10,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Compute Kaplan-Meier survival at each distinct observed time in survival.csv. event=1 is "
            "death and event=0 is right censoring. Deaths at a time occur before censor removals at "
            "that same time. Report the pre-event risk set, death count, and post-event survival. Use "
            "integer time rendered as a string as id."
        )
        mutation = {
            "treated_censoring_as_deaths": [{"id": k, **v, "events": v["events"] + 1} for k, v in expected.items()]
        }
    elif operation == "diagnostic-thresholds":
        bins = [(0.2, 2, 3), (0.5, rng.randint(1, 4), rng.randint(1, 4)), (0.8, 3, 1)]
        rows = []
        for score, positive, negative in bins:
            for label, count in [(1, positive), (0, negative)]:
                rows.extend([{"sample": f"s{len(rows)+i}", "label": label, "score": score} for i in range(count)])
        thresholds = [0.2, 0.5, 0.8, 1.0]
        inputs = {"predictions.csv": csv_text(rows), "thresholds.txt": "\n".join(map(str, thresholds)) + "\n"}
        positive_total = sum(p for _, p, _ in bins)
        negative_total = sum(n for _, _, n in bins)
        for threshold in thresholds:
            tp = sum(p for score, p, _ in bins if score >= threshold)
            fp = sum(n for score, _, n in bins if score >= threshold)
            expected[str(threshold)] = {
                "tp": tp,
                "fp": fp,
                "tn": negative_total - fp,
                "fn": positive_total - tp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / positive_total,
            }
        columns = {
            name: Column(kind="integer", unit="samples", description=name.upper() + " count with label=1 positive")
            for name in ["tp", "fp", "tn", "fn"]
        }
        columns.update(
            {
                name: Column(
                    kind="number",
                    unit="proportion",
                    description=name + "; zero if denominator is zero",
                    atol=1e-10,
                    rtol=1e-8,
                )
                for name in ["precision", "recall"]
            }
        )
        prompt = (
            "For each threshold in thresholds.txt, classify score >= threshold as positive in "
            "predictions.csv; label=1 is the true positive class. Report confusion counts, precision "
            "and recall, using zero for a zero denominator. Use the threshold string from the file as "
            "id."
        )
        mutation = {
            "swapped_positive_class": [
                {"id": k, **v, "tp": v["tn"], "tn": v["tp"], "fp": v["fn"], "fn": v["fp"]} for k, v in expected.items()
            ]
        }
    else:
        assert operation == "permutation-mean-test"
        n, total_success = 4, rng.choice([3, 5])
        observed_success = rng.choice([max(0, total_success - 4), min(4, total_success)])
        a = [1] * observed_success + [0] * (n - observed_success)
        b = [1] * (total_success - observed_success) + [0] * (4 - total_success + observed_success)
        rows = [{"sample": f"s{i}", "group": "A" if i < 4 else "B", "value": v} for i, v in enumerate(a + b)]
        rng.shuffle(rows)
        observed = Fraction(sum(a), 4) - Fraction(sum(b), 4)
        extremes = 0
        for x in range(max(0, total_success - 4), min(4, total_success) + 1):
            if abs(Fraction(x, 4) - Fraction(total_success - x, 4)) >= abs(observed):
                extremes += math.comb(total_success, x) * math.comb(8 - total_success, 4 - x)
        expected = {
            "comparison": {
                "difference": float(observed),
                "pvalue": extremes / math.comb(8, 4),
                "permutations": math.comb(8, 4),
            }
        }
        inputs = {"groups.csv": csv_text(rows)}
        columns = {
            "difference": Column(
                kind="number", unit="outcome units", description="observed mean(A)-mean(B)", atol=1e-10, rtol=1e-8
            ),
            "pvalue": Column(
                kind="number",
                unit="probability",
                description="exact two-sided permutation probability including ties",
                atol=1e-10,
                rtol=1e-8,
            ),
            "permutations": Column(
                kind="integer", unit="label assignments", description="number of distinct sample assignments"
            ),
        }
        prompt = (
            "Compute the exact two-sided permutation test of mean(A)-mean(B) for groups.csv, preserving"
            " the observed group sizes. Treat rows as distinct samples even when values match. "
            "Enumerate every label assignment; count absolute differences >= the observed absolute "
            "difference, including ties and the observed assignment. No Monte Carlo or plus-one "
            "correction. Return id=comparison."
        )
        mutation = {
            "one_sided_tail": [
                {"id": "comparison", **expected["comparison"], "pvalue": expected["comparison"]["pvalue"] / 2}
            ]
        }
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), mutation
    )


SKILLS = {
    "enrichment-universe": ("background-universe", "hypergeometric-tail", "multiple-testing"),
    "design-estimability": ("model-rank", "confounding", "valid-stopping"),
    "paired-treatment-effect": ("paired-design", "technical-replicates", "missing-visits"),
    "adjusted-linear-effect": ("covariate-adjustment", "signed-effects", "ols"),
    "odds-ratio-contingency": ("effect-direction", "zero-cells", "odds-ratio"),
    "kaplan-meier": ("risk-sets", "right-censoring", "tied-events"),
    "diagnostic-thresholds": ("positive-class", "threshold-boundaries", "precision-recall"),
    "permutation-mean-test": ("exchangeability", "exact-permutation", "two-sided-tests"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        ("gmt", "gene-lists") if name == "enrichment-universe" else ("csv-header",),
        ("https://www.statsmodels.org/stable/examples/index.html",),
        partial(generate_statistics, operation=name),
    )
    for name, skills in SKILLS.items()
)
