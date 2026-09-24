# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent clinical-table solutions using only public inputs and the standard library."""

import json
import math
import statistics
from collections import defaultdict
from functools import partial
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import table
from experiments.post_training.bio_tasks.solvers.statistics import reduced_rows


def cox_derivatives(
    design: list[list[float]], times: list[int], events: list[bool], beta: list[float]
) -> tuple[list[float], list[list[float]]]:
    """Evaluate the Breslow partial-likelihood score and observed information."""
    dimension = len(beta)
    eta = [sum(a * b for a, b in zip(row, beta, strict=True)) for row in design]
    offset = max(eta)
    weights = [math.exp(value - offset) for value in eta]
    score = [0.0] * dimension
    information = [[0.0] * dimension for _ in beta]
    for time in sorted({time for time, event in zip(times, events, strict=True) if event}):
        deaths = [i for i, value in enumerate(times) if value == time and events[i]]
        risk = [i for i, value in enumerate(times) if value >= time]
        total = math.fsum(weights[i] for i in risk)
        means = [math.fsum(weights[i] * design[i][j] for i in risk) / total for j in range(dimension)]
        for j in range(dimension):
            score[j] += math.fsum(design[i][j] for i in deaths) - len(deaths) * means[j]
            for k in range(dimension):
                moment = math.fsum(weights[i] * design[i][j] * design[i][k] for i in risk) / total
                information[j][k] += len(deaths) * (moment - means[j] * means[k])
    return score, information


def cox_fit(design: list[list[float]], times: list[int], events: list[bool]) -> tuple[list[float], list[float]]:
    """Fit an unpenalized Breslow Cox model and return coefficients and standard errors."""
    dimension = len(design[0])
    means = [statistics.mean(row[j] for row in design) for j in range(dimension)]
    centered = [[value - mean for value, mean in zip(row, means, strict=True)] for row in design]
    beta = [0.0] * dimension
    for _ in range(50):
        score, information = cox_derivatives(centered, times, events, beta)
        solution = reduced_rows([[*row, score[j]] for j, row in enumerate(information)])
        step = [float(row[-1]) for row in solution]
        beta = [value + change for value, change in zip(beta, step, strict=True)]
        if max(map(abs, step)) < 1e-10:
            break
    else:
        raise ValueError("Cox reference solution did not converge")
    _, information = cox_derivatives(centered, times, events, beta)
    inverse = reduced_rows([row + [int(j == k) for k in range(dimension)] for j, row in enumerate(information)])
    return beta, [math.sqrt(float(row[dimension + j])) for j, row in enumerate(inverse)]


def solve_clinical(inputs: Path, operation: str) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    patients = {
        row["id"]: row
        for row in table(inputs / "patients.tsv", delimiter="\t")
        if row["trt"] and (query["treatment"] == "all" or row["trt"] == query["treatment"])
    }
    if operation.endswith("paired-visits"):
        visits = defaultdict(list)
        for row in table(inputs / "visits.tsv", delimiter="\t"):
            if row["id"] in patients:
                visits[row["id"]].append(row)
        deltas = defaultdict(list)
        for rows in visits.values():
            by_day = {int(row["day"]): row for row in rows}
            days = sorted(day for day in by_day if day > 0 and abs(day - query["target_day"]) <= query["window_days"])
            if 0 not in by_day or not days:
                continue
            nearest = min(days, key=lambda day: (abs(day - query["target_day"]), day))
            for biomarker in query["biomarkers"]:
                before, after = by_day[0][biomarker], by_day[nearest][biomarker]
                if before and after:
                    deltas[biomarker].append(float(after) - float(before))
        return [
            {
                "id": key,
                "pairs": len(deltas[key]),
                "mean_change": statistics.mean(deltas[key]),
                "median_change": statistics.median(deltas[key]),
            }
            for key in query["biomarkers"]
        ]
    outcomes = {row["id"]: row for row in table(inputs / "outcomes.tsv", delimiter="\t")}
    rows = [{**row, **outcomes[identifier]} for identifier, row in patients.items()]
    if operation.endswith("kaplan-meier"):
        times = [int(row["time"]) for row in rows]
        events = [row["status"] != "0" for row in rows]
        answer = []
        for horizon in query["horizons_days"]:
            survival = 1.0
            for time in sorted(set(times)):
                if time > horizon:
                    break
                deaths = sum(t == time and event for t, event in zip(times, events, strict=True))
                survival *= 1 - deaths / sum(t >= time for t in times)
            answer.append(
                {
                    "id": str(horizon),
                    "patients": len(rows),
                    "at_risk": sum(t >= horizon for t in times),
                    "events": sum(t <= horizon and event for t, event in zip(times, events, strict=True)),
                    "censored": sum(t <= horizon and not event for t, event in zip(times, events, strict=True)),
                    "survival": survival,
                }
            )
        return answer
    biomarker = query["biomarker"]
    complete = [row for row in rows if all(row[key] for key in ["time", "status", "age", "sex", biomarker])]
    design = [
        [
            float(row[biomarker]) if biomarker == "albumin" else math.log(float(row[biomarker])),
            float(row["age"]) / 10,
            float(row["sex"] == "m"),
        ]
        for row in complete
    ]
    times = [int(row["time"]) for row in complete]
    events = [row["status"] != "0" for row in complete]
    coefficients, standard_errors = cox_fit(design, times, events)
    return [
        {
            "id": name,
            "coefficient": beta,
            "standard_error": se,
            "hazard_ratio": math.exp(beta),
            "pvalue": math.erfc(abs(beta / se) / math.sqrt(2)),
            "patients": len(complete),
            "events": sum(events),
        }
        for name, beta, se in zip(["biomarker", "age_decades", "male"], coefficients, standard_errors, strict=True)
    ]


SOLVERS = {
    name: partial(solve_clinical, operation=name)
    for name in ("real-clinical-kaplan-meier", "real-clinical-adjusted-cox", "real-clinical-paired-visits")
}
