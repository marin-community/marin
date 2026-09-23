# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-reading statistics with exact small-sample calculations."""

import math
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import tab_rows, table


def reduced_rows(matrix: list[list]) -> list[list[Fraction]]:
    rows = [[Fraction(x) for x in row] for row in matrix]
    pivot = 0
    for column in range(len(rows[0])):
        candidate = next((r for r in range(pivot, len(rows)) if rows[r][column]), None)
        if candidate is None:
            continue
        rows[pivot], rows[candidate] = rows[candidate], rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [x / scale for x in rows[pivot]]
        for r in range(len(rows)):
            if r != pivot:
                factor = rows[r][column]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[pivot], strict=True)]
        pivot += 1
        if pivot == len(rows):
            break
    return rows


def matrix_rank(matrix: list[list]) -> int:
    return sum(any(row) for row in reduced_rows(matrix))


def solve_estimability(inputs: Path) -> list[dict]:
    cohorts = defaultdict(list)
    for row in table(inputs / "design.csv"):
        cohorts[row["cohort"]].append([1, int(row["treatment"]), int(row["batch"])])
    return [
        {
            "id": name,
            "rank": matrix_rank(rows),
            "estimable": int(matrix_rank(rows) == matrix_rank([*rows, [0, 1, 0]])),
            "n_patients": len(rows),
        }
        for name, rows in cohorts.items()
    ]


def solve_adjusted(inputs: Path) -> list[dict]:
    rows = table(inputs / "patients.csv")
    x = [[1, int(row["treatment"]), int(row["batch"])] for row in rows]
    y = [int(row["outcome"]) for row in rows]
    normal = [
        [sum(row[i] * row[j] for row in x) for j in range(3)]
        + [sum(row[i] * value for row, value in zip(x, y, strict=True))]
        for i in range(3)
    ]
    coefficients = [row[-1] for row in reduced_rows(normal)]
    return [
        {
            "id": "cohort",
            "treatment_effect": float(coefficients[1]),
            "batch_effect": float(coefficients[2]),
            "n_patients": len(rows),
        }
    ]


def solve_enrichment(inputs: Path) -> list[dict]:
    universe = set((inputs / "universe.txt").read_text().splitlines())
    query = set((inputs / "query.txt").read_text().splitlines()) & universe
    answer = []
    population, draws = len(universe), len(query)
    for name, _, *genes in tab_rows(inputs / "terms.gmt"):
        eligible = set(genes) & universe
        size, overlap = len(eligible), len(eligible & query)
        if size < 2:
            continue
        numerator = sum(
            math.comb(size, x) * math.comb(population - size, draws - x)
            for x in range(overlap, min(size, draws) + 1)
            if 0 <= draws - x <= population - size
        )
        answer.append(
            {"id": name, "overlap": overlap, "term_size": size, "pvalue": numerator / math.comb(population, draws)}
        )
    ordered = sorted(answer, key=lambda row: row["pvalue"])
    bound = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        bound = min(bound, ordered[i]["pvalue"] * len(ordered) / (i + 1))
        ordered[i]["padj"] = bound
    return answer


def solve_paired(inputs: Path) -> list[dict]:
    measurements = defaultdict(lambda: defaultdict(list))
    for row in table(inputs / "measurements.csv"):
        measurements[row["patient"]][row["visit"]].append(float(row["value"]))
    differences = []
    for visits in measurements.values():
        if "baseline" in visits and "post" in visits:
            differences.append(
                sum(visits["post"]) / len(visits["post"]) - sum(visits["baseline"]) / len(visits["baseline"])
            )
    return [{"id": "cohort", "mean_delta": sum(differences) / len(differences), "n_pairs": len(differences)}]


def solve_odds(inputs: Path) -> list[dict]:
    answer = []
    for row in table(inputs / "counts.csv"):
        a, b, c, d = [
            int(row[name]) for name in ["exposed_case", "exposed_control", "unexposed_case", "unexposed_control"]
        ]
        total = a + b + c + d
        corrected = int(0 in [a, b, c, d])
        if corrected:
            a, b, c, d = [x + 0.5 for x in [a, b, c, d]]
        answer.append({"id": row["study"], "odds_ratio": (a / b) / (c / d), "corrected": corrected, "n": total})
    return answer


def solve_survival(inputs: Path) -> list[dict]:
    rows = table(inputs / "survival.csv")
    times = sorted({int(row["time"]) for row in rows})
    survival = 1.0
    answer = []
    for time in times:
        risk = sum(int(row["time"]) >= time for row in rows)
        deaths = sum(int(row["time"]) == time and row["event"] == "1" for row in rows)
        survival *= 1 - deaths / risk
        answer.append({"id": str(time), "at_risk": risk, "events": deaths, "survival": survival})
    return answer


def solve_thresholds(inputs: Path) -> list[dict]:
    rows = table(inputs / "predictions.csv")
    answer = []
    for text in (inputs / "thresholds.txt").read_text().splitlines():
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for row in rows:
            predicted = float(row["score"]) >= float(text)
            positive = row["label"] == "1"
            counts["tp" if predicted and positive else "fp" if predicted else "fn" if positive else "tn"] += 1
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        answer.append(
            {
                "id": text,
                **counts,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            }
        )
    return answer


def solve_permutation(inputs: Path) -> list[dict]:
    rows = table(inputs / "groups.csv")
    values = [Fraction(row["value"]) for row in rows]
    a = [Fraction(row["value"]) for row in rows if row["group"] == "A"]
    b = [Fraction(row["value"]) for row in rows if row["group"] == "B"]
    observed = sum(a) / len(a) - sum(b) / len(b)
    extreme = total = 0
    for chosen in combinations(range(len(rows)), len(a)):
        left = sum(values[i] for i in chosen)
        difference = left / len(a) - (sum(values) - left) / len(b)
        extreme += abs(difference) >= abs(observed)
        total += 1
    return [{"id": "comparison", "difference": float(observed), "pvalue": extreme / total, "permutations": total}]


SOLVERS = {
    "enrichment-universe": solve_enrichment,
    "design-estimability": solve_estimability,
    "paired-treatment-effect": solve_paired,
    "adjusted-linear-effect": solve_adjusted,
    "odds-ratio-contingency": solve_odds,
    "kaplan-meier": solve_survival,
    "diagnostic-thresholds": solve_thresholds,
    "permutation-mean-test": solve_permutation,
}
