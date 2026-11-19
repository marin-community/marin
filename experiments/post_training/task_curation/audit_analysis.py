# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile independently labeled, stratified quality-review audit records."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20261113
BOOTSTRAP_REPLICATES = 20000


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def weighted_fraction(rows: list[dict], category: str, population: int) -> float:
    return sum(row["weight"] for row in rows if row["classification"] == category) / population


def analyze(root: Path) -> dict:
    frozen = root / "frozen-blind-v2"
    summary = json.loads((frozen / "sample-summary.json").read_text())
    keys = {row["case_id"]: row for row in json_lines(frozen / "keys.jsonl")}
    causes = {row["case_id"]: row for row in json_lines(root / "claim-causes.jsonl")}
    reviewers, validation = {}, {}
    for label in ("A", "B"):
        packet = frozen / label
        labels = json_lines(packet / "labels.jsonl")
        order = json.loads((packet / "batch.json").read_text())["case_order"]
        if [row["case_id"] for row in labels] != order or set(order) != set(keys):
            raise ValueError("Reviewer labels must cover the frozen packet exactly in order")
        reviewers[label] = {row["case_id"]: row for row in labels}
        validation[label] = {
            "counts": dict(Counter(row["verdict"] for row in labels)),
            "sha256": hashlib.sha256((packet / "labels.jsonl").read_bytes()).hexdigest(),
        }
    rows = []
    for case_id, key in sorted(keys.items()):
        first, second = (reviewers[label][case_id] for label in ("A", "B"))
        classification = "uncertain_or_disputed"
        definite = first["verdict"] == second["verdict"] and first["verdict"] != "ambiguous"
        if definite:
            if first["verdict"] == "fail":
                classification = "true_defect"
            elif causes[case_id]["claim_scope"] == "harness_only":
                classification = "harness_false_positive"
            else:
                classification = "other_false_positive"
        rows.append(
            {
                "case_id": case_id,
                "stratum": key["stratum"],
                "weight": key["population_count"] / key["sample_count"],
                "claim_scope": causes[case_id]["claim_scope"],
                "reviewer_a": first,
                "reviewer_b": second,
                "definite_consensus": definite,
                "lean_consensus": first["lean"] == second["lean"],
                "classification": classification,
            }
        )
    population = summary["population_size"]
    strata = {}
    rng = np.random.default_rng(SEED)
    bootstrap = np.zeros(BOOTSTRAP_REPLICATES)
    for name in ("rubric_mismatch", "rare_other", "other"):
        sizes = summary["strata"][name]
        selected = [row for row in rows if row["stratum"] == name]
        strata[name] = {**sizes, **Counter(row["classification"] for row in selected)}
        indicators = np.array([row["classification"] == "harness_false_positive" for row in selected])
        means = (
            np.full(BOOTSTRAP_REPLICATES, indicators.mean())
            if sizes["population"] == sizes["sample"]
            else rng.choice(indicators, (BOOTSTRAP_REPLICATES, len(indicators)), replace=True).mean(axis=1)
        )
        bootstrap += sizes["population"] / population * means
    measured = weighted_fraction(rows, "harness_false_positive", population)
    result = {
        "sample_size": len(rows),
        "issue_population": population,
        "reviewer_validation": validation,
        "strata": strata,
        "classifications": dict(Counter(row["classification"] for row in rows)),
        "weighted_confirmed_harness_false_discovery_fraction": measured,
        "weighted_confirmed_any_false_discovery_fraction": (
            measured + weighted_fraction(rows, "other_false_positive", population)
        ),
        "weighted_uncertain_fraction": weighted_fraction(rows, "uncertain_or_disputed", population),
        "harness_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "census_stratum_fixed": True,
        "definite_consensus": sum(row["definite_consensus"] for row in rows),
        "lean_consensus": sum(row["lean_consensus"] for row in rows),
        "material_threshold": 0.1,
        "material": measured >= 0.1,
        "limitation": (
            "Issue-positive population only; uncertainty is not clean. Intervals condition on two "
            "instructed-independent same-configuration model reviewers and do not remove shared blind spots."
        ),
    }
    (root / "recomputed-review.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "recomputed-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))


if __name__ == "__main__":
    main()
