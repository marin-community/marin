# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize executable MT-MBPP pass@1 per checkpoint and language.

Each checkpoint-language pair takes its grades from the first plan in ``PLAN_ORDER`` that graded it (Olmix ran only in
us-east1; the other three raced in europe-west4 and us-east5). Pairs graded in more than one plan are compared
problem by problem, which checks that greedy generation reproduced across runs. The 17-language mean weights languages
equally; its 95% interval and the paired differences between mixtures come from a bootstrap that resamples MBPP
problems (task ids) within each language, jointly across mixtures.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.summarize_results \
    --grading DIR --output DIR
"""

import argparse
import csv
import gzip
import json
from itertools import combinations
from pathlib import Path

import numpy as np

PLAN_ORDER = ("plan_east1", "plan_euw4", "plan_east5b", "plan_east5")
MIXTURES = {
    "proportional_1e21-2f1a48": "Proportional",
    "unimax8_1e21-d685cd": "UniMax-8",
    "olmixq_t9_kl0p005_cap04_1e21-3f95f2": "Olmix",
    "lwspu_t9_snc_cap08_1e21-e8e9d7": "MARINER",
}
BOOTSTRAP = 10_000
SEED = 0


def load(grading: Path) -> dict:
    """{(plan, checkpoint, language): {task_id: passed}} for every graded pair."""
    out = {}
    for path in grading.glob("plan_*/*/mt_mbpp_*.jsonl.gz"):
        plan, checkpoint = path.parts[-3], path.parts[-2]
        language = path.name.removeprefix("mt_mbpp_").removesuffix(".jsonl.gz")
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle]
        out[(plan, checkpoint, language)] = {r["task_id"]: bool(r["passed"]) for r in rows}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graded = load(args.grading)
    languages = sorted({k[2] for k in graded})
    chosen, agreement = {}, []
    for checkpoint in MIXTURES:
        for language in languages:
            plans = [p for p in PLAN_ORDER if (p, checkpoint, language) in graded]
            if not plans:
                raise ValueError(f"No grades for {checkpoint}/{language}")
            chosen[(checkpoint, language)] = (plans[0], graded[(plans[0], checkpoint, language)])
            for other in plans[1:]:
                a, b = graded[(plans[0], checkpoint, language)], graded[(other, checkpoint, language)]
                agreement.append(
                    {
                        "checkpoint": checkpoint,
                        "language": language,
                        "plans": [plans[0], other],
                        "problems": len(a),
                        "differ": sum(a[t] != b[t] for t in a),
                    }
                )
    table = []
    for (checkpoint, language), (plan, grades) in sorted(chosen.items()):
        table.append(
            {
                "mixture": MIXTURES[checkpoint],
                "checkpoint": checkpoint,
                "language": language,
                "plan": plan,
                "scored": len(grades),
                "passed": sum(grades.values()),
                "pass@1": sum(grades.values()) / len(grades),
            }
        )
    # Bootstrap: resample task ids within each language, the same draw for every mixture.
    rng = np.random.default_rng(SEED)
    draws = {m: np.zeros(BOOTSTRAP) for m in MIXTURES}
    for language in languages:
        ids = sorted(chosen[(next(iter(MIXTURES)), language)][1])
        if any(sorted(chosen[(c, language)][1]) != ids for c in MIXTURES):
            raise ValueError(f"Scored problems differ across mixtures: {language}")
        index = rng.integers(0, len(ids), size=(BOOTSTRAP, len(ids)))
        for checkpoint in MIXTURES:
            passed = np.array([chosen[(checkpoint, language)][1][t] for t in ids], dtype=float)
            draws[checkpoint] += passed[index].mean(axis=1) / len(languages)
    means = {MIXTURES[c]: float(np.mean([r["pass@1"] for r in table if r["checkpoint"] == c])) for c in MIXTURES}
    summary = {
        "languages": len(languages),
        "mean_pass@1": {
            MIXTURES[c]: {"mean": means[MIXTURES[c]], "ci95": [float(x) for x in np.percentile(draws[c], [2.5, 97.5])]}
            for c in MIXTURES
        },
        "differences": {
            f"{MIXTURES[b]} - {MIXTURES[a]}": {
                "mean": means[MIXTURES[b]] - means[MIXTURES[a]],
                "ci95": [float(x) for x in np.percentile(draws[b] - draws[a], [2.5, 97.5])],
                "p_leq_0": float(np.mean(draws[b] - draws[a] <= 0)),
            }
            for a, b in combinations(MIXTURES, 2)
        },
        "cross_run_agreement": agreement,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cross_run_agreement"}, indent=1))
    print("cross-run pairs:", len(agreement), "with differences:", [a for a in agreement if a["differ"]])


if __name__ == "__main__":
    main()
