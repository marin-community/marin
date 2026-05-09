"""Mine the 46-statement dataset for features that predict Δα(rubric − bare).

Goal: a generalizable rule for when to use rubric vs bare. Compute candidate
features per statement and test their correlation with Δα.

Features:
  - α_bare (population-level baseline agreement)
  - α_phase_4 (population-level rubric agreement)
  - Δα (target)
  - n_carve_outs: count of "unless", "except", "however", "by default",
    "typically", "in general", "if" clauses in spec text
  - spec_len: spec text length in chars
  - trivial_clean_rate: % of cells where all 3 judges score in {4, 5} under bare
  - score_distribution_entropy_bare: entropy of judge scores under bare
    (high = graded usage; low = ceiling effect)
  - n_explicit_examples: count of spec examples
  - score_dispersion_bare: stdev of bare scores per cell, averaged across cells
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from e9_kappa_alternatives import krippendorff_alpha_interval
from e9_build_qualitative_inputs import load_judgments

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")

CARVE_OUT_PATTERNS = [
    r"\bunless\b", r"\bexcept\b", r"\bhowever\b", r"\bby default\b",
    r"\btypically\b", r"\bin general\b", r"\bin most cases\b",
    r"\bordinarily\b", r"\bappropriate\b", r"\bunless explicitly\b",
    r"\bappropriate(ly)?\b", r"\bif (the )?user\b",
]


def count_carve_outs(text: str) -> int:
    n = 0
    for p in CARVE_OUT_PATTERNS:
        n += len(re.findall(p, text, flags=re.IGNORECASE))
    return n


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0: return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def main():
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    by_cell = load_judgments()

    rows = []
    for sid, stmt in spec.items():
        # Cells for this statement, both conditions
        bare_tuples, p4_tuples = [], []
        bare_score_counts = Counter()
        n_trivial_clean = 0
        n_total_full = 0
        for (s, c, scen, gen), jd in by_cell.items():
            if s != sid: continue
            if not all(j in jd for j in ("gpt", "gemini", "claude")): continue
            tup = (jd["gpt"]["score"], jd["gemini"]["score"], jd["claude"]["score"])
            if c == "variant_A":
                bare_tuples.append(tup)
                for x in tup: bare_score_counts[x] += 1
                n_total_full += 1
                if all(x in (4, 5) for x in tup): n_trivial_clean += 1
            elif c == "rubric_plus_spec":
                p4_tuples.append(tup)

        if len(bare_tuples) < 2 or len(p4_tuples) < 2:
            continue
        a_bare = krippendorff_alpha_interval(bare_tuples)
        a_p4 = krippendorff_alpha_interval(p4_tuples)
        if a_bare is None or a_p4 is None:
            continue
        d_alpha = a_p4 - a_bare

        text = stmt.get("text", "")
        n_carve = count_carve_outs(text)
        spec_len = len(text)
        trivial = 100 * n_trivial_clean / n_total_full if n_total_full else 0
        h_bare = entropy(bare_score_counts)
        n_examples = len((stmt.get("metadata") or {}).get("examples", []) or [])

        rows.append({
            "sid": sid,
            "alpha_bare": a_bare,
            "alpha_p4": a_p4,
            "delta_alpha": d_alpha,
            "n_carve_outs": n_carve,
            "spec_len": spec_len,
            "trivial_clean_pct": trivial,
            "score_entropy_bare": h_bare,
            "n_examples": n_examples,
        })

    # Print full table
    print("# Per-statement features + Δα\n")
    print("Sorted by Δα descending.\n")
    print("| statement | α_bare | α_p4 | Δα | n_carve | trivial% | spec_len | entropy | n_examples |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    rows.sort(key=lambda r: -r["delta_alpha"])
    for r in rows:
        print(f"| {r['sid']} | {r['alpha_bare']:+.3f} | {r['alpha_p4']:+.3f} | {r['delta_alpha']:+.3f} | "
              f"{r['n_carve_outs']} | {r['trivial_clean_pct']:.0f}% | {r['spec_len']} | "
              f"{r['score_entropy_bare']:.2f} | {r['n_examples']} |")

    # Pearson correlations
    def pearson(xs, ys):
        n = len(xs)
        if n < 3: return None
        mx = sum(xs)/n; my = sum(ys)/n
        cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/n
        sx = (sum((x-mx)**2 for x in xs)/n) ** 0.5
        sy = (sum((y-my)**2 for y in ys)/n) ** 0.5
        if sx*sy == 0: return None
        return cov / (sx*sy)

    print("\n# Correlations of Δα with each feature (n={})\n".format(len(rows)))
    feats = ["alpha_bare", "alpha_p4", "n_carve_outs", "spec_len",
             "trivial_clean_pct", "score_entropy_bare", "n_examples"]
    print("| feature | Pearson(feat, Δα) |")
    print("|---|--:|")
    ys = [r["delta_alpha"] for r in rows]
    for f in feats:
        xs = [r[f] for r in rows]
        p = pearson(xs, ys)
        print(f"| {f} | {p:+.3f} |" if p is not None else f"| {f} | n/a |")

    # Bucket analysis: classify by Δα and see feature distributions
    print("\n# Bucket analysis: distinguishing 'rubric helps' (Δα > +0.05) from 'rubric hurts' (Δα < -0.05)\n")
    helps = [r for r in rows if r["delta_alpha"] > 0.05]
    hurts = [r for r in rows if r["delta_alpha"] < -0.05]
    neutral = [r for r in rows if -0.05 <= r["delta_alpha"] <= 0.05]
    print(f"helps:   {len(helps)} statements")
    print(f"neutral: {len(neutral)} statements")
    print(f"hurts:   {len(hurts)} statements\n")

    print("| feature | mean (helps) | mean (neutral) | mean (hurts) |")
    print("|---|--:|--:|--:|")
    for f in feats:
        mh = sum(r[f] for r in helps)/max(1, len(helps))
        mn = sum(r[f] for r in neutral)/max(1, len(neutral))
        mt = sum(r[f] for r in hurts)/max(1, len(hurts))
        print(f"| {f} | {mh:.3f} | {mn:.3f} | {mt:.3f} |")

    # Test some candidate rules
    print("\n# Candidate predictive rules — apply to all 46 and measure accuracy\n")
    def evaluate(rule_name, predict_fn):
        # predict_fn(row) → True (predict rubric helps) or False (predict rubric hurts)
        # ground truth: row["delta_alpha"] > 0.05 means rubric truly helps
        correct = 0
        tp = fp = tn = fn = 0
        for r in rows:
            truth = r["delta_alpha"] > 0.05
            pred = predict_fn(r)
            if pred and truth: tp += 1
            elif pred and not truth: fp += 1
            elif not pred and not truth: tn += 1
            else: fn += 1
        n = len(rows)
        acc = (tp + tn) / n
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2*prec*rec / max(1e-9, prec+rec)
        print(f"  {rule_name}")
        print(f"    accuracy: {acc:.2f}, precision: {prec:.2f}, recall: {rec:.2f}, f1: {f1:.2f}")
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

    # Rule 1: always use rubric
    evaluate("R1 (always use rubric)", lambda r: True)
    # Rule 2: only if α_bare in [0.3, 0.65]
    evaluate("R2 (α_bare in [0.3, 0.65])", lambda r: 0.3 <= r["alpha_bare"] <= 0.65)
    # Rule 3: α_bare in [0.3, 0.65] AND n_carve_outs <= 2
    evaluate("R3 (α_bare in [0.3, 0.65] AND n_carve_outs <= 2)",
             lambda r: 0.3 <= r["alpha_bare"] <= 0.65 and r["n_carve_outs"] <= 2)
    # Rule 4: α_bare < 0.85 AND n_carve_outs == 0
    evaluate("R4 (α_bare < 0.85 AND n_carve_outs == 0)",
             lambda r: r["alpha_bare"] < 0.85 and r["n_carve_outs"] == 0)
    # Rule 5: α_bare < 0.85 AND n_carve_outs <= 1
    evaluate("R5 (α_bare < 0.85 AND n_carve_outs <= 1)",
             lambda r: r["alpha_bare"] < 0.85 and r["n_carve_outs"] <= 1)
    # Rule 6: never use rubric (always bare)
    evaluate("R6 (never use rubric)", lambda r: False)
    # Rule 7: α_bare < 0.85 AND trivial_clean < 90%
    evaluate("R7 (α_bare < 0.85 AND trivial_clean < 90%)",
             lambda r: r["alpha_bare"] < 0.85 and r["trivial_clean_pct"] < 90)
    # Rule 8: simple just α_bare in middle range, no carve-out filter
    evaluate("R8 (α_bare in [0.4, 0.7])", lambda r: 0.4 <= r["alpha_bare"] <= 0.7)


if __name__ == "__main__":
    main()
