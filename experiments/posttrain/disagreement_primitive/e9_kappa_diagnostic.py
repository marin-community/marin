# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-statement Fleiss' kappa by judging condition (Q1 diagnostic for the
spec-repair loop).

Reads grounding/per_judgment.jsonl (produced by e8_rationale_grounding.py).
For each (statement_id, condition), computes Fleiss' kappa across the 3 judges
on the binary "problematic" collapse:

  - var_A / var_B / phase_4 (rubric+spec): score in {1,2} -> problematic
  - full_spec (phase 3): decision in {non-compliant, ambiguous} -> problematic

Cases require all-3-judge coverage. The output is a per-statement record with
all four kappa values and the delta(var_A -> phase_4), which is the primary
loop diagnostic: a large positive delta indicates the rubric is force-picking
an interpretation that the bare spec text leaves ambiguous; a large negative
delta indicates the rubric introduces conflict the spec text doesn't carry.

Outputs:
  - stdout: human-readable table sorted by var_A kappa ascending
  - per_statement_kappa_by_condition.jsonl: one JSON record per statement

Reproducibility: pure stdlib, deterministic, no LM calls. Run from repo root.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_JUDGMENT = ROOT / "grounding" / "per_judgment.jsonl"
OUT_JSONL = ROOT / "per_statement_kappa_by_condition.jsonl"

CONDITIONS = ["variant_A", "variant_B", "rubric_plus_spec", "full_spec"]
PRETTY = {
    "variant_A": "var_A",
    "variant_B": "var_B",
    "rubric_plus_spec": "phase_4",
    "full_spec": "full_spec",
}


def is_problematic(row):
    if row["condition"] == "full_spec":
        d = row.get("decision")
        if d is None:
            return None
        return 1 if d.lower() in ("non-compliant", "non_compliant", "ambiguous") else 0
    s = row.get("score")
    if s is None:
        return None
    try:
        s = int(s)
    except Exception:
        return None
    return 1 if s in (1, 2) else 0


def fleiss_kappa(table):
    """Fleiss' kappa for binary categories, fixed N raters per subject."""
    if not table:
        return None
    N = sum(table[0])
    if N < 2:
        return None
    n = len(table)
    totals = [0.0, 0.0]
    for row in table:
        totals[0] += row[0]
        totals[1] += row[1]
    p = [t / (n * N) for t in totals]
    P_e = sum(pk * pk for pk in p)
    P_bar = sum((sum(nk * nk for nk in row) - N) / (N * (N - 1)) for row in table) / n
    if P_e >= 1.0 - 1e-12:
        return None
    return (P_bar - P_e) / (1.0 - P_e)


def main():
    rows = defaultdict(lambda: defaultdict(dict))
    n_loaded = 0
    n_skipped = 0
    with open(PER_JUDGMENT) as f:
        for line in f:
            r = json.loads(line)
            v = is_problematic(r)
            if v is None:
                n_skipped += 1
                continue
            key = (r["scenario_idx"], r["generator"])
            rows[(r["statement_id"], r["condition"])][key][r["judge"]] = v
            n_loaded += 1

    print(f"loaded {n_loaded:,} judgments, skipped {n_skipped:,}", file=sys.stderr)

    results = defaultdict(dict)
    for (stmt, cond), cases in rows.items():
        table = [
            (sum(1 for v in jdict.values() if v == 1), sum(1 for v in jdict.values() if v == 0))
            for jdict in cases.values()
            if len(jdict) >= 3
        ]
        kap = fleiss_kappa(table) if table else None
        results[stmt][cond] = (kap, len(table))

    statements = list(results.keys())

    def sort_key(stmt):
        v = results[stmt].get("variant_A", (None, 0))[0]
        return (v if v is not None else 99, stmt)

    statements.sort(key=sort_key)

    # write per-statement JSONL
    with open(OUT_JSONL, "w") as f:
        for s in statements:
            rec = {"statement_id": s}
            for cond in CONDITIONS:
                k, n = results[s].get(cond, (None, 0))
                rec[f"kappa_{PRETTY[cond]}"] = k
                rec[f"n_{PRETTY[cond]}"] = n
            v_a = rec.get("kappa_var_A")
            v_p = rec.get("kappa_phase_4")
            rec["delta_var_A_to_phase_4"] = None if (v_a is None or v_p is None) else round(v_p - v_a, 4)
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {OUT_JSONL}", file=sys.stderr)

    # stdout: full table sorted by var_A asc
    hdr = (
        f"{'statement':<42} {'κ_var_A':>8} {'κ_var_B':>8} "
        f"{'κ_phase4':>9} {'κ_full':>7} {'Δ(A→P4)':>9}    "
        f"{'n_A':>3} {'n_B':>3} {'n_P4':>4} {'n_F':>3}"
    )
    print(hdr)
    print("-" * len(hdr))

    def fmt(v):
        return "  n/a " if v is None else f"{v:+.3f}"

    pop = {c: [] for c in CONDITIONS}
    for s in statements:
        v_a = results[s].get("variant_A", (None, 0))
        v_b = results[s].get("variant_B", (None, 0))
        v_p = results[s].get("rubric_plus_spec", (None, 0))
        v_f = results[s].get("full_spec", (None, 0))
        delta = None if (v_a[0] is None or v_p[0] is None) else v_p[0] - v_a[0]
        line = (
            f"{s:<42} {fmt(v_a[0]):>8} {fmt(v_b[0]):>8} "
            f"{fmt(v_p[0]):>9} {fmt(v_f[0]):>7} "
            f"{fmt(delta):>9}    "
            f"{v_a[1]:>3} {v_b[1]:>3} {v_p[1]:>4} {v_f[1]:>3}"
        )
        print(line)
        for cond in CONDITIONS:
            k = results[s].get(cond, (None, 0))[0]
            if k is not None:
                pop[cond].append(k)

    # population summary
    print()
    print("population summary (per-statement κ across 46 statements):")
    print(f"  {'condition':<20} {'n':>3}  {'median':>8}  {'p25':>7}  {'p75':>7}  {'κ<0':>5}  {'κ<0.4':>6}")
    for cond in CONDITIONS:
        vals = sorted(pop[cond])
        if not vals:
            continue
        med = statistics.median(vals)
        p25 = vals[len(vals) // 4]
        p75 = vals[3 * len(vals) // 4]
        n_neg = sum(1 for v in vals if v < 0)
        n_low = sum(1 for v in vals if v < 0.4)
        print(f"  {PRETTY[cond]:<20} {len(vals):>3}  " f"{med:+.3f}  {p25:+.3f}  {p75:+.3f}  " f"{n_neg:>5}  {n_low:>6}")

    # delta summary
    print()
    print("delta(var_A → phase_4) — interpretation aid:")
    deltas = sorted(
        [
            (s, results[s]["rubric_plus_spec"][0] - results[s]["variant_A"][0])
            for s in statements
            if results[s]["variant_A"][0] is not None and results[s]["rubric_plus_spec"][0] is not None
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n  TOP 5 RUBRIC-FORCE-PICK (delta most positive — rubric resolves spec ambiguity):")
    for s, d in deltas[:5]:
        v_a = results[s]["variant_A"][0]
        v_p = results[s]["rubric_plus_spec"][0]
        print(f"    {s:<42} var_A={v_a:+.3f}  phase4={v_p:+.3f}  Δ={d:+.3f}")

    print("\n  TOP 5 RUBRIC-INTRODUCES-CONFLICT (delta most negative — rubric distorts):")
    for s, d in deltas[-5:]:
        v_a = results[s]["variant_A"][0]
        v_p = results[s]["rubric_plus_spec"][0]
        print(f"    {s:<42} var_A={v_a:+.3f}  phase4={v_p:+.3f}  Δ={d:+.3f}")


if __name__ == "__main__":
    main()
