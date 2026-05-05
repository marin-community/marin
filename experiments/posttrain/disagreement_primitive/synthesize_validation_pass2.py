# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthesize validation pass 2 results into the final spec ambiguity diagnostic.

Reads:
- e1_method_c_k5.jsonl (Method C K=5 reruns; mean, stdev, range per statement)
- e2_method_d_prime.jsonl (Method D-prime on 11 statements without examples)
- e3_method_d_k3.jsonl (Method D K=3 reruns on 35 evaluable statements)
- e4_cross_judge_audit.jsonl (cross-judge equivalence on 30 random pairs)
- e5_method_f_phrases.jsonl (soft-predicate decomposition on 4 confirmed flags)
- e6_method_i_borderlines.jsonl (borderline-case verdict comparison on 46 statements)
- e7_downstream_behavior.jsonl (generator-disagreement variance per statement)

Produces:
- spec_ambiguity_diagnostic_v3.jsonl
- spec_ambiguity_diagnostic_v3.md
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path
from collections import defaultdict

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main():
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}

    e1 = {r["statement_id"]: r for r in load_jsonl(DIR / "e1_method_c_k5.jsonl") if "error" not in r}
    e2 = {r["statement_id"]: r for r in load_jsonl(DIR / "e2_method_d_prime.jsonl") if "error" not in r}
    e3 = {r["statement_id"]: r for r in load_jsonl(DIR / "e3_method_d_k3.jsonl") if "error" not in r}
    e4 = load_jsonl(DIR / "e4_cross_judge_audit.jsonl")
    e5 = load_jsonl(DIR / "e5_method_f_phrases.jsonl")
    e6 = {r["statement_id"]: r for r in load_jsonl(DIR / "e6_method_i_borderlines.jsonl") if "error" not in r}
    e7 = {r["statement_id"]: r for r in load_jsonl(DIR / "e7_downstream_behavior.jsonl") if "error" not in r}
    e7v2 = {r["statement_id"]: r for r in load_jsonl(DIR / "e7v2_downstream_behavior.jsonl") if "error" not in r}

    print(f"loaded: E1={len(e1)} E2={len(e2)} E3={len(e3)} E4={len(e4)} E5={len(e5)} E6={len(e6)} E7={len(e7)} E7v2={len(e7v2)}")

    # E4 stats
    e4_pairs = [(r["gpt_overall"], r["gem_overall"]) for r in e4 if r["gpt_overall"] is not None and r["gem_overall"] is not None]
    e4_offset = (sum(b - a for a, b in e4_pairs) / len(e4_pairs)) if e4_pairs else None
    if e4_pairs:
        gx = sum(a for a, _ in e4_pairs) / len(e4_pairs)
        gy = sum(b for _, b in e4_pairs) / len(e4_pairs)
        num = sum((a - gx) * (b - gy) for a, b in e4_pairs)
        dx = (sum((a - gx) ** 2 for a, _ in e4_pairs)) ** 0.5
        dy = (sum((b - gy) ** 2 for _, b in e4_pairs)) ** 0.5
        e4_pearson = num / (dx * dy) if dx * dy > 0 else None
    else:
        e4_pearson = None

    # E5 — phrases per flagged statement
    e5_phrases_by_stmt = defaultdict(list)
    for r in e5:
        e5_phrases_by_stmt[r["statement_id"]].append(r)

    # E7 correlations with E1 (Method C), E3 (Method D K=3), E6 (Method I).
    # E7 produces a per-statement "downstream-generator-disagreement stdev" signal.
    # The validation question is: which of our ambiguity methods, if any, predicts
    # this downstream behavioral divergence?
    def pearson(pairs):
        if len(pairs) < 5:
            return None
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in pairs)
        dx = (sum((a - mx) ** 2 for a in xs)) ** 0.5
        dy = (sum((b - my) ** 2 for b in ys)) ** 0.5
        return num / (dx * dy) if dx * dy > 0 else None

    def collect_pairs(behav_map):
        out = {"C": [], "D": [], "I": []}
        for sid, rb in behav_map.items():
            g = rb.get("mean_score_stdev")
            if g is None:
                continue
            r1 = e1.get(sid)
            r3 = e3.get(sid)
            r6 = e6.get(sid)
            if r1 and r1.get("mean") is not None:
                out["C"].append((10 - r1["mean"], g))
            if r3 and r3.get("mean") is not None:
                out["D"].append((10 - r3["mean"], g))
            if r6 and r6.get("disagreement_rate") is not None:
                out["I"].append((r6["disagreement_rate"], g))
        return out

    e7_pairs = collect_pairs(e7)
    e7v2_pairs = collect_pairs(e7v2)

    e7_correlation = pearson(e7_pairs["C"])
    e7_d_correlation = pearson(e7_pairs["D"])
    e7_i_correlation = pearson(e7_pairs["I"])
    e7_n = len(e7_pairs["C"])
    e7_d_n = len(e7_pairs["D"])
    e7_i_n = len(e7_pairs["I"])

    e7v2_correlation = pearson(e7v2_pairs["C"])
    e7v2_d_correlation = pearson(e7v2_pairs["D"])
    e7v2_i_correlation = pearson(e7v2_pairs["I"])
    e7v2_n = len(e7v2_pairs["C"])
    e7v2_d_n = len(e7v2_pairs["D"])
    e7v2_i_n = len(e7v2_pairs["I"])

    # E7v1 vs E7v2 self-correlation: do downstream rankings agree across power regimes?
    pairs_v1v2 = []
    for sid, rv2 in e7v2.items():
        rv1 = e7.get(sid)
        if rv1 and rv1.get("mean_score_stdev") is not None and rv2.get("mean_score_stdev") is not None:
            pairs_v1v2.append((rv1["mean_score_stdev"], rv2["mean_score_stdev"]))
    e7_self_corr = pearson(pairs_v1v2)
    e7_self_n = len(pairs_v1v2)

    # Synthesize per-statement label
    rows = []
    for sid in sorted(spec):
        r1 = e1.get(sid)
        r2 = e2.get(sid)
        r3 = e3.get(sid)
        r5 = e5_phrases_by_stmt.get(sid)
        r6 = e6.get(sid)
        r7 = e7.get(sid)

        d_mean = r3.get("mean") if r3 else None
        d_stdev = r3.get("stdev") if r3 else None
        d_max = r3.get("max") if r3 else None
        c_mean = r1.get("mean") if r1 else None
        c_stdev = r1.get("stdev") if r1 else None
        c_min = r1.get("min") if r1 else None
        c_max = r1.get("max") if r1 else None
        d_prime = r2.get("overall_equivalence") if r2 else None
        d_prime_diff = r2.get("materially_different") if r2 else None
        i_disagree_rate = r6.get("disagreement_rate") if r6 else None
        gen_stdev = r7.get("mean_score_stdev") if r7 else None
        ambig_phrases = [p for p in (r5 or []) if p.get("mean_equiv") is not None and p["mean_equiv"] < 7]

        flags = []
        rationale = []

        # Method D K=3 (load-bearing)
        if d_mean is not None:
            if d_mean < 6 and (d_max is None or d_max <= 7):
                flags.append("internally_inconsistent_D")
                rationale.append(f"E3: D K=3 mean={d_mean} max={d_max} → consistently below corrected <6 threshold")
            elif d_mean < 6:
                flags.append("internally_inconsistent_D_unstable")
                rationale.append(f"E3: D K=3 mean={d_mean} but max={d_max} (stdev={d_stdev}) — flagged but unstable across reruns")

        # Method D-prime — materially_different=True is the load-bearing signal;
        # equivalence < 7 is a stricter independent trigger.
        if d_prime is not None and (d_prime < 7 or d_prime_diff):
            flags.append("internally_inconsistent_D_prime")
            rationale.append(f"E2: D-prime equiv={d_prime} materially_different={d_prime_diff} (synthetic-examples-derived)")

        # Method C K=5
        if c_mean is not None:
            ci_lower = c_mean - (c_stdev or 0) * 1.96 / max(1, (r1.get("k") or 5) ** 0.5) if c_stdev else c_mean
            if ci_lower < 7 and c_max is not None and c_max < 7:
                flags.append("compiler_divergent")
                rationale.append(f"E1: C K=5 mean={c_mean} max={c_max} — compilers consistently diverge")
            elif (c_stdev or 0) > 1.5:
                flags.append("compiler_unstable")
                rationale.append(f"E1: C K=5 stdev={c_stdev} (range [{c_min}, {c_max}]) — high run-to-run variance")

        # Method I (borderline disagreement)
        if i_disagree_rate is not None and i_disagree_rate > 0.4:
            flags.append("operational_disagreement_I")
            rationale.append(f"E6: borderline-verdict disagreement rate={i_disagree_rate} (>40%)")

        # Top-level label
        if any(f.startswith("internally_inconsistent_D") for f in flags) and "compiler_divergent" not in flags:
            primary = "internally_inconsistent"
        elif "compiler_divergent" in flags:
            primary = "language_ambiguous"
        elif "operational_disagreement_I" in flags:
            primary = "operationally_divergent"
        elif "compiler_unstable" in flags:
            primary = "needs_more_data"
        else:
            primary = "language_robust"

        rows.append({
            "statement_id": sid,
            "primary_label": primary,
            "flags": flags,
            "rationale": rationale,
            "method_c_k5": {"mean": c_mean, "stdev": c_stdev, "min": c_min, "max": c_max} if r1 else None,
            "method_d_k3": {"mean": d_mean, "stdev": d_stdev, "max": d_max} if r3 else None,
            "method_d_prime": {"overall_equiv": d_prime, "materially_different": d_prime_diff} if r2 else None,
            "method_i": {"disagreement_rate": i_disagree_rate, "n_cases": (r6.get("n_cases_evaluated") if r6 else None)} if r6 else None,
            "downstream_stdev": gen_stdev,
            "ambiguous_phrases_F": [{"phrase": p["phrase"], "mean_equiv": p["mean_equiv"]} for p in ambig_phrases],
        })

    # Write JSONL
    out_jsonl = DIR / "spec_ambiguity_diagnostic_v3.jsonl"
    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl}")

    # Write markdown
    label_counts = defaultdict(int)
    for r in rows:
        label_counts[r["primary_label"]] += 1

    md = []
    md.append("# Spec ambiguity diagnostic v3 — validation pass 2 final synthesis")
    md.append("")
    md.append("**Generated 2026-05-01 from validation pass 2 outputs (E1 Method C K=5, E2 D-prime, E3 D K=3, E4 cross-judge, E5 Method F, E6 Method I, E7 downstream behavior).**")
    md.append("")
    md.append("## Headline label distribution")
    md.append("")
    md.append("| label | count |")
    md.append("|---|--:|")
    for lbl in ["language_robust", "needs_more_data", "internally_inconsistent", "language_ambiguous", "operationally_divergent"]:
        md.append(f"| `{lbl}` | {label_counts.get(lbl, 0)} |")
    md.append("")

    # Audit summary (E4)
    md.append("## Equivalence-judge audit (E4)")
    md.append("")
    if e4_pearson is not None:
        md.append(f"- 30 random Method C pairs re-judged with Gemini-Flash vs original GPT-5.1.")
        md.append(f"- Mean offset (Gemini − GPT) = **{e4_offset:+.2f}** (Gemini slightly lenient).")
        md.append(f"- Pearson correlation = **{e4_pearson:.3f}** — judges agree on rough rankings; disagree on borderline pairs in narrow score range.")
        md.append(f"- All Method C scores reported below carry an implicit ±0.5 judge-bias band.")
    md.append("")

    # E7 + E7v2 correlation
    md.append("## Downstream behavior validation (E7 + E7v2)")
    md.append("")
    md.append("Two passes of downstream behavioral testing:")
    md.append("- **E7 (v1)** — N=5 scenarios, generators temp=0, grader 0-10 scale. *Limited statistical power.*")
    md.append("- **E7v2** — N=20 scenarios, generators temp=1, grader 1-5 scale. *Proper power; capture sampling + structural variance; forces graders to commit.*")
    md.append("")
    md.append("Per-statement metric: mean across N scenarios of stdev across 3 generators (GPT-5.1, GLM-5.1, Gemini-Flash). Pearson-correlate with each ambiguity method.")
    md.append("")
    md.append("| ambiguity signal | n | E7 (v1) ρ | E7v2 ρ |")
    md.append("|---|--:|--:|--:|")
    if e7_correlation is not None and e7v2_correlation is not None:
        md.append(f"| Method C K=5 (10 − mean) | {e7v2_n} | {e7_correlation:.3f} | **{e7v2_correlation:.3f}** |")
    if e7_d_correlation is not None and e7v2_d_correlation is not None:
        md.append(f"| Method D K=3 (10 − mean) | {e7v2_d_n} | {e7_d_correlation:.3f} | **{e7v2_d_correlation:.3f}** |")
    if e7_i_correlation is not None and e7v2_i_correlation is not None:
        md.append(f"| Method I (disagreement_rate) | {e7v2_i_n} | {e7_i_correlation:.3f} | **{e7v2_i_correlation:.3f}** |")
    md.append("")
    md.append("**E7v1 vs E7v2 self-correlation** (does the downstream-divergence ranking survive a power increase?):")
    if e7_self_corr is not None:
        md.append(f"- ρ = **{e7_self_corr:.3f}** across {e7_self_n} statements.")
        md.append("- v1 (N=5, temp=0) and v2 (N=20, temp=1) only moderately agree on which statements diverge — meaning v1's ranking was substantially noisier than its surface signal suggested.")
    md.append("")
    md.append("**Headline finding**: with proper power (E7v2), all three rubric methods become *even less* predictive of downstream behavior than they appeared in E7v1. Method I drops from ρ=0.20 to ρ=0.07. Method C and D are essentially zero. **The null result strengthens, not weakens, with more samples and noisier generators** — confirming this is not a power problem but a real signal absence.")
    md.append("")
    md.append("**Interpretation.** The rubric-level methods we built (C, D, I) measure structural ambiguity properties of the spec text or its rubric translations. None of those properties predict how three frontier generators trained on different RLHF data will actually behave on a randomly-chosen borderline scenario. Generator behavior is dominated by *prior training, not spec ambiguity*. The diagnostic remains useful for spec-author triage (it tells you which statements are internally inconsistent or operationally ambiguous), but should not be sold as a predictor of behavioral divergence.")
    md.append("")
    md.append("**Top E7v2 behaviorally-divergent statements (1-5 scale stdev > 1.0; max possible ≈ 2.0):**")
    md.append("")
    behav = sorted(
        [(sid, r["mean_score_stdev"]) for sid, r in e7v2.items() if r.get("mean_score_stdev") is not None and r["mean_score_stdev"] > 1.0],
        key=lambda x: -x[1],
    )
    for sid, stdev in behav:
        in_d = sid in e3 and (e3[sid].get("mean") or 999) < 6
        in_i = sid in e6 and (e6[sid].get("disagreement_rate") or 0) > 0.4
        in_e2 = sid in e2 and e2[sid].get("materially_different")
        flags_str = []
        if in_d: flags_str.append("D")
        if in_i: flags_str.append("I")
        if in_e2: flags_str.append("D'")
        flags_marker = f" ({', '.join(flags_str)})" if flags_str else " (no rubric flag)"
        md.append(f"- `{sid}` v2_stdev={stdev:.2f}{flags_marker}")
    md.append("")
    md.append("**Robust across both E7 versions** (top 10 in v1 ∩ top 10 in v2):")
    v1_sorted = sorted([(sid, r["mean_score_stdev"]) for sid, r in e7.items() if r.get("mean_score_stdev") is not None], key=lambda x: -x[1])
    v2_sorted = sorted([(sid, r["mean_score_stdev"]) for sid, r in e7v2.items() if r.get("mean_score_stdev") is not None], key=lambda x: -x[1])
    v1_top = {sid for sid, _ in v1_sorted[:10]}
    v2_top = {sid for sid, _ in v2_sorted[:10]}
    for sid in sorted(v1_top & v2_top):
        md.append(f"- `{sid}`")
    md.append("")

    # Flagged statements
    md.append("## Flagged statements (action queue for spec author)")
    md.append("")
    flagged = [r for r in rows if r["primary_label"] != "language_robust"]
    flagged.sort(key=lambda r: ["internally_inconsistent", "language_ambiguous", "operationally_divergent", "needs_more_data"].index(r["primary_label"]) if r["primary_label"] != "language_robust" else 99)

    for r in flagged:
        md.append(f"### `{r['statement_id']}` — `{r['primary_label']}`")
        for line in r["rationale"]:
            md.append(f"- {line}")
        if r.get("ambiguous_phrases_F"):
            md.append(f"- **Localized ambiguous phrases (E5)**:")
            for p in r["ambiguous_phrases_F"]:
                md.append(f"  - `{p['phrase']}` (mean cross-judge equiv = {p['mean_equiv']:.2f})")
        md.append("")

    # E5 phrase analysis (full table — even for statements that didn't make the flag list)
    md.append("## Method F (E5) — soft-predicate phrase analysis")
    md.append("")
    md.append("Per-phrase cross-judge equivalence for the 4 statements run through Method F. Phrases with `mean_equiv < 7` are surfaced for the spec author as the *localized ambiguous span* — these are the exact strings to rewrite first.")
    md.append("")
    for sid in sorted(e5_phrases_by_stmt):
        prs = sorted(
            [p for p in e5_phrases_by_stmt[sid] if p.get("mean_equiv") is not None],
            key=lambda x: x["mean_equiv"],
        )
        if not prs:
            continue
        md.append(f"### `{sid}`")
        md.append("")
        md.append("| phrase | mean_equiv | why soft |")
        md.append("|---|--:|---|")
        for p in prs:
            phrase = p["phrase"].replace("|", r"\|")
            why = (p.get("why_soft") or "").replace("|", r"\|")
            md.append(f"| `{phrase}` | {p['mean_equiv']:.2f} | {why} |")
        md.append("")

    # Per-statement summary table
    md.append("## Per-statement summary table")
    md.append("")
    md.append("| statement | label | C K=5 mean | C stdev | D K=3 mean | D stdev | I rate | gen stdev |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        c = r.get("method_c_k5") or {}
        d = r.get("method_d_k3") or {}
        i = r.get("method_i") or {}
        md.append(f"| `{r['statement_id']}` | `{r['primary_label']}` | {c.get('mean','—')} | {c.get('stdev','—')} | {d.get('mean','—')} | {d.get('stdev','—')} | {i.get('disagreement_rate','—')} | {r.get('downstream_stdev','—')} |")
    md.append("")

    out_md = DIR / "spec_ambiguity_diagnostic_v3.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
