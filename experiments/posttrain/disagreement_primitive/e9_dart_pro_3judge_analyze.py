"""3-judge α with GPT+Pro+Claude where data permits, GPT+Pro elsewhere.
Re-buckets all 46 statements at T₁=0.5.
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

import sys
sys.path.insert(0, "experiments/posttrain/disagreement_primitive")
from e9_kappa_alternatives import krippendorff_alpha_interval

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT = Path(".agents/logbooks/dart_pro_judge_audit.md")
T1 = 0.5


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


# Build the 3-judge ensemble: GPT + Gemini-Pro + Claude (Pro replaces Flash entirely)
baseline = load_jsonl(DIR / "per_judgment_opposite.jsonl")
pro = load_jsonl(DIR / "per_judgment_pro_audit.jsonl")

# Per-cell: by_cell[(sid, scen, gen, condition)][judge] = score
by_cell = defaultdict(dict)
for r in baseline:
    if r.get("score") is None: continue
    if r.get("judge") in ("gpt", "claude") and r.get("condition") in ("variant_A", "rubric_plus_spec"):
        ck = (r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
        by_cell[ck][r["judge"]] = r["score"]
for r in pro:
    if r.get("score") is None: continue
    ck = (r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
    by_cell[ck]["gemini-pro"] = r["score"]


def alpha_per_sid(condition, judges):
    by_sid = defaultdict(list)
    for ck, scores in by_cell.items():
        sid, scen, gen, cond = ck
        if cond != condition: continue
        triple = tuple(scores.get(j) for j in judges)
        by_sid[sid].append(triple)
    out = {}
    for sid, triples in by_sid.items():
        full = [t for t in triples if all(x is not None for x in t)]
        a = krippendorff_alpha_interval(full) if full else None
        out[sid] = (a, len(full), len(triples))
    return out


# 3-judge with Pro: gpt + gemini-pro + claude (only 8 statements have all 3)
gpro3_v = alpha_per_sid("variant_A", ("gpt", "gemini-pro", "claude"))
gpro3_p = alpha_per_sid("rubric_plus_spec", ("gpt", "gemini-pro", "claude"))

# 2-judge with Pro: gpt + gemini-pro (covers all 46)
gpro2_v = alpha_per_sid("variant_A", ("gpt", "gemini-pro"))
gpro2_p = alpha_per_sid("rubric_plus_spec", ("gpt", "gemini-pro"))

# 2-judge with Flash baseline: gpt + gemini (Flash) — for comparison
def alpha_with_flash(condition, judges):
    by_sid = defaultdict(list)
    for r in baseline:
        if r.get("score") is None: continue
        if r.get("condition") != condition: continue
        if r.get("judge") not in ("gpt", "gemini", "claude"): continue
        ck = (r["statement_id"], r["scenario_idx"], r["generator"])
        # Need to track per-cell judge scores
    # Simpler: use by_cell extended with "gemini" (Flash) instead
    extra = defaultdict(dict)
    for r in baseline:
        if r.get("score") is None: continue
        if r.get("condition") != condition: continue
        if r.get("judge") not in ("gpt", "gemini", "claude"): continue
        ck = (r["statement_id"], r["scenario_idx"], r["generator"])
        extra[ck][r["judge"]] = r["score"]
    by_sid = defaultdict(list)
    for ck, scores in extra.items():
        sid = ck[0]
        triple = tuple(scores.get(j) for j in judges)
        by_sid[sid].append(triple)
    out = {}
    for sid, triples in by_sid.items():
        full = [t for t in triples if all(x is not None for x in t)]
        a = krippendorff_alpha_interval(full) if full else None
        out[sid] = (a, len(full), len(triples))
    return out


flash3_v = alpha_with_flash("variant_A", ("gpt", "gemini", "claude"))
flash3_p = alpha_with_flash("rubric_plus_spec", ("gpt", "gemini", "claude"))
flash2_v = alpha_with_flash("variant_A", ("gpt", "gemini"))
flash2_p = alpha_with_flash("rubric_plus_spec", ("gpt", "gemini"))


def fmt(x):
    return f"{x:+.3f}" if x is not None else "n/a"


def bucket(a_bare, a_p4):
    if a_bare is None or a_p4 is None: return "?"
    if a_bare >= T1 and a_p4 >= T1: return "A"
    if a_bare < T1 and a_p4 >= T1: return "B"
    if a_bare >= T1 and a_p4 < T1: return "C"
    return "D"


sids = sorted(set(gpro2_v) | set(gpro2_p))

parts = ["# DART Pro-judge audit — 3-judge ensemble (GPT-5.1 + Gemini-3.1-Pro + Claude Sonnet 4.6)", ""]
parts.append("**Generated**: 2026-05-10")
parts.append("")
parts.append("## Setup")
parts.append("")
parts.append("Replaces Gemini-3-Flash with **Gemini-3.1-Pro** in the judge ensemble. Pro judging done with `thinking_level=\"low\"` + `temperature=0` (per dart.md Gotcha 17). All 46 spec statements × 4 generators × 2 conditions (variant_A bare, rubric_plus_spec phase_4).")
parts.append("")
parts.append("**Coverage caveat**: Claude has judgments on only 8 of 46 statements (carry-over from older work — Claude was added late and never re-judged the full spec). True 3-judge α is computable on those 8; for the remaining 38 statements we report 2-judge α (GPT + Pro).")
parts.append("")
parts.append("## Headline result: per-statement Pro-2j α (GPT + Pro), all 46 statements")
parts.append("")
parts.append("| statement | 2j α_bare (Pro) | 2j α_p4 (Pro) | 2j α_p4 (Flash) | Δ_p4 (Pro−Flash) | Pro bucket | Flash bucket | shift |")
parts.append("|---|--:|--:|--:|--:|:-:|:-:|---|")

table_rows = []
for sid in sids:
    pv = gpro2_v.get(sid, (None, 0, 0))[0]
    pp = gpro2_p.get(sid, (None, 0, 0))[0]
    fv = flash2_v.get(sid, (None, 0, 0))[0]
    fp = flash2_p.get(sid, (None, 0, 0))[0]
    bp = bucket(pv, pp)
    bf = bucket(fv, fp)
    delta = (pp - fp) if (pp is not None and fp is not None) else None
    shift = "" if bp == bf else f"{bf}→{bp}"
    table_rows.append((sid, pv, pp, fp, delta, bp, bf, shift))

table_rows.sort(key=lambda r: -(r[2] if r[2] is not None else -99))

for row in table_rows:
    sid, pv, pp, fp, delta, bp, bf, shift = row
    parts.append(f"| {sid} | {fmt(pv)} | {fmt(pp)} | {fmt(fp)} | {fmt(delta)} | {bp} | {bf} | {shift} |")

parts.append("")

# Bucket distributions
pro_bk = Counter(bucket(gpro2_v.get(s,(None,0,0))[0], gpro2_p.get(s,(None,0,0))[0]) for s in sids)
flash_bk = Counter(bucket(flash2_v.get(s,(None,0,0))[0], flash2_p.get(s,(None,0,0))[0]) for s in sids)

parts.append("## Bucket distribution at T₁=0.5 (2-judge α: GPT + Pro vs. GPT + Flash)\n")
parts.append("| bucket | Pro judge | Flash judge |")
parts.append("|---|--:|--:|")
for b in ["A", "B", "C", "D", "?"]:
    parts.append(f"| {b} | {pro_bk.get(b,0)} | {flash_bk.get(b,0)} |")
parts.append("")

# Δα distribution
deltas = [(r[3]) for r in table_rows if r[3] is not None]
deltas_v = []
for sid in sids:
    pv = gpro2_v.get(sid, (None,0,0))[0]
    fv = flash2_v.get(sid, (None,0,0))[0]
    if pv is not None and fv is not None:
        deltas_v.append(pv - fv)

parts.append("## Δα distribution (Pro − Flash)")
parts.append("")
for label, ds in [("variant_A (bare)", deltas_v), ("rubric_plus_spec (phase_4)", deltas)]:
    if ds:
        ds_sorted = sorted(ds)
        n_better = sum(1 for d in ds if d > 0.05)
        n_worse = sum(1 for d in ds if d < -0.05)
        parts.append(f"\n### {label} (n={len(ds)})")
        parts.append(f"- mean: {sum(ds)/len(ds):+.3f}")
        parts.append(f"- median: {ds_sorted[len(ds_sorted)//2]:+.3f}")
        parts.append(f"- min: {min(ds):+.3f}, max: {max(ds):+.3f}")
        parts.append(f"- statements where Pro improves α (Δ > +0.05): {n_better}")
        parts.append(f"- statements where Pro reduces α (Δ < −0.05): {n_worse}")
        parts.append(f"- statements where ~unchanged: {len(ds) - n_better - n_worse}")

parts.append("")

# 3-judge subset
if gpro3_p:
    parts.append("## 3-judge α on the 8 statements with Claude data (GPT + Pro + Claude)")
    parts.append("")
    parts.append("| statement | 3j Pro α_bare | 3j Pro α_p4 | 3j Flash α_p4 | Δ_p4 | Pro bucket | Flash bucket |")
    parts.append("|---|--:|--:|--:|--:|:-:|:-:|")
    rows3 = []
    for sid in sorted(gpro3_p):
        pv = gpro3_v.get(sid, (None, 0, 0))[0]
        pp = gpro3_p.get(sid, (None, 0, 0))[0]
        fp = flash3_p.get(sid, (None, 0, 0))[0]
        fv = flash3_v.get(sid, (None, 0, 0))[0]
        if pp is None: continue
        rows3.append((sid, pv, pp, fp, fv))
    rows3.sort(key=lambda r: -(r[2] if r[2] is not None else -99))
    for sid, pv, pp, fp, fv in rows3:
        bp = bucket(pv, pp)
        bf = bucket(fv, fp)
        delta = (pp - fp) if (pp is not None and fp is not None) else None
        parts.append(f"| {sid} | {fmt(pv)} | {fmt(pp)} | {fmt(fp)} | {fmt(delta)} | {bp} | {bf} |")
parts.append("")

OUT.write_text("\n".join(parts))
print(f"wrote {OUT}")
print(f"\nPro buckets (2-judge):   {dict(pro_bk)}")
print(f"Flash buckets (2-judge): {dict(flash_bk)}")
if deltas:
    print(f"\nphase_4 Δα Pro−Flash: mean={sum(deltas)/len(deltas):+.3f}, n_better={sum(1 for d in deltas if d>0.05)}, n_worse={sum(1 for d in deltas if d<-0.05)}")
