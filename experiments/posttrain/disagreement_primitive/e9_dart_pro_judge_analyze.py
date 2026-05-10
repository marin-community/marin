"""Analyze the Pro-judge audit: per-statement α with Gemini-Pro replacing Gemini-Flash
in the 3-judge ensemble, compared to baseline Flash α. Re-derives the A/B/C/D bucketing.

Output: .agents/logbooks/dart_pro_judge_audit.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from e9_kappa_alternatives import krippendorff_alpha_interval

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT = Path(".agents/logbooks/dart_pro_judge_audit.md")

T1 = 0.5


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def load_combined_with_judge_swap(swap_judge: bool) -> list[dict]:
    """Build the combined judgment set:
       - GPT and Claude rows from per_judgment_opposite.jsonl (existing)
       - Gemini rows from EITHER per_judgment_opposite.jsonl (Flash, swap_judge=False)
                       OR per_judgment_pro_audit.jsonl (Pro, swap_judge=True)
    """
    rows = []
    baseline = load_jsonl(DIR / "per_judgment_opposite.jsonl")
    pro = load_jsonl(DIR / "per_judgment_pro_audit.jsonl") if swap_judge else []
    pro_keys = {(r["statement_id"], r["scenario_idx"], r["generator"], r["condition"]): r
                for r in pro if r.get("score") is not None}

    for r in baseline:
        if r.get("score") is None: continue
        cond = r.get("condition")
        if cond not in ("variant_A", "rubric_plus_spec"): continue
        if r.get("judge") == "gemini" and swap_judge:
            ck = (r["statement_id"], r["scenario_idx"], r["generator"], cond)
            pro_row = pro_keys.get(ck)
            if pro_row is not None:
                # substitute Pro score
                rows.append({**r, "score": pro_row["score"], "judge": "gemini"})
            # else: drop Flash row entirely if we're meant to swap and have no Pro replacement
        else:
            rows.append(r)
    return rows


def alpha_per_statement(rows: list[dict], condition: str) -> dict:
    by_sid_cell: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("condition") != condition: continue
        sid = r.get("statement_id")
        if not sid: continue
        ck = (r["scenario_idx"], r["generator"])
        by_sid_cell[sid][ck][r["judge"]] = r["score"]

    out = {}
    for sid, by_cell in by_sid_cell.items():
        triples = []
        for ck, scores in by_cell.items():
            triple = (scores.get("gpt"), scores.get("gemini"), scores.get("claude"))
            triples.append(triple)
        a = krippendorff_alpha_interval(triples)
        n_full = sum(1 for t in triples if all(x is not None for x in t))
        n_any = len(triples)
        out[sid] = (a, n_full, n_any)
    return out


def main():
    print("Loading judgments...")
    rows_flash = load_combined_with_judge_swap(swap_judge=False)
    rows_pro = load_combined_with_judge_swap(swap_judge=True)
    print(f"  Flash-judge ensemble: {len(rows_flash)} judgments")
    print(f"  Pro-judge   ensemble: {len(rows_pro)} judgments")

    # Per-statement α under both
    print("\nComputing α per statement...")
    flash_bare = alpha_per_statement(rows_flash, "variant_A")
    flash_phase4 = alpha_per_statement(rows_flash, "rubric_plus_spec")
    pro_bare = alpha_per_statement(rows_pro, "variant_A")
    pro_phase4 = alpha_per_statement(rows_pro, "rubric_plus_spec")

    sids = sorted(set(flash_bare) | set(flash_phase4) | set(pro_bare) | set(pro_phase4))

    def bucket_label(a_bare, a_phase4):
        if a_bare is None or a_phase4 is None:
            return "?"
        if a_bare >= T1 and a_phase4 >= T1: return "A"
        if a_bare < T1 and a_phase4 >= T1: return "B"
        if a_bare >= T1 and a_phase4 < T1: return "C"
        return "D"

    parts = ["# DART Pro-judge audit — Gemini-3.1-Pro replaces Gemini-3-Flash as judge", ""]
    parts.append(f"Generated: 2026-05-10")
    parts.append(f"All 46 spec statements × all 4 generators × 2 conditions (variant_A bare, rubric_plus_spec phase_4).")
    parts.append(f"Pro-judge call config: `gemini-3.1-pro-preview` + `thinking_level=\"low\"` + `temperature=0`.")
    parts.append(f"Flash baseline judgments come from `per_judgment_opposite.jsonl`; Pro replacements from `per_judgment_pro_audit.jsonl`.\n")

    parts.append("## Per-statement α and bucket (sorted by Pro-phase4 α descending)\n")
    parts.append("| statement | Flash α_bare | Flash α_phase4 | Flash bucket | Pro α_bare | Pro α_phase4 | Pro bucket | Δα_phase4 | bucket shift |")
    parts.append("|---|--:|--:|:-:|--:|--:|:-:|--:|---|")

    rows_table = []
    for sid in sids:
        af_b, _, _ = flash_bare.get(sid, (None, 0, 0))
        af_p, _, _ = flash_phase4.get(sid, (None, 0, 0))
        ap_b, _, _ = pro_bare.get(sid, (None, 0, 0))
        ap_p, _, _ = pro_phase4.get(sid, (None, 0, 0))
        bf = bucket_label(af_b, af_p)
        bp = bucket_label(ap_b, ap_p)
        delta = (ap_p - af_p) if (ap_p is not None and af_p is not None) else None
        shift = "" if bf == bp else f"{bf}→{bp}"
        rows_table.append((sid, af_b, af_p, bf, ap_b, ap_p, bp, delta, shift))

    rows_table.sort(key=lambda r: -(r[5] if r[5] is not None else -99))

    def fmt(x):
        return f"{x:+.3f}" if x is not None else "n/a"

    for sid, af_b, af_p, bf, ap_b, ap_p, bp, delta, shift in rows_table:
        parts.append(f"| {sid} | {fmt(af_b)} | {fmt(af_p)} | {bf} | {fmt(ap_b)} | {fmt(ap_p)} | {bp} | {fmt(delta)} | {shift} |")
    parts.append("")

    # Bucket counts
    flash_buckets = [bucket_label(flash_bare.get(s,(None,0,0))[0], flash_phase4.get(s,(None,0,0))[0]) for s in sids]
    pro_buckets = [bucket_label(pro_bare.get(s,(None,0,0))[0], pro_phase4.get(s,(None,0,0))[0]) for s in sids]

    from collections import Counter
    fc = Counter(flash_buckets)
    pc = Counter(pro_buckets)
    parts.append("## Bucket distributions\n")
    parts.append("| bucket | Flash judge | Pro judge |")
    parts.append("|---|--:|--:|")
    for b in ["A","B","C","D","?"]:
        parts.append(f"| {b} | {fc.get(b,0)} | {pc.get(b,0)} |")
    parts.append("")

    # Δα distribution
    deltas = [r[7] for r in rows_table if r[7] is not None]
    if deltas:
        deltas.sort()
        parts.append("## Δα (Pro − Flash) distribution on phase_4\n")
        parts.append(f"- n = {len(deltas)}")
        parts.append(f"- mean: {sum(deltas)/len(deltas):+.3f}")
        parts.append(f"- median: {deltas[len(deltas)//2]:+.3f}")
        parts.append(f"- min: {min(deltas):+.3f}, max: {max(deltas):+.3f}")
        n_better = sum(1 for d in deltas if d > 0.05)
        n_worse = sum(1 for d in deltas if d < -0.05)
        n_same = len(deltas) - n_better - n_worse
        parts.append(f"- |Δ| ≤ 0.05: {n_same} statements (~unchanged)")
        parts.append(f"- Δ > 0.05 (Pro improves agreement): {n_better} statements")
        parts.append(f"- Δ < -0.05 (Pro reduces agreement): {n_worse} statements")
        parts.append("")

    # Bucket-D specific
    parts.append("## Bucket D (the 14 DART work-surface statements)\n")
    BUCKET_D = ["avoid_abuse","comply_with_laws","no_topic_off_limits","sexual_content_involving_minors",
                "be_clear","assume_objective_pov","do_not_lie","formatting","protect_privileged_messages",
                "refusal_style","letter_and_spirit","be_thorough_but_efficient","ask_clarifying_questions",
                "prevent_imminent_harm"]
    parts.append("| statement | Flash α_phase4 | Pro α_phase4 | Δ | Flash bucket | Pro bucket |")
    parts.append("|---|--:|--:|--:|:-:|:-:|")
    for sid in BUCKET_D:
        af_b, _, _ = flash_bare.get(sid, (None,0,0))
        af_p, _, _ = flash_phase4.get(sid, (None,0,0))
        ap_b, _, _ = pro_bare.get(sid, (None,0,0))
        ap_p, _, _ = pro_phase4.get(sid, (None,0,0))
        bf = bucket_label(af_b, af_p)
        bp = bucket_label(ap_b, ap_p)
        d = (ap_p - af_p) if (ap_p is not None and af_p is not None) else None
        parts.append(f"| {sid} | {fmt(af_p)} | {fmt(ap_p)} | {fmt(d)} | {bf} | {bp} |")
    parts.append("")

    OUT.write_text("\n".join(parts))
    print(f"\nwrote {OUT}")
    print(f"  Flash buckets: {dict(fc)}")
    print(f"  Pro buckets:   {dict(pc)}")


if __name__ == "__main__":
    main()
