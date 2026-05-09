"""Compare DART Run 1 (GPT-5.1) vs Run 2 (Gemini 3 Pro) compiler diagnoses.

Tests compiler-stability of DART: do the two compilers AGREE on diagnosis,
recommendation, and edit proposals on the same 14 Bucket D statements?

If they agree → DART's diagnoses are model-independent (high confidence).
If they disagree → diagnoses are LM-specific (need ensembling or human triage).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DIR = Path("experiments/posttrain/disagreement_primitive")
GPT_PATH = DIR / "dart_diagnoses.jsonl"
GEMINI_PATH = DIR / "dart_diagnoses_gemini.jsonl"
OUT = Path(".agents/logbooks/dart_run1_vs_run2_compare.md")


def load(path):
    return {r["statement_id"]: r for r in [json.loads(l) for l in path.open()] if "error" not in r}


def main():
    gpt = load(GPT_PATH)
    gem = load(GEMINI_PATH)
    common = sorted(set(gpt) & set(gem))
    print(f"Comparing {len(common)} statements common to both runs")

    parts = ["# DART Run 1 (GPT-5.1) vs Run 2 (Gemini 3 Pro) — compiler comparison\n"]
    parts.append(f"Statements compared: {len(common)}\n")

    # Top-line agreement counts
    diag_match = sum(1 for sid in common if gpt[sid]["diagnosis"] == gem[sid]["diagnosis"])
    rec_match = sum(1 for sid in common if gpt[sid]["recommendation"] == gem[sid]["recommendation"])

    parts.append(f"## Top-line agreement\n")
    parts.append(f"- **Diagnosis agreement**: {diag_match}/{len(common)} = {100*diag_match/max(1,len(common)):.0f}%")
    parts.append(f"- **Recommendation agreement**: {rec_match}/{len(common)} = {100*rec_match/max(1,len(common)):.0f}%\n")

    # Diagnosis distributions
    gpt_diag = Counter(gpt[sid]["diagnosis"] for sid in common)
    gem_diag = Counter(gem[sid]["diagnosis"] for sid in common)
    parts.append(f"### Diagnosis distribution\n")
    parts.append("| diagnosis | GPT-5.1 | Gemini 3 Pro |")
    parts.append("|---|--:|--:|")
    for d in ("rubric_drift", "spec_ambiguity", "both", "irreducible"):
        parts.append(f"| {d} | {gpt_diag.get(d, 0)} | {gem_diag.get(d, 0)} |")
    parts.append("")

    # Per-statement comparison table
    parts.append("## Per-statement comparison\n")
    parts.append("| statement | GPT diagnosis | Gemini diagnosis | match? | GPT rec | Gemini rec | match? |")
    parts.append("|---|---|---|:-:|---|---|:-:|")
    for sid in common:
        g = gpt[sid]
        m = gem[sid]
        diag_ok = "✓" if g["diagnosis"] == m["diagnosis"] else "✗"
        rec_ok = "✓" if g["recommendation"] == m["recommendation"] else "✗"
        parts.append(f"| {sid} | {g['diagnosis']} | {m['diagnosis']} | {diag_ok} | {g['recommendation']} | {m['recommendation']} | {rec_ok} |")
    parts.append("")

    # Disagreement detail — for each disagreement, show evidence
    disagree_diag = [sid for sid in common if gpt[sid]["diagnosis"] != gem[sid]["diagnosis"]]
    if disagree_diag:
        parts.append(f"## Disagreement detail ({len(disagree_diag)} statements)\n")
        for sid in disagree_diag:
            g = gpt[sid]
            m = gem[sid]
            parts.append(f"\n### `{sid}`\n")
            parts.append(f"- **GPT-5.1**: diagnosis={g['diagnosis']}, recommendation={g['recommendation']}")
            parts.append(f"  - {g.get('evidence_summary','')[:300]}")
            parts.append(f"- **Gemini 3 Pro**: diagnosis={m['diagnosis']}, recommendation={m['recommendation']}")
            parts.append(f"  - {m.get('evidence_summary','')[:300]}")
            n_gpt_re = len(g.get("rubric_edits") or [])
            n_gpt_se = len(g.get("spec_edits_for_author_review") or [])
            n_gem_re = len(m.get("rubric_edits") or [])
            n_gem_se = len(m.get("spec_edits_for_author_review") or [])
            parts.append(f"- Edits: GPT proposes {n_gpt_re} rubric / {n_gpt_se} spec; Gemini proposes {n_gem_re} rubric / {n_gem_se} spec")

    # Agreement detail summary
    agree_diag = [sid for sid in common if gpt[sid]["diagnosis"] == gem[sid]["diagnosis"]]
    if agree_diag:
        parts.append(f"\n## Agreement detail ({len(agree_diag)} statements)\n")
        parts.append("Statements where both compilers reached the same diagnosis:\n")
        by_d = {}
        for sid in agree_diag:
            d = gpt[sid]["diagnosis"]
            by_d.setdefault(d, []).append(sid)
        for d in sorted(by_d):
            parts.append(f"- **{d}** ({len(by_d[d])}): {', '.join(by_d[d])}")
        parts.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    main()
