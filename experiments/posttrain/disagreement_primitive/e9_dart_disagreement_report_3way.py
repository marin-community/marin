"""DART disagreement report — 3-compiler version (GPT-5.1 + Gemini 3 Pro + Claude Sonnet 4.6).

Generalizes e9_dart_disagreement_report.py to N=3 compilers. Per statement:
  - Tally 3 diagnoses → diagnostic consensus tier (3-of-3, 2-of-3, 3-way split)
  - For each pair of compilers (3 pairs), align edits + ensemble-classify direction
  - Aggregate to overall tier (T1-T4 with 3-compiler logic)

Output: .agents/logbooks/dart_disagreement_report_3way.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, load_jsonl, write_jsonl
from e9_dart_disagreement_report import (
    align_edits,
    classify_direction_gem,
    classify_direction_gpt,
    trunc,
)
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
GPT_PATH = DIR / "dart_diagnoses.jsonl"
GEM_PATH = DIR / "dart_diagnoses_gemini.jsonl"
CLA_PATH = DIR / "dart_diagnoses_claude.jsonl"
OUT_REPORT = Path(".agents/logbooks/dart_disagreement_report_3way.md")
OUT_DATA = DIR / "dart_aligned_pairs_3way.jsonl"


def diagnosis_tier(diags: list[str]) -> str:
    """3 diagnoses → tier by consensus level."""
    c = Counter(diags)
    most_common, count = c.most_common(1)[0]
    if count == 3:
        return "consensus"
    if count == 2:
        return "plurality"
    return "split"


def overall_tier(diag_tier: str, all_pair_directions: list[str]) -> str:
    """Combine diagnostic-tier + edit-pair direction classifications → T1/T2/T3/T4."""
    if diag_tier == "split":
        return "T4"
    # Any opposite-direction pair → T3 regardless of diagnosis-tier
    if any(d == "opposite_direction" for d in all_pair_directions):
        return "T3"
    if any(d == "different_scope" for d in all_pair_directions):
        return "T2"
    if any(d == "disputed" for d in all_pair_directions):
        return "T2"  # disputed direction → light review
    return "T1"


def ensemble_direction(cls_gpt: dict, cls_gem: dict) -> str:
    g = cls_gpt.get("direction")
    m = cls_gem.get("direction")
    if g == m and g is not None:
        return g
    return "disputed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_REPORT)
    args = ap.parse_args()

    gpt = {r["statement_id"]: r for r in load_jsonl(GPT_PATH) if "error" not in r}
    gem = {r["statement_id"]: r for r in load_jsonl(GEM_PATH) if "error" not in r}
    cla = {r["statement_id"]: r for r in load_jsonl(CLA_PATH) if "error" not in r}
    common = sorted(set(gpt) & set(gem) & set(cla))
    print(f"3-way comparison on {len(common)} statements")

    rubrics_v1 = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics.jsonl") if "error" not in r}

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    g = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    log = RawAPILogger("e9_dart_disagreement_classifier_3way")
    print(f"  raw log dir: {log.run_dir}\n")

    # For each statement, compute 3 pairwise alignments and classify each aligned pair
    out_rows = []
    n_pairs_classified = 0
    PAIR_NAMES = [("gpt", "gem"), ("gpt", "cla"), ("gem", "cla")]
    diagnosers = {"gpt": gpt, "gem": gem, "cla": cla}

    for sid in common:
        diags = {n: diagnosers[n][sid]["diagnosis"] for n in ("gpt", "gem", "cla")}
        recs = {n: diagnosers[n][sid]["recommendation"] for n in ("gpt", "gem", "cla")}
        diag_tier = diagnosis_tier(list(diags.values()))

        v1_anchors = (rubrics_v1.get(sid, {}) or {}).get("anchors", {})
        # 3 pairwise alignments
        pair_alignments = {}
        for a, b in PAIR_NAMES:
            r_a, r_b = diagnosers[a][sid], diagnosers[b][sid]
            pair_alignments[(a, b)] = align_edits(r_a, r_b)

        # Classify each aligned pair (rubric or spec) in each pair-of-compilers
        all_directions = []
        for (a, b), pairs in pair_alignments.items():
            for p in pairs["rubric_pairs"]:
                anchor_k = p["anchor"]
                anc_v1 = v1_anchors.get(anchor_k, {}) or {}
                original = anc_v1.get("criterion", "")
                edit_a = p[f"{a}_edit"].get("new_criterion") if f"{a}_edit" in p else p["gpt_edit"].get("new_criterion")
                edit_b_text = p[f"{b}_edit"].get("new_criterion") if f"{b}_edit" in p else p["gem_edit"].get("new_criterion")
                # `align_edits` keys gpt_edit/gem_edit, but here we may have any pair. Map.
                edit_a = p["gpt_edit"].get("new_criterion") if a == "gpt" else (
                         p["gem_edit"].get("new_criterion") if a == "gem" else p["gpt_edit"].get("new_criterion"))
                # Simpler: we know align_edits was called as align_edits(r_a, r_b) and stores as gpt/gem keys
                edit_a_text = p["gpt_edit"].get("new_criterion")
                edit_b_text = p["gem_edit"].get("new_criterion")
                try:
                    cls_gpt = classify_direction_gpt(log, oai, sid, f"rubric anchor {anchor_k} ({a} vs {b})",
                                                    original, edit_a_text, edit_b_text)
                except Exception as e:
                    cls_gpt = {"direction": "ERROR", "rationale": str(e)[:200]}
                try:
                    cls_gem = classify_direction_gem(log, g, sid, f"rubric anchor {anchor_k} ({a} vs {b})",
                                                    original, edit_a_text, edit_b_text)
                except Exception as e:
                    cls_gem = {"direction": "ERROR", "rationale": str(e)[:200]}
                p["pair"] = (a, b)
                p["classification_gpt"] = cls_gpt
                p["classification_gem"] = cls_gem
                p["ensemble"] = ensemble_direction(cls_gpt, cls_gem)
                all_directions.append(p["ensemble"])
                n_pairs_classified += 1

            for p in pairs["spec_pairs"]:
                original = p["gpt_edit"].get("old_phrase", "") or p["gem_edit"].get("old_phrase", "")
                edit_a_text = p["gpt_edit"].get("new_phrase", "")
                edit_b_text = p["gem_edit"].get("new_phrase", "")
                try:
                    cls_gpt = classify_direction_gpt(log, oai, sid, f"spec phrase ({a} vs {b}): {original[:60]}",
                                                    original, edit_a_text, edit_b_text)
                except Exception as e:
                    cls_gpt = {"direction": "ERROR", "rationale": str(e)[:200]}
                try:
                    cls_gem = classify_direction_gem(log, g, sid, f"spec phrase ({a} vs {b}): {original[:60]}",
                                                    original, edit_a_text, edit_b_text)
                except Exception as e:
                    cls_gem = {"direction": "ERROR", "rationale": str(e)[:200]}
                p["pair"] = (a, b)
                p["classification_gpt"] = cls_gpt
                p["classification_gem"] = cls_gem
                p["ensemble"] = ensemble_direction(cls_gpt, cls_gem)
                all_directions.append(p["ensemble"])
                n_pairs_classified += 1

        tier = overall_tier(diag_tier, all_directions)
        out_rows.append({
            "statement_id": sid,
            "diagnoses": diags,
            "recommendations": recs,
            "diagnosis_tier": diag_tier,
            "tier": tier,
            "pair_alignments": {f"{a}_{b}": pair_alignments[(a, b)] for a, b in PAIR_NAMES},
            "ensemble_directions": all_directions,
        })
        d_str = ", ".join(f"{n}={d}" for n, d in diags.items())
        print(f"  {sid:42s} {tier} diag_tier={diag_tier:9s}  [{d_str}]")

    write_jsonl(out_rows, OUT_DATA)
    print(f"\nclassified {n_pairs_classified} pair-classifications across 3 compiler-pairs")

    # Render report
    parts = ["# DART 3-way Disagreement Report — GPT-5.1 + Gemini 3 Pro + Claude Sonnet 4.6\n"]
    parts.append(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    parts.append(f"Statements: {len(out_rows)}")
    parts.append(f"Pair-classifications: {n_pairs_classified} (each by both GPT and Gemini classifiers)\n")

    parts.append("## Triage table\n")
    parts.append("Tier definitions (3-compiler version):")
    parts.append("- **T1** — diagnosis 3-of-3 OR 2-of-3 plurality, all aligned-edit pairs same-direction → action-safe with normal review")
    parts.append("- **T2** — diagnosis ≥ plurality, at least one different-scope or disputed-direction pair → light review")
    parts.append("- **T3** ⚠️ — at least one OPPOSITE-direction edit pair (any of the 3 compiler-pairs) → flag for spec-author review")
    parts.append("- **T4** — 3-way diagnostic split → genuinely contested; all 3 proposals visible to spec authors\n")

    # Sort by tier
    tier_order = {"T3": 0, "T4": 1, "T2": 2, "T1": 3}
    sorted_rows = sorted(out_rows, key=lambda r: (tier_order.get(r["tier"], 9), r["statement_id"]))
    parts.append("| tier | statement | diag GPT | diag Gemini | diag Claude | diag tier | rec GPT | rec Gemini | rec Claude |")
    parts.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted_rows:
        d = r["diagnoses"]; rec = r["recommendations"]
        parts.append(f"| **{r['tier']}** | {r['statement_id']} | {d['gpt']} | {d['gem']} | {d['cla']} | {r['diagnosis_tier']} | {rec['gpt']} | {rec['gem']} | {rec['cla']} |")
    parts.append("")

    parts.append("## Tier counts\n")
    tcount = Counter(r["tier"] for r in out_rows)
    dcount = Counter(r["diagnosis_tier"] for r in out_rows)
    for t in ("T1", "T2", "T3", "T4"):
        parts.append(f"- **{t}**: {tcount.get(t, 0)} statements")
    parts.append("")
    parts.append("Diagnosis-tier breakdown:")
    for d in ("consensus", "plurality", "split"):
        parts.append(f"- **{d}**: {dcount.get(d, 0)} statements")
    parts.append("")

    # Per-statement detail
    for r in sorted_rows:
        sid = r["statement_id"]
        parts.append(f"\n---\n\n## {r['tier']} — `{sid}`\n")
        parts.append(f"### Diagnoses (tier: {r['diagnosis_tier']})\n")
        parts.append(f"- **GPT-5.1**: {r['diagnoses']['gpt']} → {r['recommendations']['gpt']}")
        parts.append(f"- **Gemini 3 Pro**: {r['diagnoses']['gem']} → {r['recommendations']['gem']}")
        parts.append(f"- **Claude Sonnet 4.6**: {r['diagnoses']['cla']} → {r['recommendations']['cla']}\n")

        # Per-pair edit classifications
        for pair_key, pa in r["pair_alignments"].items():
            a_name, b_name = pair_key.split("_")
            label_map = {"gpt": "GPT", "gem": "Gemini", "cla": "Claude"}
            label = f"{label_map[a_name]} ↔ {label_map[b_name]}"
            if not (pa["rubric_pairs"] or pa["spec_pairs"]):
                continue
            parts.append(f"### Edit alignments — {label}\n")
            for p in pa["rubric_pairs"]:
                a_k = p["anchor"]
                ensemble = p.get("ensemble", "?")
                emoji = "🔥" if ensemble == "opposite_direction" else ("⚠️" if ensemble == "disputed" else "✓")
                cls_g = p["classification_gpt"].get("direction", "?")
                cls_m = p["classification_gem"].get("direction", "?")
                parts.append(f"- {emoji} **Anchor {a_k}** — ensemble: **{ensemble}** (GPT-classifier: {cls_g}, Gemini-classifier: {cls_m})")
            for p in pa["spec_pairs"]:
                ensemble = p.get("ensemble", "?")
                emoji = "🔥" if ensemble == "opposite_direction" else ("⚠️" if ensemble == "disputed" else "✓")
                cls_g = p["classification_gpt"].get("direction", "?")
                cls_m = p["classification_gem"].get("direction", "?")
                parts.append(f"- {emoji} **Spec phrase**: `{trunc(p['gpt_edit'].get('old_phrase'), 100)}` — ensemble: **{ensemble}** (GPT-classifier: {cls_g}, Gemini-classifier: {cls_m})")
            parts.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
