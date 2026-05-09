"""DART disagreement report — surfaces edit-level disagreements between
GPT-5.1 (Run 1) and Gemini 3 Pro (Run 2) compiler diagnoses.

Three steps:
1. Auto-align edits per statement (deterministic — by anchor number for
   rubric edits; by old_phrase substring overlap for spec edits).
2. For each aligned pair, classify direction (same / opposite / different_scope)
   via ENSEMBLE of GPT-5.1 + Gemini 3 Pro. Keep both classifications visible.
3. Compute tier per statement (T1-T4) and render markdown report.

Output: .agents/logbooks/dart_disagreement_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, call_gpt_json, load_jsonl, write_jsonl
from e8_phase2_cross_model import _GEMINI_SAFETY_BLOCK_NONE
from e9_dart_compiler_gemini import GEMINI_MODEL, parse_json_strict
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
GPT_PATH = DIR / "dart_diagnoses.jsonl"
GEM_PATH = DIR / "dart_diagnoses_gemini.jsonl"
OUT_REPORT = Path(".agents/logbooks/dart_disagreement_report.md")
OUT_PAIRS = DIR / "dart_aligned_pairs.jsonl"

OVERLAP_THRESHOLD = 0.6  # spec-edit alignment by char overlap


CLASSIFIER_SYSTEM = """You classify whether two proposed text edits push in the SAME direction, OPPOSITE directions, or have DIFFERENT scope.

Context: Two LM compilers independently proposed edits to a Model Spec statement (or its rubric anchor). We want to know whether their proposals are aligned, contradictory, or addressing different aspects.

You will be given:
- LOCUS: which anchor or spec phrase the edits address
- ORIGINAL: the text being modified
- EDIT_A: one compiler's proposed replacement
- EDIT_B: the other compiler's proposed replacement

Classify into one of:
- "same_direction": both edits push the rule the SAME way (both narrow, both broaden, both clarify the same restriction). Wording may differ.
- "opposite_direction": edits push the rule OPPOSITE ways (one narrows, the other broadens; one adds a permission, the other adds a prohibition; etc.)
- "different_scope": both edits push the same direction but address DIFFERENT aspects of the change (e.g., one adds a meta-task carve-out, the other clarifies generator handling)

Be concrete. Quote a phrase from each edit that makes the direction explicit.

OUTPUT (single JSON, no markdown):
{
  "direction": "same_direction" | "opposite_direction" | "different_scope",
  "rationale": "<1-2 sentences quoting load-bearing phrases from each edit>"
}
"""


def _char_overlap(a: str, b: str) -> float:
    """Fraction of shorter-string chars that appear (in order) in the longer."""
    if not a or not b: return 0.0
    a, b = a.lower(), b.lower()
    if a in b or b in a:
        return 1.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    # Sliding-window approximation: longest common contiguous substring length / len(short)
    n = len(short); m = len(long_)
    best = 0
    for size in range(min(n, 80), 0, -1):
        for i in range(0, n - size + 1):
            if short[i:i+size] in long_:
                best = max(best, size)
                break
        if best >= n * OVERLAP_THRESHOLD:
            break
    return best / max(1, n)


def align_edits(g: dict, m: dict) -> dict:
    """Pair GPT and Gemini edits for one statement. Returns:
       {
         'rubric_pairs': [{anchor, gpt_edit, gem_edit}],
         'rubric_unmatched_gpt': [...],
         'rubric_unmatched_gem': [...],
         'spec_pairs': [{old_phrase, gpt_edit, gem_edit, overlap_pct}],
         'spec_unmatched_gpt': [...],
         'spec_unmatched_gem': [...],
       }
    """
    out = {"rubric_pairs": [], "rubric_unmatched_gpt": [], "rubric_unmatched_gem": [],
           "spec_pairs": [], "spec_unmatched_gpt": [], "spec_unmatched_gem": []}

    # Rubric: align by anchor number
    g_by_anchor = {e.get("anchor"): e for e in (g.get("rubric_edits") or [])}
    m_by_anchor = {e.get("anchor"): e for e in (m.get("rubric_edits") or [])}
    for a in sorted(set(g_by_anchor) | set(m_by_anchor)):
        ge, me = g_by_anchor.get(a), m_by_anchor.get(a)
        if ge and me:
            out["rubric_pairs"].append({"anchor": a, "gpt_edit": ge, "gem_edit": me})
        elif ge:
            out["rubric_unmatched_gpt"].append({"anchor": a, "edit": ge})
        elif me:
            out["rubric_unmatched_gem"].append({"anchor": a, "edit": me})

    # Spec: align by old_phrase overlap (substring or ≥ OVERLAP_THRESHOLD char overlap)
    g_specs = list(g.get("spec_edits_for_author_review") or [])
    m_specs = list(m.get("spec_edits_for_author_review") or [])
    matched_g, matched_m = set(), set()
    for i, ge in enumerate(g_specs):
        for j, me in enumerate(m_specs):
            if j in matched_m: continue
            o = _char_overlap(ge.get("old_phrase", ""), me.get("old_phrase", ""))
            if o >= OVERLAP_THRESHOLD:
                out["spec_pairs"].append({
                    "old_phrase_gpt": ge.get("old_phrase"),
                    "old_phrase_gem": me.get("old_phrase"),
                    "gpt_edit": ge, "gem_edit": me, "overlap": o,
                })
                matched_g.add(i); matched_m.add(j)
                break
    for i, ge in enumerate(g_specs):
        if i not in matched_g:
            out["spec_unmatched_gpt"].append({"edit": ge})
    for j, me in enumerate(m_specs):
        if j not in matched_m:
            out["spec_unmatched_gem"].append({"edit": me})
    return out


def classify_direction_gpt(log, oai, sid, locus, original, edit_a, edit_b) -> dict:
    user = (
        f"STATEMENT: {sid}\n"
        f"LOCUS: {locus}\n\n"
        f"ORIGINAL:\n{original}\n\n"
        f"EDIT_A (GPT-5.1):\n{edit_a}\n\n"
        f"EDIT_B (Gemini 3 Pro):\n{edit_b}\n\n"
        "Classify per the schema."
    )
    return call_gpt_json(log, oai, role="dart_classifier_gpt",
                        key={"sid": sid, "locus": locus[:60]},
                        system=CLASSIFIER_SYSTEM, user=user, max_tokens=600)


def classify_direction_gem(log, gem, sid, locus, original, edit_a, edit_b,
                           thinking_budget=128) -> dict:
    user = (
        f"STATEMENT: {sid}\n"
        f"LOCUS: {locus}\n\n"
        f"ORIGINAL:\n{original}\n\n"
        f"EDIT_A (GPT-5.1):\n{edit_a}\n\n"
        f"EDIT_B (Gemini 3 Pro):\n{edit_b}\n\n"
        "Classify per the schema."
    )
    config = types.GenerateContentConfig(
        system_instruction=CLASSIFIER_SYSTEM,
        max_output_tokens=600,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        response_mime_type="application/json",
        safety_settings=_GEMINI_SAFETY_BLOCK_NONE,
    )
    raw = log.call(role="dart_classifier_gem",
                   key={"sid": sid, "locus": locus[:60]},
                   fn=lambda: gem.models.generate_content(
                       model=GEMINI_MODEL, contents=user, config=config))
    return parse_json_strict(raw.text or "")


def trunc(s, n=300):
    s = (s or "").replace("\r", "")
    return s if len(s) <= n else s[:n] + "…"


def compute_tier(diag_match: bool, pair_directions: list[str]) -> str:
    """Map (diagnosis match) × (per-pair directions) → T1/T2/T3.
       T4 returned separately when diagnoses don't match.
    """
    if not diag_match:
        return "T4"
    if not pair_directions:
        return "T1"  # same diagnosis, no aligned-edit pairs to compare
    if any(d == "opposite_direction" for d in pair_directions):
        return "T3"
    if any(d == "different_scope" for d in pair_directions):
        return "T2"
    return "T1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_REPORT)
    args = ap.parse_args()

    gpt = {r["statement_id"]: r for r in load_jsonl(GPT_PATH) if "error" not in r}
    gem = {r["statement_id"]: r for r in load_jsonl(GEM_PATH) if "error" not in r}
    common = sorted(set(gpt) & set(gem))

    # Load original rubrics + spec for ORIGINAL text
    rubrics_v1 = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics.jsonl") if "error" not in r}
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}

    print(f"Aligning + classifying {len(common)} statements")
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    g = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    log = RawAPILogger("e9_dart_disagreement_classifier")
    print(f"  raw log dir: {log.run_dir}\n")

    # Per-statement: align + classify each aligned pair
    out_rows = []
    n_pairs_classified = 0
    for sid in common:
        gpt_r, gem_r = gpt[sid], gem[sid]
        diag_match = gpt_r["diagnosis"] == gem_r["diagnosis"]
        pairs = align_edits(gpt_r, gem_r)

        # Classify each rubric pair
        rubric_pair_directions = []
        for p in pairs["rubric_pairs"]:
            anchor_k = p["anchor"]
            anc_v1 = (rubrics_v1.get(sid, {}) or {}).get("anchors", {}).get(anchor_k, {}) or {}
            original = anc_v1.get("criterion", "")
            edit_a = p["gpt_edit"].get("new_criterion", "")
            edit_b = p["gem_edit"].get("new_criterion", "")
            try:
                cls_gpt = classify_direction_gpt(log, oai, sid, f"rubric anchor {anchor_k}",
                                                 original, edit_a, edit_b)
            except Exception as e:
                cls_gpt = {"direction": "ERROR", "rationale": f"{e}"[:200]}
            try:
                cls_gem = classify_direction_gem(log, g, sid, f"rubric anchor {anchor_k}",
                                                 original, edit_a, edit_b)
            except Exception as e:
                cls_gem = {"direction": "ERROR", "rationale": f"{e}"[:200]}
            p["classification_gpt"] = cls_gpt
            p["classification_gem"] = cls_gem
            # Ensemble: if both agree, use that. Else mark as "disputed".
            if cls_gpt.get("direction") == cls_gem.get("direction"):
                p["ensemble"] = cls_gpt.get("direction")
            else:
                p["ensemble"] = "disputed"
            rubric_pair_directions.append(p["ensemble"])
            n_pairs_classified += 1

        # Classify each spec pair
        spec_pair_directions = []
        for p in pairs["spec_pairs"]:
            original = p["gpt_edit"].get("old_phrase", "") or p["gem_edit"].get("old_phrase", "")
            edit_a = p["gpt_edit"].get("new_phrase", "")
            edit_b = p["gem_edit"].get("new_phrase", "")
            try:
                cls_gpt = classify_direction_gpt(log, oai, sid, f"spec phrase: {original[:80]}",
                                                 original, edit_a, edit_b)
            except Exception as e:
                cls_gpt = {"direction": "ERROR", "rationale": f"{e}"[:200]}
            try:
                cls_gem = classify_direction_gem(log, g, sid, f"spec phrase: {original[:80]}",
                                                 original, edit_a, edit_b)
            except Exception as e:
                cls_gem = {"direction": "ERROR", "rationale": f"{e}"[:200]}
            p["classification_gpt"] = cls_gpt
            p["classification_gem"] = cls_gem
            if cls_gpt.get("direction") == cls_gem.get("direction"):
                p["ensemble"] = cls_gpt.get("direction")
            else:
                p["ensemble"] = "disputed"
            spec_pair_directions.append(p["ensemble"])
            n_pairs_classified += 1

        all_directions = rubric_pair_directions + spec_pair_directions
        tier = compute_tier(diag_match, all_directions)
        out_rows.append({
            "statement_id": sid,
            "diag_gpt": gpt_r["diagnosis"],
            "diag_gem": gem_r["diagnosis"],
            "diag_match": diag_match,
            "tier": tier,
            "rec_gpt": gpt_r["recommendation"],
            "rec_gem": gem_r["recommendation"],
            "n_rubric_pairs": len(pairs["rubric_pairs"]),
            "n_spec_pairs": len(pairs["spec_pairs"]),
            "n_rubric_unmatched_gpt": len(pairs["rubric_unmatched_gpt"]),
            "n_rubric_unmatched_gem": len(pairs["rubric_unmatched_gem"]),
            "n_spec_unmatched_gpt": len(pairs["spec_unmatched_gpt"]),
            "n_spec_unmatched_gem": len(pairs["spec_unmatched_gem"]),
            "pairs": pairs,
            "ensemble_directions": all_directions,
        })
        print(f"  {sid:42s} {tier:3s} diag={'=' if diag_match else '≠':1s} rubric_pairs={len(pairs['rubric_pairs'])} spec_pairs={len(pairs['spec_pairs'])} ensemble={all_directions}")

    # Persist
    write_jsonl(out_rows, OUT_PAIRS)
    print(f"\nclassified {n_pairs_classified} aligned pairs")
    print(f"wrote {OUT_PAIRS}")

    # Render report
    parts = ["# DART Disagreement Report — GPT-5.1 vs Gemini 3 Pro\n"]
    parts.append(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    parts.append(f"Sources: `dart_diagnoses.jsonl` (GPT-5.1, Run 1), `dart_diagnoses_gemini.jsonl` (Gemini 3 Pro, Run 2)")
    parts.append(f"Pairs classified: {n_pairs_classified} (each by both GPT-5.1 and Gemini 3 Pro)\n")

    parts.append("## Triage table\n")
    parts.append("Tier definitions:")
    parts.append("- **T1** — same diagnosis, all aligned-edit pairs same-direction → action-safe with normal review")
    parts.append("- **T2** — same diagnosis, at least one different-scope edit (no opposites) → light review needed")
    parts.append("- **T3** ⚠️ — same diagnosis, at least one OPPOSITE-direction edit pair → flag for spec-author review with both proposals quoted")
    parts.append("- **T4** — different diagnoses → ideally tiebreak via 3rd compiler\n")

    # Sort by tier (T3, T4 first)
    tier_order = {"T3": 0, "T4": 1, "T2": 2, "T1": 3}
    sorted_rows = sorted(out_rows, key=lambda r: (tier_order.get(r["tier"], 9), r["statement_id"]))
    parts.append("| tier | statement | diag (GPT) | diag (Gemini) | recommendation (GPT) | recommendation (Gemini) | rubric pairs | spec pairs | edit ensemble directions |")
    parts.append("|---|---|---|---|---|---|--:|--:|---|")
    for r in sorted_rows:
        directions = ", ".join(r["ensemble_directions"]) or "—"
        diag_marker = "✓" if r["diag_match"] else "✗"
        parts.append(f"| **{r['tier']}** | {r['statement_id']} | {r['diag_gpt']} {diag_marker} | {r['diag_gem']} | {r['rec_gpt']} | {r['rec_gem']} | {r['n_rubric_pairs']} | {r['n_spec_pairs']} | {directions} |")
    parts.append("")

    parts.append("## Tier counts\n")
    from collections import Counter
    tcount = Counter(r["tier"] for r in out_rows)
    for t in ("T1", "T2", "T3", "T4"):
        parts.append(f"- **{t}**: {tcount.get(t, 0)} statements")
    parts.append("")

    # Per-statement detail, T3 + T4 first
    for r in sorted_rows:
        sid = r["statement_id"]
        parts.append(f"\n---\n\n## {r['tier']} — `{sid}`\n")
        parts.append(f"**Diagnosis**: GPT={r['diag_gpt']} | Gemini={r['diag_gem']}  (match: {r['diag_match']})")
        parts.append(f"**Recommendation**: GPT={r['rec_gpt']} | Gemini={r['rec_gem']}\n")

        pairs = r["pairs"]
        if pairs["rubric_pairs"]:
            parts.append("### Aligned rubric edits\n")
            for p in pairs["rubric_pairs"]:
                a = p["anchor"]
                cls_g = p["classification_gpt"].get("direction", "?")
                cls_m = p["classification_gem"].get("direction", "?")
                ensemble = p["ensemble"]
                emoji = "🔥" if ensemble == "opposite_direction" else ("⚠️" if ensemble == "disputed" else "✓")
                parts.append(f"#### {emoji} Anchor {a} — ensemble verdict: **{ensemble}**\n")
                parts.append(f"  - Direction-classifier (GPT-5.1): **{cls_g}** — _{trunc(p['classification_gpt'].get('rationale',''), 250)}_")
                parts.append(f"  - Direction-classifier (Gemini 3 Pro): **{cls_m}** — _{trunc(p['classification_gem'].get('rationale',''), 250)}_\n")
                parts.append(f"  **GPT proposed**: {trunc(p['gpt_edit'].get('new_criterion'), 500)}\n")
                parts.append(f"  **Gemini proposed**: {trunc(p['gem_edit'].get('new_criterion'), 500)}\n")
        if pairs["spec_pairs"]:
            parts.append("### Aligned spec edits\n")
            for p in pairs["spec_pairs"]:
                cls_g = p["classification_gpt"].get("direction", "?")
                cls_m = p["classification_gem"].get("direction", "?")
                ensemble = p["ensemble"]
                emoji = "🔥" if ensemble == "opposite_direction" else ("⚠️" if ensemble == "disputed" else "✓")
                parts.append(f"#### {emoji} Spec phrase — ensemble verdict: **{ensemble}**\n")
                parts.append(f"  **Targeted phrase**: `{trunc(p['gpt_edit'].get('old_phrase'), 200)}`")
                parts.append(f"  - Direction-classifier (GPT-5.1): **{cls_g}** — _{trunc(p['classification_gpt'].get('rationale',''), 250)}_")
                parts.append(f"  - Direction-classifier (Gemini 3 Pro): **{cls_m}** — _{trunc(p['classification_gem'].get('rationale',''), 250)}_\n")
                parts.append(f"  **GPT proposed**: {trunc(p['gpt_edit'].get('new_phrase'), 500)}\n")
                parts.append(f"  **Gemini proposed**: {trunc(p['gem_edit'].get('new_phrase'), 500)}\n")
        # Unmatched
        unm = (len(pairs['rubric_unmatched_gpt']) + len(pairs['rubric_unmatched_gem'])
               + len(pairs['spec_unmatched_gpt']) + len(pairs['spec_unmatched_gem']))
        if unm:
            parts.append(f"### Unmatched edits (proposed by only one compiler)\n")
            for u in pairs["rubric_unmatched_gpt"]:
                parts.append(f"- 🟦 **GPT-only** rubric Anchor {u['anchor']}: {trunc(u['edit'].get('new_criterion'), 300)}")
            for u in pairs["rubric_unmatched_gem"]:
                parts.append(f"- 🟧 **Gemini-only** rubric Anchor {u['anchor']}: {trunc(u['edit'].get('new_criterion'), 300)}")
            for u in pairs["spec_unmatched_gpt"]:
                parts.append(f"- 🟦 **GPT-only** spec edit on `{trunc(u['edit'].get('old_phrase'), 100)}`: {trunc(u['edit'].get('new_phrase'), 300)}")
            for u in pairs["spec_unmatched_gem"]:
                parts.append(f"- 🟧 **Gemini-only** spec edit on `{trunc(u['edit'].get('old_phrase'), 100)}`: {trunc(u['edit'].get('new_phrase'), 300)}")
            parts.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
