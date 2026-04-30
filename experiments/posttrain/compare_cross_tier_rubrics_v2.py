# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diff report for the cross-tier rubric regen experiment.

Compares production rubrics (`cross_tier_rubrics_seed22_gemini_flash_repair1.jsonl`)
against v2 regen rubrics (`cross_tier_rubrics_v2.jsonl`) and emits a single
markdown report. For each pair:

- v2 RATIONALE block (the new info), with verbatim audit on spec_clauses_anchored_on
- Field-by-field similarity (difflib.SequenceMatcher.ratio) and side-by-side text
- Heuristic classification: low similarity + topic-targeted = topic_opinion_removed candidate
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
PROD_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_seed22_gemini_flash_repair1.jsonl"
V2_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_v2.jsonl"
OUT_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_v2_diff_report.md"

TOPIC_TARGETED_PAIRS = {
    ("assume_objective_pov__do_not_encourage_self_harm", 0),  # self-harm tp=0 (Gate 1 fail)
    ("assume_objective_pov__do_not_encourage_self_harm", 3),  # self-harm tp=3 (Gate 1 pass)
    ("avoid_hateful_content__present_perspectives", 3),  # dogwhistles tp=3 (Gate 1 fail)
    ("avoid_targeted_political_manipulation__be_kind", 0),  # political clause
    ("avoid_targeted_political_manipulation__do_not_lie", 0),  # political clause
    ("formatting__letter_and_spirit", 4),  # formatting clause
}

RUBRIC_FIELDS = [
    ("dominant_rubric", "GOOD"),
    ("dominant_rubric", "BAD"),
    ("dominant_rubric", "KEY_TENSION"),
    ("non_leakage_rubric", "GOOD"),
    ("non_leakage_rubric", "BAD"),
    ("non_leakage_rubric", "KEY_TENSION"),
]
WORKED_FIELDS = [
    ("worked_example", "spec_compliant"),
    ("worked_example", "subordinate_over_satisfaction"),
    ("worked_example", "pathological_refusal"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec_per_id() -> dict[str, str]:
    """Map statement_id → concatenated lower-case text (text + examples).

    Strict per-pair audit: clauses must match the dominant or subordinate
    statement of the rubric being audited. DO NOT use a full-spec
    concatenation — see `.agents/projects/executable_specifications.md`
    § "MAJOR LEARNING".
    """
    out = {}
    for line in SPEC_PATH.open():
        s = json.loads(line)
        chunks = [s.get("text", "")]
        for ex in (s.get("metadata") or {}).get("examples") or []:
            chunks.append(ex.get("description", ""))
            chunks.append(ex.get("user_query", ""))
            chunks.append(ex.get("good_response", ""))
            chunks.append(ex.get("bad_response", ""))
        out[s["id"]] = "\n".join(chunks).lower()
    return out


def clause_in_pair(clause: str, row: dict, spec_per_id: dict[str, str]) -> bool:
    dom_id = row.get("dominant_id")
    sub_id = row.get("subordinate_id")
    relevant = spec_per_id.get(dom_id, "") + "\n" + spec_per_id.get(sub_id, "")
    return (clause or "").lower().strip() in relevant.lower()


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def classify(rubric_field_sims: list[float], pair_id: str, tp: int) -> str:
    """Heuristic classification for the diff. Manual override is expected."""
    avg_sim = sum(rubric_field_sims) / max(len(rubric_field_sims), 1)
    is_topic = (pair_id, tp) in TOPIC_TARGETED_PAIRS
    if avg_sim > 0.65:
        return "neutral_re_application"
    if avg_sim < 0.35 and is_topic:
        return "topic_opinion_removed (likely)"
    if avg_sim < 0.35:
        return "interpretive_change (high diff, neutral pair)"
    if is_topic:
        return "topic_opinion_removed (moderate diff)"
    return "interpretive_change"


def fmt_field(label: str, prod_text: str, v2_text: str, similarity: float) -> str:
    return (
        f"\n#### {label} (similarity: {similarity:.2f})\n\n"
        f"<details><summary>production</summary>\n\n```\n{prod_text or '(empty)'}\n```\n\n</details>\n\n"
        f"<details><summary>v2</summary>\n\n```\n{v2_text or '(empty)'}\n```\n\n</details>\n"
    )


def render_pair(prod: dict[str, Any], v2: dict[str, Any], spec_per_id: dict[str, str]) -> str:
    pair_id = prod["pair_id"]
    tp = prod["tension_point_idx"]
    is_topic = (pair_id, tp) in TOPIC_TARGETED_PAIRS

    p_parsed = prod["parsed"]
    v_parsed = v2["parsed"]

    field_sims: list[float] = []
    field_blocks: list[str] = []
    for top, sub in RUBRIC_FIELDS + WORKED_FIELDS:
        p_text = (p_parsed.get(top) or {}).get(sub, "") or ""
        v_text = (v_parsed.get(top) or {}).get(sub, "") or ""
        s = sim(p_text, v_text)
        field_sims.append(s)
        field_blocks.append(fmt_field(f"{top}.{sub}", p_text, v_text, s))

    rubric_sims = field_sims[:6]
    avg_rubric_sim = sum(rubric_sims) / len(rubric_sims)
    classification = classify(rubric_sims, pair_id, tp)

    rationale = v_parsed.get("rationale", {})
    clauses = rationale.get("spec_clauses_anchored_on", [])
    verbatim_results = [(c, clause_in_pair(c, v2, spec_per_id)) for c in clauses]
    verbatim_pass = sum(1 for _, ok in verbatim_results if ok)
    verbatim_total = len(verbatim_results)

    out = []
    out.append(f"## {pair_id} tp={tp}\n")
    out.append(f"**Tension**: {prod.get('tension_name', '')}\n")
    out.append(f"**Dominant**: `{prod['dominant_id']}` | **Subordinate**: `{prod['subordinate_id']}`\n")
    out.append(
        f"**Topic-targeted**: {'YES' if is_topic else 'no'} | "
        f"**Avg rubric similarity**: {avg_rubric_sim:.2f} | "
        f"**Classification**: `{classification}`\n"
    )
    out.append("")
    out.append("### v2 RATIONALE")
    out.append("")
    out.append(
        f"**Verbatim audit**: {verbatim_pass}/{verbatim_total} spec_clauses_anchored_on are verbatim-matchable to spec text."
    )
    out.append("")
    out.append("**spec_clauses_anchored_on**:")
    for c, ok in verbatim_results:
        marker = "OK" if ok else "FAIL"
        truncated = c if len(c) <= 200 else c[:200] + "..."
        out.append(f"- [{marker}] {truncated}")
    out.append("")
    out.append("**interpretive_choices_made**:")
    out.append(f"> {rationale.get('interpretive_choices_made', '(missing)')}")
    out.append("")
    out.append("**alternative_readings_rejected**:")
    out.append(f"> {rationale.get('alternative_readings_rejected', '(missing)')}")
    out.append("")
    out.append("### Field-by-field diff")
    out.extend(field_blocks)
    out.append("\n---\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", type=Path, default=PROD_PATH)
    ap.add_argument("--v2", type=Path, default=V2_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    prod_rows = load_jsonl(args.prod)
    v2_rows = load_jsonl(args.v2)
    spec_per_id = load_spec_per_id()

    prod_by_key = {(r["pair_id"], r["tension_point_idx"]): r for r in prod_rows}
    v2_by_key = {(r["pair_id"], r["tension_point_idx"]): r for r in v2_rows}

    common_keys = sorted(set(prod_by_key) & set(v2_by_key))
    only_prod = sorted(set(prod_by_key) - set(v2_by_key))
    only_v2 = sorted(set(v2_by_key) - set(prod_by_key))

    body_parts: list[str] = []
    body_parts.append("# Cross-tier rubric regen diff report (v2 vs production)")
    body_parts.append("")
    body_parts.append(f"**Production source**: `{args.prod.relative_to(WORKTREE)}` ({len(prod_rows)} rows)")
    body_parts.append(f"**V2 source**: `{args.v2.relative_to(WORKTREE)}` ({len(v2_rows)} rows)")
    body_parts.append(
        f"**Common pairs**: {len(common_keys)}, **only-prod**: {len(only_prod)}, **only-v2**: {len(only_v2)}"
    )
    body_parts.append("")
    body_parts.append(
        "**v2 changes**: pass ALL spec examples (no `N_EXAMPLES_PER_STATEMENT=2` cap); strip topic-specific REQUIREMENTS at lines 86-89; add `rationale` field with verbatim spec quotes; otherwise unchanged from `write_gemini_cross_tier_seed_rubrics.py`. Same `gemini-3-flash-preview` model, `thinking_budget=0`, `temperature=0.2`."
    )
    body_parts.append("")

    body_parts.append("## Summary table")
    body_parts.append("")
    body_parts.append("| pair_id | tp | topic-targeted | avg sim | classification | verbatim audit |")
    body_parts.append("|---|---:|:---:|---:|---|:---:|")
    summary_rows = []
    for key in common_keys:
        prod = prod_by_key[key]
        v2 = v2_by_key[key]
        is_topic = key in TOPIC_TARGETED_PAIRS
        p_parsed = prod["parsed"]
        v_parsed = v2["parsed"]
        sims = []
        for top, sub in RUBRIC_FIELDS:
            p_text = (p_parsed.get(top) or {}).get(sub, "") or ""
            v_text = (v_parsed.get(top) or {}).get(sub, "") or ""
            sims.append(sim(p_text, v_text))
        avg = sum(sims) / len(sims)
        classification = classify(sims, key[0], key[1])
        clauses = v_parsed.get("rationale", {}).get("spec_clauses_anchored_on", [])
        vp = sum(1 for c in clauses if clause_in_pair(c, v2, spec_per_id))
        vt = len(clauses)
        body_parts.append(
            f"| `{key[0]}` | {key[1]} | {'YES' if is_topic else ''} | {avg:.2f} | {classification} | {vp}/{vt} |"
        )
        summary_rows.append((key, avg, classification, is_topic))
    body_parts.append("")

    body_parts.append("## Aggregate stats")
    body_parts.append("")
    avg_all = sum(s for _, s, _, _ in summary_rows) / len(summary_rows)
    avg_topic = [s for _, s, _, t in summary_rows if t]
    avg_neutral = [s for _, s, _, t in summary_rows if not t]
    body_parts.append(f"- Mean rubric similarity (all {len(summary_rows)} pairs): {avg_all:.3f}")
    if avg_topic:
        body_parts.append(f"- Mean similarity on topic-targeted ({len(avg_topic)}): {sum(avg_topic)/len(avg_topic):.3f}")
    if avg_neutral:
        body_parts.append(
            f"- Mean similarity on topic-neutral ({len(avg_neutral)}): {sum(avg_neutral)/len(avg_neutral):.3f}"
        )
    classes: dict[str, int] = {}
    for _, _, c, _ in summary_rows:
        classes[c] = classes.get(c, 0) + 1
    body_parts.append(f"- Classifications: {classes}")
    body_parts.append("")
    body_parts.append(
        "Lower similarity = bigger diff. Topic-targeted should be lower than neutral if the prompt patches were load-bearing."
    )
    body_parts.append("")
    body_parts.append("---")
    body_parts.append("")

    for key in common_keys:
        body_parts.append(render_pair(prod_by_key[key], v2_by_key[key], spec_per_id))

    args.out.write_text("\n".join(body_parts))
    print(f"Wrote diff report to {args.out}")
    print(f"  - common pairs: {len(common_keys)}")
    print(f"  - mean similarity: {avg_all:.3f}")
    if avg_topic and avg_neutral:
        print(f"  - topic-targeted mean: {sum(avg_topic)/len(avg_topic):.3f}")
        print(f"  - topic-neutral mean:  {sum(avg_neutral)/len(avg_neutral):.3f}")
    print(f"  - classifications: {classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
