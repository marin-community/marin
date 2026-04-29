# ruff: noqa: B007, E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""4-model matrix comparison report for cross-tier rubric writers.

Reads the 4 v2 outputs (Flash baseline + GPT-5.1 + Pro + GLM-5.1) and emits
a single markdown report comparing them on:

- Schema validity
- Spec-groundedness (verbatim audit, **case-insensitive** — fixes v1 bug)
- Rationale richness (clause counts, char lengths)
- Per-rubric 4-column side-by-side
- Cost summary

Usage:
    uv run python experiments/posttrain/compare_cross_tier_rubrics_4model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
OUT_PATH = STAGE3 / "cross_tier_rubrics_v2_4model_matrix_report.md"

MODEL_FILES = [
    ("flash", STAGE3 / "cross_tier_rubrics_v2.jsonl", "gemini-3-flash-preview", "thinking_budget=0"),
    ("gpt51", STAGE3 / "cross_tier_rubrics_v2_gpt51.jsonl", "gpt-5.1", "reasoning_effort=none"),
    ("pro", STAGE3 / "cross_tier_rubrics_v2_pro.jsonl", "gemini-3-pro-preview", "thinking_budget=128"),
    ("glm51", STAGE3 / "cross_tier_rubrics_v2_glm51.jsonl", "zai-org/GLM-5.1", "no reasoning toggle"),
]

TOPIC_TARGETED_PAIRS = {
    ("assume_objective_pov__do_not_encourage_self_harm", 0),
    ("assume_objective_pov__do_not_encourage_self_harm", 3),
    ("avoid_hateful_content__present_perspectives", 3),
    ("avoid_targeted_political_manipulation__be_kind", 0),
    ("avoid_targeted_political_manipulation__do_not_lie", 0),
    ("formatting__letter_and_spirit", 4),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec_per_id() -> dict[str, str]:
    """Map statement_id → concatenated lower-case text (text + all examples).

    STRICT per-pair audit: a clause is valid only if it appears in the
    dominant or subordinate statement of the rubric being audited.

    DO NOT reintroduce a full-spec concatenation. The earlier loose
    full-spec match was producing false passes; see
    `.agents/projects/executable_specifications.md` § "MAJOR LEARNING".
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


def verbatim_match_ci(quote: str, row: dict, spec_per_id: dict[str, str]) -> bool:
    """Strict per-pair: clause must appear in dominant OR subordinate statement only."""
    dom_id = row.get("dominant_id")
    sub_id = row.get("subordinate_id")
    relevant = spec_per_id.get(dom_id, "") + "\n" + spec_per_id.get(sub_id, "")
    return (quote or "").lower().strip() in relevant.lower()


def extract_rubric_metrics(rows: list[dict[str, Any]], spec_per_id: dict[str, str]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    schema_ok = sum(1 for r in rows if r["diag"].get("schema_ok"))
    parse_ok = sum(1 for r in rows if r["diag"].get("parse_ok", True))
    clauses_per = [len(r["parsed"].get("rationale", {}).get("spec_clauses_anchored_on", [])) for r in rows]
    interp_chars = [len(r["parsed"].get("rationale", {}).get("interpretive_choices_made", "") or "") for r in rows]
    alt_chars = [len(r["parsed"].get("rationale", {}).get("alternative_readings_rejected", "") or "") for r in rows]

    total_clauses = 0
    verbatim_pass = 0
    for r in rows:
        clauses = r["parsed"].get("rationale", {}).get("spec_clauses_anchored_on", [])
        for c in clauses:
            total_clauses += 1
            if verbatim_match_ci(c, r, spec_per_id):
                verbatim_pass += 1

    return {
        "n": len(rows),
        "schema_ok": f"{schema_ok}/{len(rows)}",
        "parse_ok": f"{parse_ok}/{len(rows)}",
        "verbatim_audit_strict": f"{verbatim_pass}/{total_clauses} ({100*verbatim_pass/max(total_clauses,1):.0f}%)",
        "avg_clauses": f"{sum(clauses_per)/len(clauses_per):.1f}",
        "min_clauses": min(clauses_per) if clauses_per else 0,
        "max_clauses": max(clauses_per) if clauses_per else 0,
        "avg_interpretive_chars": f"{sum(interp_chars)/len(interp_chars):.0f}",
        "avg_alternatives_chars": f"{sum(alt_chars)/len(alt_chars):.0f}",
    }


def field_text(row: dict[str, Any], top: str, sub: str) -> str:
    return ((row.get("parsed") or {}).get(top) or {}).get(sub, "") or ""


def render_per_pair(model_rows: dict[str, dict[tuple, dict]], pair: tuple, spec_per_id: dict[str, str]) -> str:
    parts = []
    pair_id, tp = pair
    is_topic = pair in TOPIC_TARGETED_PAIRS
    parts.append(f"## {pair_id} tp={tp}")
    parts.append("")
    parts.append(f"**Topic-targeted**: {'YES' if is_topic else 'no'}")
    parts.append("")

    # tension name from any column that has the row
    for label, _, _, _ in MODEL_FILES:
        if pair in model_rows.get(label, {}):
            r = model_rows[label][pair]
            parts.append(f"**Tension**: {r.get('tension_name', '')}")
            parts.append(f"**Dominant**: `{r['dominant_id']}` | **Subordinate**: `{r['subordinate_id']}`")
            break
    parts.append("")

    # rationale block per model
    parts.append("### Rationale per model")
    parts.append("")
    for label, _, model_name, knobs in MODEL_FILES:
        if pair not in model_rows.get(label, {}):
            parts.append(f"#### {label} (`{model_name}`, {knobs})")
            parts.append("> *(no row)*")
            parts.append("")
            continue
        r = model_rows[label][pair]
        rationale = r["parsed"].get("rationale", {})
        clauses = rationale.get("spec_clauses_anchored_on", [])
        verbatim_count = sum(1 for c in clauses if verbatim_match_ci(c, r, spec_per_id))
        parts.append(f"#### {label} (`{model_name}`, {knobs}) — verbatim audit: {verbatim_count}/{len(clauses)}")
        parts.append("")
        parts.append("**spec_clauses_anchored_on**:")
        for c in clauses:
            ok = verbatim_match_ci(c, r, spec_per_id)
            marker = "OK" if ok else "MISS"
            t = c if len(c) <= 200 else c[:200] + "..."
            parts.append(f"- [{marker}] {t}")
        parts.append("")
        parts.append("**interpretive_choices_made**:")
        parts.append(f"> {rationale.get('interpretive_choices_made', '(missing)')}")
        parts.append("")
        parts.append("**alternative_readings_rejected**:")
        parts.append(f"> {rationale.get('alternative_readings_rejected', '(missing)')}")
        parts.append("")

    # rubric criteria per model
    parts.append("### dominant_rubric.GOOD per model")
    parts.append("")
    for label, _, model_name, _ in MODEL_FILES:
        if pair not in model_rows.get(label, {}):
            continue
        text = field_text(model_rows[label][pair], "dominant_rubric", "GOOD")
        parts.append(f"<details><summary>{label} (`{model_name}`)</summary>\n\n```\n{text}\n```\n\n</details>")
    parts.append("")

    parts.append("### dominant_rubric.BAD per model")
    parts.append("")
    for label, _, model_name, _ in MODEL_FILES:
        if pair not in model_rows.get(label, {}):
            continue
        text = field_text(model_rows[label][pair], "dominant_rubric", "BAD")
        parts.append(f"<details><summary>{label} (`{model_name}`)</summary>\n\n```\n{text}\n```\n\n</details>")
    parts.append("")

    parts.append("### non_leakage_rubric.BAD per model")
    parts.append("")
    for label, _, model_name, _ in MODEL_FILES:
        if pair not in model_rows.get(label, {}):
            continue
        text = field_text(model_rows[label][pair], "non_leakage_rubric", "BAD")
        parts.append(f"<details><summary>{label} (`{model_name}`)</summary>\n\n```\n{text}\n```\n\n</details>")
    parts.append("")

    parts.append("### worked_example.spec_compliant per model")
    parts.append("")
    for label, _, model_name, _ in MODEL_FILES:
        if pair not in model_rows.get(label, {}):
            continue
        text = field_text(model_rows[label][pair], "worked_example", "spec_compliant")
        parts.append(f"<details><summary>{label} (`{model_name}`)</summary>\n\n```\n{text}\n```\n\n</details>")
    parts.append("")
    parts.append("---")
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    spec_per_id = load_spec_per_id()

    by_model: dict[str, list[dict]] = {}
    indexed: dict[str, dict[tuple, dict]] = {}
    for label, path, _, _ in MODEL_FILES:
        rows = load_jsonl(path)
        by_model[label] = rows
        indexed[label] = {(r["pair_id"], r["tension_point_idx"]): r for r in rows}

    all_pairs: set[tuple] = set()
    for d in indexed.values():
        all_pairs.update(d.keys())
    common = sorted(
        p for p in all_pairs if all(p in indexed[label] for label, _, _, _ in MODEL_FILES if by_model[label])
    )
    all_sorted = sorted(all_pairs)

    out: list[str] = []
    out.append("# Cross-tier rubric writer — 4-model matrix comparison")
    out.append("")
    out.append(
        "Same v2 system prompt across all four (rationale field, all spec examples, no topic-specific REQUIREMENTS). Same 22 cross-tier pairs."
    )
    out.append("")

    out.append("## Model summary")
    out.append("")
    out.append("| label | model | settings | rows |")
    out.append("|---|---|---|---:|")
    for label, _, model_name, knobs in MODEL_FILES:
        out.append(f"| `{label}` | `{model_name}` | {knobs} | {len(by_model[label])} |")
    out.append("")

    out.append("## Aggregate metrics")
    out.append("")
    out.append("| metric | flash | gpt51 | pro | glm51 |")
    out.append("|---|---|---|---|---|")
    metrics = {label: extract_rubric_metrics(by_model[label], spec_per_id) for label, _, _, _ in MODEL_FILES}
    for key in [
        "n",
        "schema_ok",
        "parse_ok",
        "verbatim_audit",
        "avg_clauses",
        "avg_interpretive_chars",
        "avg_alternatives_chars",
    ]:
        row = f"| {key} |"
        for label, _, _, _ in MODEL_FILES:
            row += f" {metrics[label].get(key, '-')} |"
        out.append(row)
    out.append("")

    out.append("## Cost (rough)")
    out.append("")
    out.append("| model | prompt tokens | output tokens | est cost |")
    out.append("|---|---:|---:|---:|")
    for label, _, model_name, _ in MODEL_FILES:
        rows = by_model[label]
        if not rows:
            out.append(f"| {label} | - | - | - |")
            continue
        if label == "gpt51":
            pin = sum(r["diag"].get("prompt_tokens") or 0 for r in rows)
            pout = sum(r["diag"].get("completion_tokens") or 0 for r in rows)
            cost = pin * 1.25 / 1e6 + pout * 10 / 1e6
        elif label in ("flash", "pro"):
            pin = sum((r["diag"].get("usage") or {}).get("prompt_token_count", 0) for r in rows)
            pout = sum((r["diag"].get("usage") or {}).get("candidates_token_count", 0) for r in rows)
            rate_in = 0.30 if label == "flash" else 1.25
            rate_out = 2.50 if label == "flash" else 10.00
            cost = pin * rate_in / 1e6 + pout * rate_out / 1e6
        elif label == "glm51":
            pin = sum(r["diag"].get("prompt_tokens") or 0 for r in rows)
            pout = sum(r["diag"].get("completion_tokens") or 0 for r in rows)
            cost = pin * 0.20 / 1e6 + pout * 0.40 / 1e6
        else:
            pin = pout = 0
            cost = 0
        out.append(f"| {label} | {pin:,} | {pout:,} | ${cost:.3f} |")
    out.append("")

    out.append(f"## Per-pair matrix ({len(all_sorted)} pairs, {len(common)} present in all 4 models)")
    out.append("")
    out.append(
        "Each pair section shows: rationale per model (with verbatim audit), then dominant_rubric.GOOD / BAD, non_leakage_rubric.BAD, worked_example.spec_compliant. Open the `<details>` blocks to read each model's text. Topic-targeted pairs (the 6 pre-identified ones) are flagged at the top."
    )
    out.append("")
    out.append("---")
    out.append("")
    for pair in all_sorted:
        out.append(render_per_pair(indexed, pair, spec_per_id))

    args.out.write_text("\n".join(out))
    print(f"Wrote {args.out}")
    print(f"  models present: {[lbl for lbl in by_model if by_model[lbl]]}")
    print(f"  pairs total: {len(all_sorted)}, in all 4 models: {len(common)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
