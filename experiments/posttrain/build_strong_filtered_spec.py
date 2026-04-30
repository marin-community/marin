# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a 'filtered union' spec containing ONLY the R1 agent edits that
achieved STRONG propagation (per exp1).

This is a quality-filtered variant of `openai_model_spec_union_round1_edits.jsonl`
which contains all 29 R1 edits regardless of propagation.

Hypothesis: quality-filtered spec should give cleaner improvement (less
noise from edits that didn't actually move their target rubric).

Output: `experiments/posttrain/specs/openai_model_spec_strong_r1_only.jsonl`

Usage:
    uv run python experiments/posttrain/build_strong_filtered_spec.py
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
BASE_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
OUT_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec_strong_r1_only.jsonl"

JUDGES = ["flash", "gpt51", "pro", "glm51"]
CHANGE_THRESHOLD = 0.3


def sim_change(a, b):
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


def citation_fires(rationale_clauses, new_example):
    fields = [new_example.get(k, "") or "" for k in ("description", "user_query", "good_response", "bad_response")]
    for clause in rationale_clauses:
        c_lower = (clause or "").lower()
        if len(c_lower) < 20:
            continue
        for f_lower in (f.lower() for f in fields):
            if not f_lower:
                continue
            if c_lower in f_lower:
                return True
            for start in range(0, max(0, len(c_lower) - 40), 20):
                window = c_lower[start : start + 60]
                if window and len(window) >= 40 and window in f_lower:
                    return True
    return False


def baseline_path(judge):
    return STAGE3 / ("cross_tier_rubrics_v2.jsonl" if judge == "flash" else f"cross_tier_rubrics_v2_{judge}.jsonl")


def classify(edit, baseline, withedit):
    b_p = baseline["parsed"]
    w_p = withedit["parsed"]
    new_clauses = w_p.get("rationale", {}).get("spec_clauses_anchored_on", []) or []
    cited = citation_fires(new_clauses, edit["new_example"])
    bad_d = sim_change(
        (b_p.get("dominant_rubric") or {}).get("BAD", ""), (w_p.get("dominant_rubric") or {}).get("BAD", "")
    )
    alt_d = sim_change(
        (b_p.get("rationale") or {}).get("alternative_readings_rejected", ""),
        (w_p.get("rationale") or {}).get("alternative_readings_rejected", ""),
    )
    we_d = sim_change(
        (b_p.get("worked_example") or {}).get("spec_compliant", ""),
        (w_p.get("worked_example") or {}).get("spec_compliant", ""),
    )
    sig = sum(1 for x in [bad_d, alt_d, we_d] if x > CHANGE_THRESHOLD)
    if cited and sig >= 1:
        return "STRONG"
    if cited:
        return "WEAK"
    if sig >= 2:
        return "AMBIG"
    return "NONE"


def main():
    # Load all R1 agent edits, classify each, keep only STRONG ones
    strong_edits = []
    for judge in JUDGES:
        edits_dir = LM_JUDGE_EDITS / judge / "proposed_edits"
        edits = [json.loads(p.read_text()) for p in sorted(edits_dir.glob("*.json"))]
        baseline = {
            (r["pair_id"], r["tension_point_idx"]): r
            for r in (json.loads(line) for line in baseline_path(judge).open() if line.strip())
        }
        with_edits_path = STAGE3 / f"cross_tier_rubrics_v2_{judge}_with_self_edits.jsonl"
        with_edits = {
            (r["pair_id"], r["tension_point_idx"]): r
            for r in (json.loads(line) for line in with_edits_path.open() if line.strip())
        }
        for e in edits:
            tp = (e["test_pair"]["pair_id"], e["test_pair"]["tension_point_idx"])
            cls = classify(e, baseline.get(tp), with_edits.get(tp))
            if cls == "STRONG":
                e["_propagation_class"] = cls
                e["_source_judge"] = judge
                strong_edits.append(e)

    # Build filtered spec
    base = [json.loads(line) for line in BASE_SPEC.open() if line.strip()]
    by_id = {r["id"]: r for r in base}
    applied = 0
    skipped = 0
    seen_examples = set()  # dedup by (target_id, user_query)
    for e in strong_edits:
        target = e.get("target_statement_id")
        if target not in by_id:
            skipped += 1
            continue
        new_example = dict(e["new_example"])
        new_example["_origin"] = f"strong_r1/{e['_source_judge']}/{e['edit_id']}"
        # Dedup: don't add the same user_query twice to the same statement
        key = (target, new_example.get("user_query", "")[:100])
        if key in seen_examples:
            continue
        seen_examples.add(key)
        meta = by_id[target].setdefault("metadata", {})
        examples = meta.setdefault("examples", [])
        examples.append(new_example)
        applied += 1

    OUT_SPEC.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in base) + "\n")
    print(f"Found {len(strong_edits)} STRONG R1 edits across 4 judges (deduplicated to {applied} unique).")
    print(f"Skipped {skipped} edits (target not in spec).")
    print(f"Wrote {OUT_SPEC}")
    print("Compare to:")
    print("  - openai_model_spec_union_round1_edits.jsonl (all 29 R1 edits)")
    print("  - openai_model_spec.jsonl (no edits, baseline)")


if __name__ == "__main__":
    main()
