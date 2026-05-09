"""DART Step 3 — bidirectional compiler diagnostic via Gemini 3 Pro (lowest thinking).

Same prompt and inputs as e9_dart_compiler.py (GPT-5.1 variant), but uses
Gemini 3 Pro with thinking_budget=0. This tests compiler-stability of DART:
do the diagnoses agree across LM compilers, or are they LM-specific?

Outputs:
  - experiments/posttrain/disagreement_primitive/dart_diagnoses_gemini.jsonl
  - .agents/logbooks/dart_run_002_diagnoses.md

Pairs with `e9_dart_compiler.py`. Compare via `e9_dart_compiler_compare.py`
(if/when written).
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

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    SPEC_PATH,
    get_examples,
    load_jsonl,
    write_jsonl,
)
from e8_phase2_cross_model import _GEMINI_SAFETY_BLOCK_NONE
from e9_dart_compiler import (
    COMPILER_SYSTEM,
    DEFAULT_BUCKET_D,
    REPORT_OUT as REPORT_GPT_OUT,
    build_user_prompt,
    rank_bare_poison,
    trunc,
    validate_diagnosis,
)
from e9_build_qualitative_inputs import load_judgments, load_response_index
from e9_rubric_poison_rank import rank_rubric_poison
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
RUBRICS_V1_PATH = DIR / "e8_rubrics.jsonl"
DIAGNOSES_OUT = DIR / "dart_diagnoses_gemini.jsonl"
REPORT_OUT = Path(".agents/logbooks/dart_run_002_diagnoses.md")

GEMINI_MODEL = "gemini-3-pro-preview"


def parse_json_strict(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            first = s[:nl].strip("`").strip()
            if first == "" or first.lower() == "json":
                s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    return json.loads(s)


def call_gemini_json(log: RawAPILogger, gem: genai.Client, role: str, key: dict[str, Any],
                     system: str, user: str, max_tokens: int = 8000,
                     thinking_budget: int = 0) -> dict[str, Any]:
    """Gemini 3 Pro JSON-mode call with explicit thinking budget."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        response_mime_type="application/json",
        safety_settings=_GEMINI_SAFETY_BLOCK_NONE,
    )
    raw = log.call(
        role=role, key=key,
        fn=lambda: gem.models.generate_content(
            model=GEMINI_MODEL, contents=user, config=config,
        ),
    )
    return parse_json_strict(raw.text or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statement-ids", nargs="*", default=DEFAULT_BUCKET_D)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="Gemini thinking budget tokens (0 = disabled if model allows; else minimum)")
    args = ap.parse_args()

    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}
    by_cell = load_judgments()
    response_idx = load_response_index()

    gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    log = RawAPILogger("e9_dart_compiler_gemini")
    print(f"DART Step 3 (Gemini variant) — bidirectional compiler diagnostic")
    print(f"  model: {GEMINI_MODEL}, thinking_budget={args.thinking_budget}")
    print(f"  statements: {len(args.statement_ids)}")
    print(f"  raw log dir: {log.run_dir}\n")

    out_rows = []
    for sid in args.statement_ids:
        if sid not in spec or sid not in rubrics:
            print(f"  SKIP missing spec or rubric: {sid}")
            continue
        bare_rows = rank_bare_poison(by_cell, sid)
        rub_rows = rank_rubric_poison(by_cell, sid)
        if not bare_rows or not rub_rows:
            print(f"  SKIP no full-3-judge cells: {sid}")
            continue
        bare_pwv_total = sum(r["bare_pwv"] for r in bare_rows)
        rub_pwv_total = sum(r["rubric_pwv"] for r in bare_rows if r["rubric_pwv"] is not None)
        v1_anchors = (rubrics.get(sid, {}) or {}).get("anchors", {})
        examples = get_examples(spec[sid])

        user = build_user_prompt(sid, spec[sid], examples, v1_anchors,
                                 bare_rows[:args.top_k], rub_rows[:args.top_k],
                                 response_idx, bare_pwv_total, rub_pwv_total)

        print(f"=== {sid} ===")
        print(f"  Σ bare_pwv: {bare_pwv_total}, Σ rubric_pwv: {rub_pwv_total}, Δ: {rub_pwv_total-bare_pwv_total:+d}")
        try:
            data = call_gemini_json(log, gem, role="dart_compiler_gemini",
                                    key={"statement_id": sid},
                                    system=COMPILER_SYSTEM, user=user,
                                    max_tokens=8000, thinking_budget=args.thinking_budget)
        except Exception as e:
            print(f"  ERROR: {e}")
            out_rows.append({"statement_id": sid, "error": str(e)[:300]})
            continue

        ok, problems = validate_diagnosis(data, spec[sid]["text"], v1_anchors)
        tag = "ok " if ok else f"WARN({len(problems)})"
        rec = data.get("recommendation", "?")
        diag = data.get("diagnosis", "?")
        n_re = len(data.get("rubric_edits") or [])
        n_se = len(data.get("spec_edits_for_author_review") or [])
        print(f"  [{tag}] diagnosis={diag}, rec={rec}, rubric_edits={n_re}, spec_edits={n_se}")
        for p in problems[:3]:
            print(f"    - {p}")

        out_rows.append({
            "statement_id": sid,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "compiler": GEMINI_MODEL,
            "thinking_budget": args.thinking_budget,
            "bare_pwv_total": bare_pwv_total,
            "rubric_pwv_total": rub_pwv_total,
            "diagnosis": data.get("diagnosis"),
            "evidence_summary": data.get("evidence_summary"),
            "rubric_edits": data.get("rubric_edits") or [],
            "spec_edits_for_author_review": data.get("spec_edits_for_author_review") or [],
            "recommendation": data.get("recommendation"),
            "validation_problems": problems,
        })

    write_jsonl(out_rows, DIAGNOSES_OUT)
    print(f"\nwrote {DIAGNOSES_OUT}")

    # Render markdown report
    parts = ["# DART Run 002 — Gemini 3 Pro compiler diagnoses\n"]
    parts.append(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    parts.append(f"Compiler: **{GEMINI_MODEL}** (thinking_budget={args.thinking_budget})")
    parts.append(f"Statements: {len(out_rows)} Bucket D statements at T₁=0.5\n")
    parts.append(f"Pairs with Run 001 (GPT-5.1 compiler): see `.agents/logbooks/dart_run_001_diagnoses.md`\n")

    parts.append("## Summary table\n")
    parts.append("| statement | Σ bare_pwv | Σ rubric_pwv | diagnosis | recommendation | n rubric edits | n spec edits |")
    parts.append("|---|--:|--:|---|---|--:|--:|")
    for r in out_rows:
        if "error" in r:
            parts.append(f"| {r['statement_id']} | — | — | ERROR | {r['error'][:60]} | — | — |")
            continue
        parts.append(f"| {r['statement_id']} | {r['bare_pwv_total']} | {r['rubric_pwv_total']} | {r['diagnosis']} | {r['recommendation']} | {len(r['rubric_edits'])} | {len(r['spec_edits_for_author_review'])} |")
    parts.append("")

    for r in out_rows:
        if "error" in r:
            continue
        parts.append(f"\n---\n\n## `{r['statement_id']}`\n")
        parts.append(f"**Diagnosis**: `{r['diagnosis']}` | **Recommendation**: `{r['recommendation']}`\n")
        parts.append(f"**Evidence summary**: {r['evidence_summary']}\n")
        if r["rubric_edits"]:
            parts.append(f"### Proposed rubric edits ({len(r['rubric_edits'])})\n")
            for e in r["rubric_edits"]:
                parts.append(f"#### Anchor {e.get('anchor')}  (confidence: {e.get('confidence')})")
                parts.append(f"**Old criterion**: {trunc(e.get('old_criterion'), 600)}")
                parts.append(f"**New criterion**: {trunc(e.get('new_criterion'), 800)}")
                parts.append(f"**Rationale**: {e.get('rationale')}\n")
        if r["spec_edits_for_author_review"]:
            parts.append(f"### Proposed spec edits — for spec-author review ({len(r['spec_edits_for_author_review'])})\n")
            parts.append("⚠️ These are PROPOSALS, not changes. Spec edits never auto-deploy.\n")
            for i, e in enumerate(r["spec_edits_for_author_review"]):
                parts.append(f"#### Spec edit {i+1}  (confidence: {e.get('confidence')})")
                parts.append(f"**Old phrase**: `{e.get('old_phrase')}`")
                parts.append(f"**Proposed**: `{e.get('new_phrase')}`")
                parts.append(f"**Rationale**: {e.get('rationale')}\n")
        if r["validation_problems"]:
            parts.append(f"### Validation problems\n")
            for p in r["validation_problems"]:
                parts.append(f"- {p}")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(parts))
    print(f"wrote {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
