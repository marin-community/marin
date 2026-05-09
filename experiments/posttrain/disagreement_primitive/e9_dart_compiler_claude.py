"""DART Step 3 — bidirectional compiler diagnostic via Claude Sonnet 4.6.

Same prompt and inputs as e9_dart_compiler.py (GPT-5.1) and
e9_dart_compiler_gemini.py (Gemini 3 Pro). Uses Claude with thinking disabled
and tool-use forcing for strict JSON schema (per the project's earlier
finding that tool-use eliminates JSON parse failures).

This is the 3rd compiler in the DART ensemble — adds tiebreak capability on
the 6 statements where GPT and Gemini gave different diagnoses.

Outputs:
  - experiments/posttrain/disagreement_primitive/dart_diagnoses_claude.jsonl
  - .agents/logbooks/dart_run_003_diagnoses.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    SPEC_PATH,
    get_examples,
    load_jsonl,
    write_jsonl,
)
from e9_claude_judge import ANTHROPIC_MODEL, ANTHROPIC_MODEL_CREATED_AT
from e9_dart_compiler import (
    COMPILER_SYSTEM,
    DEFAULT_BUCKET_D,
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
DIAGNOSES_OUT = DIR / "dart_diagnoses_claude.jsonl"
REPORT_OUT = Path(".agents/logbooks/dart_run_003_diagnoses.md")


# Anthropic tool-use schema mirroring the COMPILER_SYSTEM JSON output spec
DART_COMPILER_TOOL = {
    "name": "submit_dart_diagnosis",
    "description": "Submit a single DART compiler diagnostic output. Call exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "enum": ["rubric_drift", "spec_ambiguity", "both", "irreducible"],
            },
            "evidence_summary": {"type": "string"},
            "rubric_edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "anchor": {"type": "string"},
                        "old_criterion": {"type": "string"},
                        "new_criterion": {"type": "string"},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["anchor", "old_criterion", "new_criterion", "rationale", "confidence"],
                },
            },
            "spec_edits_for_author_review": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_phrase": {"type": "string"},
                        "new_phrase": {"type": "string"},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["old_phrase", "new_phrase", "rationale", "confidence"],
                },
            },
            "recommendation": {
                "type": "string",
                "enum": ["adopt_rubric_edit", "drop_rubric", "escalate_spec", "both", "irreducible"],
            },
        },
        "required": ["diagnosis", "evidence_summary", "rubric_edits",
                     "spec_edits_for_author_review", "recommendation"],
    },
}


def call_claude_compiler(api_key: str, system: str, user: str,
                         max_tokens: int = 8000) -> dict[str, Any]:
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [DART_COMPILER_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_dart_diagnosis"},
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body, timeout=180.0,
    )
    if resp.status_code != 200:
        snippet = resp.text[:600].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"HTTP {resp.status_code}: {snippet}")
    return resp.json()


def extract_tool_args(api_resp: dict[str, Any]) -> dict[str, Any]:
    blocks = api_resp.get("content") or []
    tu = next((b for b in blocks if b.get("type") == "tool_use"), None)
    if tu is None:
        types_seen = [b.get("type") for b in blocks]
        raise ValueError(f"no tool_use block; got types={types_seen}")
    args = tu.get("input")
    if not isinstance(args, dict):
        raise ValueError("tool_use.input not a dict")
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statement-ids", nargs="*", default=DEFAULT_BUCKET_D)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set; run: source .env2")

    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}
    by_cell = load_judgments()
    response_idx = load_response_index()

    log = RawAPILogger("e9_dart_compiler_claude")
    print(f"DART Step 3 (Claude variant) — bidirectional compiler diagnostic")
    print(f"  model: {ANTHROPIC_MODEL}, thinking disabled, tool-use forced")
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
            key = {"statement_id": sid, "model": ANTHROPIC_MODEL,
                   "model_created_at": ANTHROPIC_MODEL_CREATED_AT}
            api_resp = log.call(role="dart_compiler_claude", key=key,
                                fn=lambda: call_claude_compiler(api_key, COMPILER_SYSTEM, user))
            data = extract_tool_args(api_resp)
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
            "compiler": ANTHROPIC_MODEL,
            "compiler_created_at": ANTHROPIC_MODEL_CREATED_AT,
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
    parts = ["# DART Run 003 — Claude Sonnet 4.6 compiler diagnoses\n"]
    parts.append(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    parts.append(f"Compiler: **{ANTHROPIC_MODEL}** (thinking disabled, tool-use forced)")
    parts.append(f"Statements: {len(out_rows)} Bucket D statements at T₁=0.5\n")
    parts.append("Pairs with Run 001 (GPT-5.1) and Run 002 (Gemini 3 Pro) — see those reports for the matching diagnoses.\n")

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
