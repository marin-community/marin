"""Extract GPT + Gemini reasoning text from raw API dumps for the original
3-generator × 46-statement × bare/phase_4 judging runs.

The pre-existing per_judgment.jsonl drops the reasoning text and keeps only stats.
We need the actual text for qualitative subagent analysis.

Reads:
  - results/raw/e8_paired_indirection/2026-05-03T21-34-28/judge_variant_a/   (gpt bare)
  - results/raw/e8_phase2_gemini/2026-05-04T00-20-11/judge_variant_a_gemini/ (gemini bare)
  - results/raw/e8_phase4_gpt/2026-05-06T01-01-59/judge_rubric_plus_spec_gpt/
  - results/raw/e8_phase4_gemini/2026-05-06T01-02-03/judge_rubric_plus_spec_gemini/

Writes:
  - experiments/posttrain/disagreement_primitive/per_judgment_reasoning.jsonl
    with rows {sid, cond_internal, scen, generator, judge, score, reasoning, spec_quotes}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_PATH = DIR / "per_judgment_reasoning.jsonl"

DUMP_SOURCES = [
    # (path to dir, judge, condition_internal)
    (Path("results/raw/e8_paired_indirection/2026-05-03T21-34-28/judge_variant_a"), "gpt", "variant_A"),
    (Path("results/raw/e8_phase2_gemini/2026-05-04T00-20-11/judge_variant_a_gemini"), "gemini", "variant_A"),
    (Path("results/raw/e8_phase4_gpt/2026-05-06T01-01-59/judge_rubric_plus_spec_gpt"), "gpt", "rubric_plus_spec"),
    (Path("results/raw/e8_phase4_gemini/2026-05-06T01-02-03/judge_rubric_plus_spec_gemini"), "gemini", "rubric_plus_spec"),
]


def parse_text_to_json(text: str) -> dict | None:
    """Best-effort parse the judge's JSON output. Strips code fences."""
    if not text: return None
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            first = s[:nl].strip("`").strip()
            if first == "" or first.lower() == "json":
                s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def extract_from_dump(path: Path) -> dict | None:
    """Return (sid, scen, gen, score, reasoning, spec_quotes) or None on failure."""
    try:
        b = json.load(path.open())
    except Exception:
        return None
    key = b.get("key") or {}
    sid = key.get("statement_id")
    scen = key.get("scenario_idx")
    gen = key.get("generator")
    if sid is None or scen is None or gen is None:
        return None
    resp = b.get("response") or {}
    text = ""
    # OpenAI shape
    if "choices" in resp:
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
    # Gemini shape
    elif "text" in resp:
        text = resp.get("text") or ""
    elif "candidates" in resp:
        cand = (resp.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
    parsed = parse_text_to_json(text)
    if not isinstance(parsed, dict):
        return {"sid": sid, "scen": scen, "gen": gen, "score": None, "reasoning": None, "spec_quotes": None, "raw_text": text[:600]}
    return {"sid": sid, "scen": scen, "gen": gen,
            "score": parsed.get("score"), "reasoning": parsed.get("reasoning"),
            "spec_quotes": parsed.get("spec_quotes")}


def main() -> int:
    rows = []
    for src_dir, judge, cond in DUMP_SOURCES:
        if not src_dir.exists():
            print(f"  WARN: missing {src_dir}")
            continue
        files = list(src_dir.glob("*.json"))
        print(f"  scanning {src_dir} ({judge}, {cond}): {len(files)} files")
        n_ok = n_fail = 0
        for f in files:
            ext = extract_from_dump(f)
            if ext is None:
                n_fail += 1
                continue
            try:
                score = int(ext["score"]) if ext["score"] is not None else None
            except (TypeError, ValueError):
                score = None
            row = {
                "judge": judge,
                "condition": cond,
                "statement_id": ext["sid"],
                "scenario_idx": ext["scen"],
                "generator": ext["gen"],
                "score": score,
                "reasoning": ext.get("reasoning"),
                "spec_quotes": ext.get("spec_quotes"),
            }
            rows.append(row)
            if score is not None:
                n_ok += 1
            else:
                n_fail += 1
        print(f"    {n_ok} ok, {n_fail} fail")

    print(f"\nTotal rows: {len(rows)}")
    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
