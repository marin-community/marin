"""Re-judge GPT-5.1 ONLY under phase_4 condition with the v2 rubrics.

Targeted validation experiment for the rubric-revision methodology:
- Use the v2 rubrics (from e8_rubrics_v2.jsonl) only on the 5 affected statements
- Re-judge ONLY with GPT-5.1 (which was the dominant outlier under v1)
- Re-judge ONLY under phase_4 (statement + rubric) — bare condition unchanged
- DO NOT call Gemini or Claude — reuse their existing v1 phase_4 judgments

Output: per_judgment_v2.jsonl
  Same schema as per_judgment_opposite.jsonl, with `rubric_version: "v2"` to
  distinguish from the v1 rows.

The downstream agreement-recalc script merges the new GPT v2 rows with the
existing Gemini + Claude v1 rows for the 3-judge ensemble metric.

NOTE: reasoning_effort="none" enforced via call_gpt_json (HARD project rule).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    SPEC_PATH,
    call_gpt_json,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
    write_jsonl,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
RUBRICS_V2_PATH = DIR / "e8_rubrics_v2.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
OUT_PATH = DIR / "per_judgment_v2.jsonl"

DEFAULT_HURT = [
    "refusal_style",
    "no_agenda",
    "comply_with_laws",
    "avoid_abuse",
    "sexual_content_involving_minors",
]

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]
GROK_LABEL = "grok-4-1-fast-non-reasoning-opposite"


def build_user_prompt(stmt, examples, rubric, user_q, response_text):
    """Phase_4 prompt: spec statement + examples + rubric + user query + response."""
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def main():
    target_sids = set(DEFAULT_HURT)
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics_v2 = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_V2_PATH) if "error" not in r}
    missing = target_sids - set(rubrics_v2.keys())
    if missing:
        raise SystemExit(f"v2 rubrics missing for: {missing}")

    # Build cells: existing 3 generators
    cells = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if sid not in target_sids: continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    # Plus Grok-opposite cells
    if OPPOSITE_RESPONSES.exists():
        for r in load_jsonl(OPPOSITE_RESPONSES):
            if "error" in r: continue
            if r.get("statement_id") not in target_sids: continue
            cells.append((r["statement_id"], r["scenario_idx"], r["generator"], r["user_query"], r["response"]))

    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    print(f"Re-judging GPT-5.1 phase_4 with v2 rubrics")
    print(f"  statements: {sorted(target_sids)}")
    print(f"  cells: {len(cells)}")
    print(f"  expected: 5 statements × (60 existing + 20 Grok) = {5 * 80} = 400 if Grok available\n")

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    log = RawAPILogger("e9_rejudge_gpt_v2")
    print(f"  raw log dir: {log.run_dir}\n")

    new_rows = []
    overall_t0 = time.time()
    n_done = 0

    def process_cell(cell):
        sid, scen, gen, user_q, resp_text = cell
        rubric_v2 = rubrics_v2[sid]
        examples = get_examples(spec[sid])
        user_text = build_user_prompt(spec[sid], examples, rubric_v2, user_q, resp_text)
        key = {"sid": sid, "scenario_idx": scen, "generator": gen,
               "condition": "phase_4", "rubric_version": "v2"}
        try:
            data = call_gpt_json(log, oai, role="judge_gpt_phase_4_v2", key=key,
                                 system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                                 user=user_text, max_tokens=1500)
            score = int(data["score"]) if data.get("score") is not None else None
            if score is not None and not 1 <= score <= 5:
                score = None
            return {
                "condition": "rubric_plus_spec",
                "judge": "gpt",
                "rubric_version": "v2",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
                "example_refs": data.get("example_refs") or [],
                "rubric_spec_tension": data.get("rubric_spec_tension"),
            }
        except Exception as exc:
            return {"condition": "rubric_plus_spec", "judge": "gpt",
                    "rubric_version": "v2", "statement_id": sid,
                    "scenario_idx": scen, "generator": gen,
                    "error": str(exc)[:200]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_cell, c): c for c in cells}
        for fut in as_completed(futs):
            r = fut.result()
            new_rows.append(r)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(cells):
                elapsed = time.time() - overall_t0
                n_err = sum(1 for x in new_rows if "error" in x)
                print(f"  progress: {n_done}/{len(cells)} ({100*n_done/len(cells):.0f}%, {elapsed:.0f}s, errors={n_err})")

    write_jsonl(sorted(new_rows, key=lambda r: (r["statement_id"], r["scenario_idx"], r["generator"])), OUT_PATH)
    n_ok = sum(1 for r in new_rows if r.get("score") is not None)
    print(f"\nwrote {OUT_PATH} ({n_ok}/{len(new_rows)} scored)")
    print(f"wall: {time.time() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
