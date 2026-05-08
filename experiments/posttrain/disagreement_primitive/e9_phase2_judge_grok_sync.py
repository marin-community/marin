"""Phase 2 — GPT-5.1 + Gemini-3-flash sync judging on the new Grok-opposite cells.

Skips Claude (that's done via Anthropic Batch API in Phase 3). Skips cells
already judged in `per_judgment_opposite.jsonl`.

Reads:
  - e9_opposite_mode_responses.jsonl (full Grok pool)
  - e8_rubrics.jsonl (1-5 rubrics, all 46 statements)
  - per_judgment_opposite.jsonl (existing 8-statement judgments to skip)

Writes:
  - per_judgment_opposite.jsonl (appends gpt + gemini rows for new 38 statements)

Note: reasoning_effort="none" enforced via call_gpt_json (HARD project rule).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    JUDGE_A_SYSTEM,
    SPEC_PATH,
    call_gpt_json,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
    write_jsonl,
)
from e8_phase2_cross_model import call_gemini_json
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
RUBRICS = DIR / "e8_rubrics.jsonl"
PER_JUDGMENT_OUT = DIR / "per_judgment_opposite.jsonl"


def build_user_prompt(stmt, examples, rubric, user_q, response_text, condition):
    if condition == "bare":
        return (f"STATEMENT TEXT:\n{stmt['text']}\n\n"
                f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
                f"USER QUERY: {user_q}\n\n"
                f"ASSISTANT RESPONSE: {response_text}\n\n"
                "Score per the schema.")
    return (f"SPEC STATEMENT:\n{stmt['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"RUBRIC:\n{render_anchors(rubric)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response_text}\n\n"
            "Score per the schema.")


def main() -> int:
    spec_by_id = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS) if "error" not in r}

    # All Grok-opposite cells
    opposite_rows = [r for r in load_jsonl(OPPOSITE_RESPONSES) if "error" not in r]

    # Find which (sid, scen, judge, condition) combos are already in per_judgment_opposite.jsonl
    already: set[tuple[str, int, str, str]] = set()
    if PER_JUDGMENT_OUT.exists():
        for r in load_jsonl(PER_JUDGMENT_OUT):
            if r.get("score") is None:
                continue
            cond = r.get("condition")
            if cond not in {"variant_A", "rubric_plus_spec"}:
                continue
            already.add((r["statement_id"], r["scenario_idx"], r["judge"], cond))

    # Build remaining work: for each Grok cell × each (judge in {gpt, gemini}) × each condition
    cells_to_do: list[tuple] = []
    for r in opposite_rows:
        sid, scen = r["statement_id"], r["scenario_idx"]
        for cond, cond_internal in (("bare", "variant_A"), ("phase_4", "rubric_plus_spec")):
            for judge in ("gpt", "gemini"):
                if (sid, scen, judge, cond_internal) not in already:
                    cells_to_do.append((sid, scen, r["generator"], r["user_query"], r["response"], cond, judge))

    print(f"Phase 2 — sync GPT + Gemini on Grok-opposite cells")
    print(f"  total Grok cells available: {len(opposite_rows)}")
    print(f"  judgments still needed: {len(cells_to_do)}")
    if not cells_to_do:
        print("  nothing to do")
        return 0

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    log = RawAPILogger("e9_phase2_judge_grok_sync")
    print(f"  raw log dir: {log.run_dir}\n")

    new_rows: list[dict[str, Any]] = []
    n_done = 0
    t0 = time.time()

    def process(item):
        sid, scen, gen, user_q, resp_text, cond, judge = item
        stmt = spec_by_id[sid]
        examples = get_examples(stmt)
        rubric = rubrics.get(sid)
        sys_text = JUDGE_A_SYSTEM if cond == "bare" else JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
        cond_internal = "variant_A" if cond == "bare" else "rubric_plus_spec"
        user_text = build_user_prompt(stmt, examples, rubric, user_q, resp_text, cond)
        key = {"sid": sid, "scenario_idx": scen, "generator": gen, "condition": cond}
        try:
            if judge == "gpt":
                data = call_gpt_json(log, oai, role=f"judge_gpt_{cond}", key=key,
                                     system=sys_text, user=user_text, max_tokens=1500)
            else:
                data = call_gemini_json(log, gem, role=f"judge_gemini_{cond}", key=key,
                                        system=sys_text, user=user_text, max_tokens=1500)
            score = int(data["score"]) if data.get("score") is not None else None
            if score is not None and not 1 <= score <= 5:
                score = None
            return {
                "condition": cond_internal, "judge": judge, "statement_id": sid,
                "scenario_idx": scen, "generator": gen, "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
                "example_refs": data.get("example_refs") or [],
                "rubric_spec_tension": data.get("rubric_spec_tension"),
            }
        except Exception as exc:
            return {"condition": cond_internal, "judge": judge, "statement_id": sid,
                    "scenario_idx": scen, "generator": gen, "error": str(exc)[:200]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, it): it for it in cells_to_do}
        for fut in as_completed(futs):
            r = fut.result()
            new_rows.append(r)
            n_done += 1
            if n_done % 100 == 0 or n_done == len(cells_to_do):
                elapsed = time.time() - t0
                n_err = sum(1 for x in new_rows if "error" in x)
                print(f"  progress: {n_done}/{len(cells_to_do)} ({100*n_done/len(cells_to_do):.0f}%, {elapsed:.0f}s, errors={n_err})")

    # Append to existing per_judgment_opposite.jsonl
    existing = load_jsonl(PER_JUDGMENT_OUT) if PER_JUDGMENT_OUT.exists() else []
    merged = existing + new_rows
    write_jsonl(merged, PER_JUDGMENT_OUT)

    n_ok = sum(1 for r in new_rows if r.get("score") is not None)
    print(f"\n== Phase 2 TOTALS ==")
    print(f"  new rows: {n_ok}/{len(new_rows)} scored")
    print(f"  output: {PER_JUDGMENT_OUT} ({len(merged)} rows total)")
    print(f"  wall: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
