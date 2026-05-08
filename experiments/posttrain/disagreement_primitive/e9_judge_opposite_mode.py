"""Judge the 160 Grok-opposite-mode responses at 1-5 scale.

For each (statement_id, scenario_idx) in e9_opposite_mode_responses.jsonl,
runs all 3 judges (gpt-5.1, gemini-3-flash, claude-sonnet-4-6 via tool-use)
under both bare (variant_A) and phase_4 (rubric_plus_spec) conditions on the
ORIGINAL 1-5 anchored scale. Same prompts as e8_paired_indirection /
e8_phase4_rubric_plus_spec. Same rubrics from e8_rubrics.jsonl.

Outputs:
  - per_judgment_opposite.jsonl                 (gpt + gemini + claude flat)
  - claude_judge_v0_opposite/<sid>/{bare,phase_4}_opposite_claude.jsonl

NOTE: reasoning_effort="none" enforced via call_gpt_json (HARD project rule).
Cost estimate: ~$4.80 total (~$2.90 Anthropic).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
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
from e9_claude_judge import ANTHROPIC_MODEL, ANTHROPIC_MODEL_CREATED_AT
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
RUBRICS_1_5 = DIR / "e8_rubrics.jsonl"
PER_JUDGMENT_OUT = DIR / "per_judgment_opposite.jsonl"
CLAUDE_OUT_DIR = DIR / "claude_judge_v0_opposite"


# Claude tool-use schema for 1-5 judging
JUDGMENT_TOOL_1_5 = {
    "name": "submit_judgment",
    "description": "Submit a single 1-5 anchored compliance judgment. Call exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {"type": "string"},
            "spec_quotes": {"type": "array", "items": {"type": "string"}},
            "rubric_quotes": {"type": "array", "items": {"type": "string"}},
            "example_refs": {"type": "array", "items": {"type": "string"}},
            "rubric_spec_tension": {"type": "boolean"},
            "tension_description": {"type": "string"},
        },
        "required": ["score", "reasoning", "spec_quotes", "rubric_quotes",
                     "example_refs", "rubric_spec_tension", "tension_description"],
    },
}


def build_user_prompt(stmt: dict[str, Any], examples: list, rubric: dict | None,
                      user_q: str, response_text: str, condition: str) -> str:
    if condition == "bare":
        return (
            f"STATEMENT TEXT:\n{stmt['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response_text}\n\n"
            "Score per the schema."
        )
    elif condition == "phase_4":
        if rubric is None:
            raise ValueError("phase_4 requires rubric")
        return (
            f"SPEC STATEMENT:\n{stmt['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"RUBRIC:\n{render_anchors(rubric)}\n\n"
            f"USER QUERY: {user_q}\n\n"
            f"ASSISTANT RESPONSE: {response_text}\n\n"
            "Score per the schema."
        )
    else:
        raise ValueError(condition)


def call_claude_tool_use(api_key: str, system: str, user: str, max_tokens: int = 600) -> dict[str, Any]:
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [JUDGMENT_TOOL_1_5],
        "tool_choice": {"type": "tool", "name": "submit_judgment"},
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json=body,
        timeout=120.0,
    )
    if resp.status_code != 200:
        snippet = resp.text[:500].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"HTTP {resp.status_code}: {snippet}")
    return resp.json()


def extract_claude_args(api_resp: dict[str, Any]) -> dict[str, Any]:
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
    spec_by_id = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_1_5) if "error" not in r}
    opposite_rows = [r for r in load_jsonl(OPPOSITE_RESPONSES) if "error" not in r]

    cells = [(r["statement_id"], r["scenario_idx"], r["generator"], r["user_query"], r["response"])
             for r in opposite_rows]
    cells.sort(key=lambda t: (t[0], t[1]))
    print(f"Judging {len(cells)} Grok-opposite cells")
    print(f"  conditions: bare, phase_4")
    print(f"  judges: gpt-5.1, gemini-3-flash, claude-sonnet-4-6 (tool-use forced)")
    print(f"  total calls: {len(cells) * 2 * 3}")

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    log = RawAPILogger("e9_judge_opposite_mode")
    print(f"  raw log dir: {log.run_dir}\n")

    per_judgment_rows: list[dict[str, Any]] = []
    claude_rows_by_sid_cond: dict[tuple[str, str], list[dict[str, Any]]] = {}

    overall_t0 = time.time()
    n_done = 0
    n_total = len(cells) * 6

    def process_cell(cell):
        sid, scen, gen, user_q, resp_text = cell
        stmt = spec_by_id[sid]
        examples = get_examples(stmt)
        rubric = rubrics.get(sid)
        local_pj: list[dict[str, Any]] = []
        local_claude: dict[tuple[str, str], dict] = {}

        for cond in ("bare", "phase_4"):
            user_text = build_user_prompt(stmt, examples, rubric, user_q, resp_text, cond)
            sys_text = JUDGE_A_SYSTEM if cond == "bare" else JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
            cond_internal = "variant_A" if cond == "bare" else "rubric_plus_spec"

            # GPT
            try:
                data = call_gpt_json(log, oai, role=f"judge_gpt_{cond}_opposite",
                                     key={"sid": sid, "scenario_idx": scen, "generator": gen, "condition": cond},
                                     system=sys_text, user=user_text, max_tokens=1500)
                score = int(data["score"]) if data.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                local_pj.append({
                    "condition": cond_internal, "judge": "gpt", "statement_id": sid,
                    "scenario_idx": scen, "generator": gen, "score": score,
                    "reasoning": data.get("reasoning"), "spec_quotes": data.get("spec_quotes") or [],
                    "rubric_quotes": data.get("rubric_quotes") or [],
                    "example_refs": data.get("example_refs") or [],
                    "rubric_spec_tension": data.get("rubric_spec_tension"),
                })
            except Exception as e:
                local_pj.append({"condition": cond_internal, "judge": "gpt", "statement_id": sid,
                                 "scenario_idx": scen, "generator": gen, "error": str(e)[:200]})

            # Gemini
            try:
                data = call_gemini_json(log, gem, role=f"judge_gemini_{cond}_opposite",
                                        key={"sid": sid, "scenario_idx": scen, "generator": gen, "condition": cond},
                                        system=sys_text, user=user_text, max_tokens=1500)
                score = int(data["score"]) if data.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                local_pj.append({
                    "condition": cond_internal, "judge": "gemini", "statement_id": sid,
                    "scenario_idx": scen, "generator": gen, "score": score,
                    "reasoning": data.get("reasoning"), "spec_quotes": data.get("spec_quotes") or [],
                    "rubric_quotes": data.get("rubric_quotes") or [],
                    "example_refs": data.get("example_refs") or [],
                    "rubric_spec_tension": data.get("rubric_spec_tension"),
                })
            except Exception as e:
                local_pj.append({"condition": cond_internal, "judge": "gemini", "statement_id": sid,
                                 "scenario_idx": scen, "generator": gen, "error": str(e)[:200]})

            # Claude (tool-use)
            try:
                key = {"sid": sid, "scenario_idx": scen, "generator": gen, "condition": cond,
                       "model": ANTHROPIC_MODEL, "model_created_at": ANTHROPIC_MODEL_CREATED_AT}
                api_resp = log.call(role=f"judge_claude_{cond}_opposite", key=key,
                                    fn=lambda: call_claude_tool_use(anthropic_key, sys_text, user_text))
                args = extract_claude_args(api_resp)
                score = int(args["score"]) if args.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                row = {
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "user_query": user_q, "response": resp_text, "score": score,
                    "reasoning": args.get("reasoning"), "spec_quotes": args.get("spec_quotes") or [],
                    "rubric_quotes": args.get("rubric_quotes") or [],
                    "example_refs": args.get("example_refs") or [],
                    "_usage": api_resp.get("usage"),
                }
                if cond == "phase_4":
                    row["rubric_spec_tension"] = args.get("rubric_spec_tension")
                    row["tension_description"] = args.get("tension_description") or ""
                local_claude[(sid, cond)] = row
                local_pj.append({
                    "condition": cond_internal, "judge": "claude", "statement_id": sid,
                    "scenario_idx": scen, "generator": gen, "score": score,
                    "reasoning": args.get("reasoning"), "spec_quotes": args.get("spec_quotes") or [],
                    "rubric_quotes": args.get("rubric_quotes") or [],
                    "example_refs": args.get("example_refs") or [],
                    "rubric_spec_tension": args.get("rubric_spec_tension"),
                })
            except Exception as e:
                local_claude[(sid, cond)] = {
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "user_query": user_q, "response": resp_text, "error": str(e)[:200],
                }
                local_pj.append({"condition": cond_internal, "judge": "claude", "statement_id": sid,
                                 "scenario_idx": scen, "generator": gen, "error": str(e)[:200]})
        return {"per_judgment": local_pj, "claude": local_claude}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process_cell, c): c for c in cells}
        for fut in as_completed(futs):
            res = fut.result()
            per_judgment_rows.extend(res["per_judgment"])
            for k, v in res["claude"].items():
                claude_rows_by_sid_cond.setdefault(k, []).append(v)
            n_done += 6
            if n_done % 60 == 0 or n_done >= n_total:
                elapsed = time.time() - overall_t0
                print(f"  progress: {n_done}/{n_total} ({100*n_done/n_total:.0f}%, {elapsed:.0f}s)")

    write_jsonl(per_judgment_rows, PER_JUDGMENT_OUT)
    print(f"\n  wrote {PER_JUDGMENT_OUT} ({len(per_judgment_rows)} rows)")

    CLAUDE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (sid, cond), rows in claude_rows_by_sid_cond.items():
        rows.sort(key=lambda r: r["scenario_idx"])
        out = CLAUDE_OUT_DIR / sid / f"{cond}_opposite_claude.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_ok = sum(1 for r in rows if r.get("score") is not None)
        print(f"  wrote {out} ({n_ok}/{len(rows)} scored)")

    n_pj_ok = sum(1 for r in per_judgment_rows if r.get("score") is not None)
    print(f"\n== TOTALS ==")
    print(f"  rows: {n_pj_ok}/{len(per_judgment_rows)} scored")
    print(f"  wall: {time.time() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
