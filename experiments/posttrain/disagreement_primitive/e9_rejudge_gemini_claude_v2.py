"""Re-judge Gemini + Claude under phase_4 with v2 rubrics on the 3 unfixed statements.

Tests the "frozen judges" hypothesis: did v2 fail to fix no_agenda /
comply_with_laws / sexual_content_involving_minors because the OTHER judges
(Gemini, Claude) weren't re-judged with v2, or because the disagreement is
genuinely irreducible?

Sync execution (per user instruction — not batch). ~480 calls (3 stmts × ~80
cells × 2 judges). Estimated cost: ~$2.40 (Gemini $0.24, Claude $2.16).

Output: appends to per_judgment_v2.jsonl with rubric_version: "v2".
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
    SPEC_PATH,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
    write_jsonl,
)
from e8_phase2_cross_model import call_gemini_json
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL, ANTHROPIC_MODEL_CREATED_AT
from e9_judge_opposite_mode import JUDGMENT_TOOL_1_5
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
RUBRICS_V2_PATH = DIR / "e8_rubrics_v2.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
OUT_PATH = DIR / "per_judgment_v2.jsonl"

UNFIXED = ["no_agenda", "comply_with_laws", "sexual_content_involving_minors"]
GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]


def build_user_prompt(stmt, examples, rubric, user_q, response_text):
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def call_claude_tool_use(api_key, system, user, max_tokens=600):
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
        json=body, timeout=120.0,
    )
    if resp.status_code != 200:
        snippet = resp.text[:500].replace(api_key, "<REDACTED>")
        raise RuntimeError(f"HTTP {resp.status_code}: {snippet}")
    return resp.json()


def extract_claude_args(api_resp):
    blocks = api_resp.get("content") or []
    tu = next((b for b in blocks if b.get("type") == "tool_use"), None)
    if tu is None:
        types_seen = [b.get("type") for b in blocks]
        raise ValueError(f"no tool_use block; got types={types_seen}")
    args = tu.get("input")
    if not isinstance(args, dict):
        raise ValueError("tool_use.input not a dict")
    return args


def main():
    target_sids = set(UNFIXED)
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics_v2 = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_V2_PATH) if "error" not in r}
    missing = target_sids - set(rubrics_v2.keys())
    if missing:
        raise SystemExit(f"v2 rubrics missing for: {missing}")

    # Already-done check (skip cells already in per_judgment_v2 for these judges)
    already = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.open():
            r = json.loads(line)
            if r.get("rubric_version") != "v2": continue
            if r.get("condition") != "rubric_plus_spec": continue
            if r.get("judge") not in {"gemini", "claude"}: continue
            if r.get("score") is None: continue
            already.add((r["statement_id"], r["scenario_idx"], r["generator"], r["judge"]))

    # Build cells
    cells = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if sid not in target_sids: continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    if OPPOSITE_RESPONSES.exists():
        for r in load_jsonl(OPPOSITE_RESPONSES):
            if "error" in r: continue
            if r.get("statement_id") not in target_sids: continue
            cells.append((r["statement_id"], r["scenario_idx"], r["generator"], r["user_query"], r["response"]))

    cells.sort(key=lambda c: (c[0], c[1], c[2]))

    # Expand to (cell × judge) work items
    work = []
    for cell in cells:
        sid, scen, gen, _, _ = cell
        for judge in ("gemini", "claude"):
            if (sid, scen, gen, judge) in already: continue
            work.append((cell, judge))

    print(f"Re-judging Gemini + Claude phase_4 with v2 rubrics on 3 unfixed statements")
    print(f"  statements: {sorted(target_sids)}")
    print(f"  cells: {len(cells)}")
    print(f"  work items (cell × judge): {len(work)} (skipped {len(cells)*2 - len(work)} already done)")

    gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False)
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    log = RawAPILogger("e9_rejudge_gemini_claude_v2")
    print(f"  raw log dir: {log.run_dir}\n")

    new_rows = []
    overall_t0 = time.time()
    n_done = 0

    def process(item):
        cell, judge = item
        sid, scen, gen, user_q, resp_text = cell
        rubric_v2 = rubrics_v2[sid]
        examples = get_examples(spec[sid])
        user_text = build_user_prompt(spec[sid], examples, rubric_v2, user_q, resp_text)
        sys_text = JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
        key = {"sid": sid, "scenario_idx": scen, "generator": gen,
               "condition": "phase_4", "rubric_version": "v2", "judge": judge}
        try:
            if judge == "gemini":
                data = call_gemini_json(log, gem, role=f"judge_gemini_phase_4_v2", key=key,
                                        system=sys_text, user=user_text, max_tokens=1500)
                score = int(data["score"]) if data.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                return {
                    "condition": "rubric_plus_spec", "judge": "gemini",
                    "rubric_version": "v2",
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "score": score, "reasoning": data.get("reasoning"),
                    "spec_quotes": data.get("spec_quotes") or [],
                    "rubric_quotes": data.get("rubric_quotes") or [],
                    "example_refs": data.get("example_refs") or [],
                    "rubric_spec_tension": data.get("rubric_spec_tension"),
                }
            else:  # claude
                key["model"] = ANTHROPIC_MODEL
                key["model_created_at"] = ANTHROPIC_MODEL_CREATED_AT
                api_resp = log.call(role="judge_claude_phase_4_v2", key=key,
                                    fn=lambda: call_claude_tool_use(anthropic_key, sys_text, user_text))
                args = extract_claude_args(api_resp)
                score = int(args["score"]) if args.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                return {
                    "condition": "rubric_plus_spec", "judge": "claude",
                    "rubric_version": "v2",
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "score": score, "reasoning": args.get("reasoning"),
                    "spec_quotes": args.get("spec_quotes") or [],
                    "rubric_quotes": args.get("rubric_quotes") or [],
                    "example_refs": args.get("example_refs") or [],
                    "rubric_spec_tension": args.get("rubric_spec_tension"),
                    "tension_description": args.get("tension_description") or "",
                    "_usage": api_resp.get("usage"),
                }
        except Exception as exc:
            return {"condition": "rubric_plus_spec", "judge": judge,
                    "rubric_version": "v2",
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "error": str(exc)[:200]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, w): w for w in work}
        for fut in as_completed(futs):
            r = fut.result()
            new_rows.append(r)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(work):
                elapsed = time.time() - overall_t0
                n_err = sum(1 for x in new_rows if "error" in x)
                print(f"  progress: {n_done}/{len(work)} ({100*n_done/len(work):.0f}%, {elapsed:.0f}s, errors={n_err})")

    # Append to per_judgment_v2.jsonl
    existing = load_jsonl(OUT_PATH) if OUT_PATH.exists() else []
    write_jsonl(existing + new_rows, OUT_PATH)
    n_ok = sum(1 for r in new_rows if r.get("score") is not None)
    print(f"\nwrote {OUT_PATH} (added {n_ok}/{len(new_rows)} new scored rows)")
    print(f"wall: {time.time() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
