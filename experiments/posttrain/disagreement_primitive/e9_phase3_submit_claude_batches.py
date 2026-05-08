"""Phase 3 — Submit two Claude batches via Anthropic Batch API (50% off).

Batches:
  - Batch A "judge_grok_opposite": Claude judges all Grok-opposite cells
    (across all 46 statements) under bare + phase_4 conditions, IF the
    cell is not already in claude_judge_v0_opposite/<sid>/.
  - Batch B "judge_existing_3gens": Claude judges existing 3 generators
    (gpt-5.1, Qwen, gemini) on the 38 statements not yet in claude_judge_v0/,
    under bare + phase_4 conditions.

Both use:
  - claude-sonnet-4-6 model
  - thinking: disabled
  - tool-use forcing with submit_judgment (1-5 schema)
  - 1-5 anchored prompts
  - Rubrics from e8_rubrics.jsonl

Persists batch state to:
  - results/raw/e9_claude_batches/2026-05-08T<...>/{judge_grok_opposite,judge_existing_3gens}_state.json

Run e9_phase4_fetch_claude_batches.py to poll + collect once complete.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    JUDGE_A_SYSTEM,
    SPEC_PATH,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL
import batch_anthropic as ba

DIR = Path("experiments/posttrain/disagreement_primitive")
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
RUBRICS = DIR / "e8_rubrics.jsonl"
CLAUDE_OPPOSITE_DIR = DIR / "claude_judge_v0_opposite"
CLAUDE_EXISTING_DIR = DIR / "claude_judge_v0"

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]

JUDGMENT_TOOL_1_5 = {
    "name": "submit_judgment",
    "description": "Submit a single 1-5 anchored compliance judgment.",
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


def already_done_set_for_dir(claude_dir: Path) -> set[tuple[str, str, int, str]]:
    """Returns {(sid, condition_short, scen, generator)} of cells already scored."""
    out: set[tuple[str, str, int, str]] = set()
    if not claude_dir.exists():
        return out
    for sid_dir in claude_dir.iterdir():
        if not sid_dir.is_dir():
            continue
        for jsonl in sid_dir.glob("*.jsonl"):
            cond_short = "bare" if "bare" in jsonl.name else "phase_4"
            for line in jsonl.open():
                r = json.loads(line)
                if r.get("score") is not None:
                    out.add((r["statement_id"], cond_short, r["scenario_idx"], r["generator"]))
    return out


def custom_id_for(sid: str, cond: str, scen: int, gen: str) -> str:
    safe_gen = gen.replace("/", "-")
    return f"{sid}::{cond}::{scen}::{safe_gen}"


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set; run: source .env2")

    spec_by_id = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS) if "error" not in r}

    # ---- Build Batch A: Claude on Grok-opposite responses ----
    opposite_rows = [r for r in load_jsonl(OPPOSITE_RESPONSES) if "error" not in r]
    done_opposite = already_done_set_for_dir(CLAUDE_OPPOSITE_DIR)

    requests_a: list[dict[str, Any]] = []
    for r in opposite_rows:
        sid = r["statement_id"]
        if sid not in spec_by_id:
            continue
        for cond_short, cond_internal in (("bare", "variant_A"), ("phase_4", "rubric_plus_spec")):
            cell_key = (sid, cond_short, r["scenario_idx"], r["generator"])
            if cell_key in done_opposite:
                continue
            stmt = spec_by_id[sid]
            user_text = build_user_prompt(
                stmt, get_examples(stmt), rubrics.get(sid),
                r["user_query"], r["response"], cond_short,
            )
            sys_text = JUDGE_A_SYSTEM if cond_short == "bare" else JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
            requests_a.append(ba.build_request(
                custom_id=custom_id_for(sid, cond_short, r["scenario_idx"], r["generator"]),
                model=ANTHROPIC_MODEL,
                system=sys_text,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=600,
                tools=[JUDGMENT_TOOL_1_5],
                tool_choice={"type": "tool", "name": "submit_judgment"},
                thinking={"type": "disabled"},
                temperature=0,
            ))

    # ---- Build Batch B: Claude on existing 3 generators × 38 statements ----
    done_existing = already_done_set_for_dir(CLAUDE_EXISTING_DIR)
    existing_rows = load_jsonl(EXISTING_RESPONSES)

    requests_b: list[dict[str, Any]] = []
    for r in existing_rows:
        sid = r.get("statement_id")
        if sid not in spec_by_id:
            continue
        scen = r["scenario_idx"]
        user_q = r["user_query"]
        for gen_label, col in GEN_KEYS:
            response_text = r.get(col)
            if not response_text:
                continue
            for cond_short, cond_internal in (("bare", "variant_A"), ("phase_4", "rubric_plus_spec")):
                cell_key = (sid, cond_short, scen, gen_label)
                if cell_key in done_existing:
                    continue
                stmt = spec_by_id[sid]
                user_text = build_user_prompt(
                    stmt, get_examples(stmt), rubrics.get(sid),
                    user_q, response_text, cond_short,
                )
                sys_text = JUDGE_A_SYSTEM if cond_short == "bare" else JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
                requests_b.append(ba.build_request(
                    custom_id=custom_id_for(sid, cond_short, scen, gen_label),
                    model=ANTHROPIC_MODEL,
                    system=sys_text,
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=600,
                    tools=[JUDGMENT_TOOL_1_5],
                    tool_choice={"type": "tool", "name": "submit_judgment"},
                    thinking={"type": "disabled"},
                    temperature=0,
                ))

    # ---- Submit ----
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    job_dir = Path("results/raw/e9_claude_batches") / ts
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 — Anthropic batch submission")
    print(f"  job_dir: {job_dir}")
    print(f"  Batch A (judge_grok_opposite): {len(requests_a)} requests")
    print(f"  Batch B (judge_existing_3gens): {len(requests_b)} requests")
    print(f"  Total: {len(requests_a) + len(requests_b)} requests\n")

    state_a = ba.submit(api_key, requests_a, job_dir, name="judge_grok_opposite") if requests_a else None
    if state_a:
        print(f"  Batch A submitted: id={state_a['batch_id']}")

    state_b = ba.submit(api_key, requests_b, job_dir, name="judge_existing_3gens") if requests_b else None
    if state_b:
        print(f"  Batch B submitted: id={state_b['batch_id']}")

    # Write a top-level pointer for Phase 4 to find
    pointer_path = DIR / "e9_claude_batch_pointer.json"
    pointer_path.write_text(json.dumps({
        "submitted_at": time.time(),
        "submitted_at_iso": ts,
        "job_dir": str(job_dir),
        "batches": {
            "judge_grok_opposite": {"batch_id": state_a["batch_id"], "n_requests": len(requests_a)} if state_a else None,
            "judge_existing_3gens": {"batch_id": state_b["batch_id"], "n_requests": len(requests_b)} if state_b else None,
        },
    }, indent=2))
    print(f"\n  pointer: {pointer_path}")
    print(f"  --> run e9_phase4_fetch_claude_batches.py to poll + collect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
