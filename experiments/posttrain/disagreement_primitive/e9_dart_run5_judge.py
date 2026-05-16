# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DART Run 5 — judging script for the example-additions experiment.

Runs 2 conditions on no_topic_off_limits:
  C_RUBRIC: spec_v1 + rubric_v2 (what §1.9.4 hierarchical rule adopted)
  C_EXAMPLES: spec_v1+Gemini_example + rubric_v1 (counterfactual — would the example alone have helped?)

3 judges (GPT-5.1 sync, Gemini sync, Claude batch). Writes to
dart_iteration/no_topic_off_limits/run5_per_judgment_round_{N}.jsonl.

Cells: same 80 cells used in Run 4 (gpt-5.1, Qwen, gemini-flash, grok-opposite).

Usage: source .env && source .env2 && .venv/bin/python e9_dart_run5_judge.py --round 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from google import genai

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_paired_indirection import (
    SPEC_PATH,
    get_examples,
    render_anchors,
    render_examples,
    call_gpt_json,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL
from e9_judge_opposite_mode import JUDGMENT_TOOL_1_5
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
SID = "no_topic_off_limits"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def load_v1_rubric() -> dict:
    for r in load_jsonl(DIR / "e8_rubrics_v1.jsonl"):
        if r["statement_id"] == SID:
            return r["rubric"]
    raise KeyError(SID)


def load_spec_record_v1() -> dict:
    for r in load_jsonl(SPEC_PATH):
        if r["id"] == SID:
            return r
    raise KeyError(SID)


def load_responses() -> list[tuple[str, int, str, str, str]]:
    """Returns list of (sid, scenario_idx, generator, user_query, response_text)."""
    cells = []
    for r in load_jsonl(DIR / "e8_responses.jsonl"):
        if r.get("statement_id") != SID:
            continue
        for col, label in [
            ("response_gpt", "gpt-5.1"),
            ("response_weak", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
            ("response_gemini", "gemini-3-flash-preview"),
        ]:
            text = r.get(col)
            if text:
                cells.append((SID, r["scenario_idx"], label, r["user_query"], text))
    for r in load_jsonl(DIR / "e9_opposite_mode_responses.jsonl"):
        if "error" in r:
            continue
        if r.get("statement_id") != SID:
            continue
        cells.append((SID, r["scenario_idx"], r["generator"], r["user_query"], r["response"]))
    cells.sort(key=lambda c: (c[1], c[2]))
    return cells


def build_user_prompt(stmt_text: str, examples: list[dict], rubric: dict, user_q: str, response_text: str) -> str:
    return (
        f"SPEC STATEMENT:\n{stmt_text}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def make_conditions(round_n: int) -> dict:
    """Return {condition_name: (spec_text, examples, rubric)}."""
    sid_dir = ITER_DIR / SID

    spec_v1 = load_spec_record_v1()
    rubric_v1 = load_v1_rubric()

    # C_RUBRIC: what hierarchical rule adopted
    rubric_v2_path = sid_dir / f"run5_rubric_v{round_n + 1}.json"
    if not rubric_v2_path.exists():
        raise SystemExit(f"missing {rubric_v2_path} — run synthesize first")
    rubric_v2 = json.loads(rubric_v2_path.read_text())

    # C_EXAMPLES: counterfactual — extract Gemini's proposed example from diagnoses, append to v1 spec
    gem_diag_path = sid_dir / f"run5_round_{round_n}_diagnoses_gem.json"
    gem_diag = json.loads(gem_diag_path.read_text())
    gem_examples = gem_diag.get("spec_example_additions") or []
    spec_with_examples = json.loads(json.dumps(spec_v1))
    spec_with_examples.setdefault("metadata", {}).setdefault("examples", [])
    for e in gem_examples:
        spec_with_examples["metadata"]["examples"].append(
            {
                "user_query": e.get("user_query", ""),
                "good_response": e.get("good_response", ""),
                "bad_response": e.get("bad_response", ""),
                "description": e.get("description", ""),
            }
        )

    return {
        "C_RUBRIC": (spec_v1["text"], get_examples(spec_v1), rubric_v2),
        "C_EXAMPLES": (spec_with_examples["text"], get_examples(spec_with_examples), rubric_v1),
    }


# --------------- Judge calls (sync GPT + Gemini, batch Claude) ---------------


def call_gemini_judge(
    client: genai.Client, log: RawAPILogger, role: str, key: dict, system: str, user: str, max_tokens: int = 1500
) -> dict:
    from google.genai import types

    # Flash supports thinking_level="minimal" (Pro does not). Use minimal for
    # judge-side calls — high throughput, classification-style task. See
    # dart.md Gotcha 17.
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        max_output_tokens=max_tokens,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    raw = log.call(
        role=role,
        key=key,
        fn=lambda: client.models.generate_content(model="gemini-3-flash-preview", contents=user, config=cfg),
    )
    text = raw.text or ""
    if "```" in text:
        text = text.split("```")[1].lstrip("json\n").strip()
    return json.loads(text) if text.strip() else {}


def submit_claude_batch(api_key: str, requests: list[dict], job_dir: Path, name: str):
    from batch_anthropic import submit

    return submit(api_key, requests, job_dir=job_dir, name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    sid_dir = ITER_DIR / SID
    sid_dir.mkdir(parents=True, exist_ok=True)
    out_path = sid_dir / f"run5_per_judgment_round_{args.round}.jsonl"

    conditions = make_conditions(args.round)
    print(f"Run 5 — round {args.round} judging on {SID}")
    print(f"  conditions: {list(conditions.keys())}")
    cells = load_responses()
    print(f"  cells: {len(cells)}\n")

    log = RawAPILogger(f"e9_dart_run5_round_{args.round}_judge")
    print(f"  raw log dir: {log.run_dir}\n")

    # Existing judgments from previous round-N runs would dedupe; for now treat as fresh
    existing_keys = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            existing_keys.add(
                (r.get("judge"), r.get("statement_id"), r.get("scenario_idx"), r.get("generator"), r.get("condition"))
            )

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = genai.Client(api_key=(os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]))
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    # ============ Sync GPT + Gemini ============
    sync_rows = []

    def call_gpt_one(cell, cond, stmt_text, examples, rubric):
        sid, scen, gen, uq, resp = cell
        if ("gpt", sid, scen, gen, cond) in existing_keys:
            return None
        user = build_user_prompt(stmt_text, examples, rubric, uq, resp)
        try:
            data = call_gpt_json(
                log,
                oai,
                role=f"run5_round_{args.round}_judge_gpt_{cond}",
                key={"sid": sid, "cond": cond, "scen": scen, "gen": gen},
                system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                user=user,
                max_tokens=1500,
            )
            score = int(data["score"]) if data.get("score") is not None else None
            return {
                "judge": "gpt",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
            }
        except Exception as e:
            return {
                "judge": "gpt",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "error": str(e)[:200],
            }

    def call_gem_one(cell, cond, stmt_text, examples, rubric):
        sid, scen, gen, uq, resp = cell
        if ("gemini", sid, scen, gen, cond) in existing_keys:
            return None
        user = build_user_prompt(stmt_text, examples, rubric, uq, resp)
        try:
            data = call_gemini_judge(
                gem,
                log,
                role=f"run5_round_{args.round}_judge_gem_{cond}",
                key={"sid": sid, "cond": cond, "scen": scen, "gen": gen},
                system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                user=user,
            )
            score = int(data["score"]) if data.get("score") is not None else None
            return {
                "judge": "gemini",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
            }
        except Exception as e:
            return {
                "judge": "gemini",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "error": str(e)[:200],
            }

    print("=== Sync GPT + Gemini ===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for cond, (stmt_text, examples, rubric) in conditions.items():
            for cell in cells:
                futs.append(ex.submit(call_gpt_one, cell, cond, stmt_text, examples, rubric))
                futs.append(ex.submit(call_gem_one, cell, cond, stmt_text, examples, rubric))
        for f in as_completed(futs):
            r = f.result()
            if r:
                sync_rows.append(r)
    print(f"  sync done: {len(sync_rows)} rows in {time.time()-t0:.0f}s")

    # Append sync to file
    with out_path.open("a") as f:
        for r in sync_rows:
            f.write(json.dumps(r) + "\n")

    # ============ Claude batch ============
    print("\n=== Claude batch ===")
    job_dir = Path(
        f"results/raw/e9_dart_run5_claude_batch/round_{args.round}_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    requests_per_cond = {}
    custom_id_map = {}

    for cond, (stmt_text, examples, rubric) in conditions.items():
        reqs = []
        for sid, scen, gen, uq, resp in cells:
            if ("claude", sid, scen, gen, cond) in existing_keys:
                continue
            user_text = build_user_prompt(stmt_text, examples, rubric, uq, resp)
            cid = f"r{args.round}::{sid}::{cond}::{scen}::{gen}"
            custom_id_map[cid] = (sid, cond, scen, gen)
            reqs.append(
                ba.build_request(
                    custom_id=cid,
                    model=ANTHROPIC_MODEL,
                    system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=1500,
                    tools=[JUDGMENT_TOOL_1_5],
                    tool_choice={"type": "tool", "name": "submit_judgment"},
                    thinking={"type": "disabled"},
                    temperature=0,
                    cache_user_prefix=ba.prefix_before(user_text),
                )
            )
        requests_per_cond[cond] = reqs
        print(f"  {cond}: {len(reqs)} requests prepared")

    batch_state = {
        "round": args.round,
        "conditions": list(conditions.keys()),
        "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
        "job_dir": str(job_dir),
        "batches": {},
    }
    cmap_path = job_dir / "custom_id_map.json"
    cmap_path.write_text(json.dumps(custom_id_map, indent=2))

    for cond, reqs in requests_per_cond.items():
        if not reqs:
            continue
        name = f"run5_round_{args.round}_{cond}"
        result = submit_claude_batch(anthropic_key, reqs, job_dir=job_dir, name=name)
        batch_state["batches"][cond] = {
            "batch_id": result["batch_id"],
            "name": name,
            "n_requests": len(reqs),
            "submitted_at": time.time(),
        }
        print(f"  {cond}: submitted batch {result['batch_id']}")

    pointer = sid_dir / f"run5_round_{args.round}_batches.json"
    pointer.write_text(json.dumps(batch_state, indent=2))
    print(f"\nWrote batch pointer: {pointer}")
    print(f"Custom ID map: {cmap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
