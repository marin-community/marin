"""DART iterative validation — submit per-judge phase_4 judgments under C1/C2/C3.

For each (statement, condition, judge) triple, this script builds 1-5 anchored
phase_4 judge prompts and dispatches them:

  C1 = rubric_v2 + original spec text  (rubric edits only)
  C2 = original v1 rubric + spec_v2     (spec edits only)
  C3 = rubric_v2 + spec_v2              (both)

  C0 (control) is intentionally NOT re-judged — analysis reuses existing
  per_judgment data.

Cells per statement:
  - 3 generators × N scenarios from e8_responses.jsonl
    (gpt-5.1, Qwen2.5-7B-Instruct-Turbo, gemini-3-flash-preview)
  - 1 generator × N scenarios from e9_opposite_mode_responses.jsonl (Grok-opposite)

Judges:
  - GPT-5.1            (sync, ThreadPoolExecutor max_workers=8, reasoning_effort=none)
  - Gemini 3 Flash     (sync, ThreadPoolExecutor max_workers=4)
  - Claude Sonnet 4.6  (Anthropic Batch API; submit only — fetch in a separate script)

Outputs:
  - per_judgment_iter_round_{N}.jsonl   (sync GPT + Gemini results, append-mode)
  - round_{N}_batches.json              (Claude batch_ids + sidecar maps for fetch)

Run:
  source .env && source .env2 && \\
    .venv/bin/python experiments/posttrain/disagreement_primitive/e9_dart_iter_judge.py \\
      --round 1 --conditions C1,C2,C3 --statements all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_paired_indirection import (
    SPEC_PATH,
    call_gpt_json,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
)
from e8_phase2_cross_model import call_gemini_json
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL, ANTHROPIC_MODEL_CREATED_AT
from e9_phase3_submit_claude_batches import JUDGMENT_TOOL_1_5
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
DART_DIR = DIR / "dart_iteration"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]

ALL_STATEMENTS = [
    "ask_clarifying_questions",
    "assume_objective_pov",
    "avoid_abuse",
    "be_clear",
    "be_thorough_but_efficient",
    "comply_with_laws",
    "do_not_lie",
    "formatting",
    "letter_and_spirit",
    "no_topic_off_limits",
    "prevent_imminent_harm",
    "protect_privileged_messages",
    "refusal_style",
]

# Cost guardrails (USD). Abort sync GPT if estimated spend exceeds this.
GPT_SYNC_BUDGET_USD = 20.0
# Approx GPT-5.1 input / output prices (per 1M tokens). Used for budget ceiling
# only — not billed. As of 2026-05: $1.25/M in, $10/M out for gpt-5.1.
GPT_PRICE_IN_PER_M = 1.25
GPT_PRICE_OUT_PER_M = 10.0


def build_user_prompt(statement_text: str, examples: list[dict[str, Any]],
                      rubric: dict[str, Any], user_q: str, response_text: str) -> str:
    """Phase_4 prompt: SPEC STATEMENT + SPEC EXAMPLES + RUBRIC + query + response."""
    return (
        f"SPEC STATEMENT:\n{statement_text}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


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
        json=body, timeout=120.0,
    )
    if resp.status_code != 200:
        snippet = resp.text[:400].replace(api_key, "<REDACTED>")
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


def custom_id_for(sid: str, condition: str, scen: int, gen: str, round_n: int) -> str:
    raw = f"r{round_n}::{sid}::{condition}::{scen}::{gen}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:32]
    return f"j_{h}"  # 34 chars, well under 64


def load_v1_rubrics() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in load_jsonl(RUBRICS_V1_PATH):
        if "error" in r:
            continue
        out[r["statement_id"]] = r["rubric"]
    return out


def load_v2_artifacts(sids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Returns (rubric_v2_by_sid, spec_v2_text_by_sid)."""
    rubrics_v2: dict[str, dict[str, Any]] = {}
    specs_v2: dict[str, str] = {}
    for sid in sids:
        rv2 = DART_DIR / sid / "rubric_v2.json"
        sv2 = DART_DIR / sid / "spec_v2.txt"
        if not rv2.exists():
            raise SystemExit(f"missing {rv2}")
        if not sv2.exists():
            raise SystemExit(f"missing {sv2}")
        rubrics_v2[sid] = json.loads(rv2.read_text())
        specs_v2[sid] = sv2.read_text().strip()
    return rubrics_v2, specs_v2


def build_cells(target_sids: set[str]) -> list[tuple[str, int, str, str, str]]:
    """Returns list of (sid, scenario_idx, generator, user_query, response_text)."""
    cells: list[tuple[str, int, str, str, str]] = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if sid not in target_sids:
            continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    if OPPOSITE_RESPONSES.exists():
        for r in load_jsonl(OPPOSITE_RESPONSES):
            if "error" in r:
                continue
            sid = r.get("statement_id")
            if sid not in target_sids:
                continue
            cells.append((sid, r["scenario_idx"], r["generator"], r["user_query"], r["response"]))
    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    return cells


def resolve_for_condition(
    condition: str,
    sid: str,
    spec_orig: dict[str, Any],
    rubrics_v1: dict[str, dict[str, Any]],
    rubrics_v2: dict[str, dict[str, Any]],
    specs_v2: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Return (statement_text, rubric_dict) for the given condition + statement."""
    if condition == "C1":
        # rubric_v2 + original spec
        return spec_orig["text"], rubrics_v2[sid]
    if condition == "C2":
        # spec_v2 + v1 rubric
        if sid not in rubrics_v1:
            raise KeyError(f"v1 rubric missing for sid={sid}")
        return specs_v2[sid], rubrics_v1[sid]
    if condition == "C3":
        # both v2
        return specs_v2[sid], rubrics_v2[sid]
    raise ValueError(f"unknown condition {condition!r}")


def append_jsonl(rows: list[dict[str, Any]], path: Path, lock: Lock) -> None:
    """Append rows to path under a lock; create parent if needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--conditions", type=str, default="C1,C2,C3",
                   help="comma-separated subset of {C1,C2,C3}")
    p.add_argument("--statements", type=str, default="all",
                   help="'all' or comma-separated statement_ids")
    p.add_argument("--judges", type=str, default="gpt,gemini,claude",
                   help="comma-separated subset of {gpt,gemini,claude}")
    p.add_argument("--gpt-workers", type=int, default=8)
    p.add_argument("--gemini-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="build cells/prompts but don't dispatch any API calls")
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ("C1", "C2", "C3"):
            raise SystemExit(f"unsupported condition {c!r}; choose from C1,C2,C3")

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges:
        if j not in ("gpt", "gemini", "claude"):
            raise SystemExit(f"unsupported judge {j!r}")

    if args.statements == "all":
        target_sids = list(ALL_STATEMENTS)
    else:
        target_sids = [s.strip() for s in args.statements.split(",") if s.strip()]
        unknown = set(target_sids) - set(ALL_STATEMENTS)
        if unknown:
            raise SystemExit(f"unknown statements: {unknown}")
    target_sid_set = set(target_sids)

    spec_by_id = {json.loads(line)["id"]: json.loads(line)
                  for line in SPEC_PATH.open() if line.strip()}
    for sid in target_sids:
        if sid not in spec_by_id:
            raise SystemExit(f"statement_id {sid} not in {SPEC_PATH}")

    rubrics_v1 = load_v1_rubrics()
    rubrics_v2, specs_v2 = load_v2_artifacts(target_sids)

    cells = build_cells(target_sid_set)
    print(f"DART iter round={args.round}")
    print(f"  conditions: {conditions}")
    print(f"  statements: {target_sids}")
    print(f"  cells: {len(cells)}  (statements × scenarios × generators)")

    work: list[tuple[str, str, int, str, str, str, str, dict[str, Any]]] = []
    # tuple: (condition, sid, scen, gen, user_q, resp_text, statement_text, rubric)
    for cond in conditions:
        for sid, scen, gen, user_q, resp_text in cells:
            try:
                stmt_text, rubric = resolve_for_condition(
                    cond, sid, spec_by_id[sid], rubrics_v1, rubrics_v2, specs_v2,
                )
            except KeyError as exc:
                print(f"  skip cond={cond} sid={sid}: {exc}")
                continue
            work.append((cond, sid, scen, gen, user_q, resp_text, stmt_text, rubric))

    print(f"  work items per judge: {len(work)} (cells × conditions)")
    n_judges_active = len(judges)
    print(f"  total judgments to dispatch: {len(work) * n_judges_active}")

    # Output paths
    sync_out = DART_DIR / f"per_judgment_iter_round_{args.round}.jsonl"
    batches_pointer = DART_DIR / f"round_{args.round}_batches.json"
    sync_lock = Lock()

    # Idempotency: skip work items already done for sync judges.
    already_done: set[tuple[str, str, str, int, str]] = set()  # (judge, cond, sid, scen, gen)
    if sync_out.exists():
        for r in load_jsonl(sync_out):
            if r.get("score") is None:
                continue
            already_done.add((r["judge"], r["condition"],
                              r["statement_id"], r["scenario_idx"], r["generator"]))
        if already_done:
            print(f"  found {len(already_done)} already-scored sync rows in {sync_out.name}")

    if args.dry_run:
        print("\nDRY RUN — exiting before dispatch")
        sample = work[:3]
        for w in sample:
            cond, sid, scen, gen, user_q, resp, stmt, rub = w
            prompt = build_user_prompt(stmt, get_examples(spec_by_id[sid]), rub, user_q, resp)
            print(f"\n--- sample cond={cond} sid={sid} scen={scen} gen={gen} ---")
            print(prompt[:600] + ("..." if len(prompt) > 600 else ""))
        return 0

    # ---- API key checks (only for judges we'll actually call) ----
    if "gpt" in judges and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; source .env first")
    if "gemini" in judges and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY not set; source .env2 first")
    if "claude" in judges and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; source .env2 first")

    log = RawAPILogger(f"e9_dart_iter_round_{args.round}")
    print(f"  raw log dir: {log.run_dir}")

    overall_t0 = time.time()
    summary: dict[str, Any] = {
        "round": args.round, "conditions": conditions, "statements": target_sids,
        "judges": judges, "n_cells": len(cells), "n_work_items_per_judge": len(work),
    }

    # ============== GPT-5.1 SYNC ==============
    if "gpt" in judges:
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        gpt_work = [w for w in work
                    if ("gpt", w[0], w[1], w[2], w[3]) not in already_done]
        print(f"\n--- GPT-5.1 sync judge: {len(gpt_work)} items "
              f"(skipped {len(work) - len(gpt_work)} already done) ---")
        gpt_t0 = time.time()
        n_done = 0
        # Approx token budget. Each prompt ~600-1500 tokens, response ~300 tokens.
        approx_in_tokens_per = 1200
        approx_out_tokens_per = 400
        spend_estimate = (len(gpt_work) * (approx_in_tokens_per * GPT_PRICE_IN_PER_M
                                           + approx_out_tokens_per * GPT_PRICE_OUT_PER_M)
                          / 1_000_000)
        print(f"  estimated GPT spend: ~${spend_estimate:.2f}")
        if spend_estimate > GPT_SYNC_BUDGET_USD:
            raise SystemExit(f"GPT estimate ${spend_estimate:.2f} exceeds budget "
                             f"${GPT_SYNC_BUDGET_USD}; aborting.")

        def process_gpt(item):
            cond, sid, scen, gen, user_q, resp_text, stmt_text, rubric = item
            examples = get_examples(spec_by_id[sid])
            user_text = build_user_prompt(stmt_text, examples, rubric, user_q, resp_text)
            key = {"sid": sid, "scenario_idx": scen, "generator": gen,
                   "condition": cond, "round": args.round}
            try:
                data = call_gpt_json(log, oai, role=f"judge_gpt_{cond}",
                                     key=key, system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                                     user=user_text, max_tokens=1500)
                score = int(data["score"]) if data.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                return {
                    "judge": "gpt", "condition": cond,
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "score": score, "reasoning": data.get("reasoning"),
                    "spec_quotes": data.get("spec_quotes") or [],
                    "rubric_quotes": data.get("rubric_quotes") or [],
                    "example_refs": data.get("example_refs") or [],
                    "rubric_spec_tension": data.get("rubric_spec_tension"),
                    "round": args.round,
                }
            except Exception as exc:
                return {"judge": "gpt", "condition": cond,
                        "statement_id": sid, "scenario_idx": scen, "generator": gen,
                        "error": str(exc)[:300], "round": args.round}

        new_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.gpt_workers) as ex:
            futs = [ex.submit(process_gpt, w) for w in gpt_work]
            for fut in as_completed(futs):
                r = fut.result()
                new_rows.append(r)
                # Append immediately (in batches of 25) for crash safety.
                if len(new_rows) >= 25:
                    append_jsonl(new_rows, sync_out, sync_lock)
                    new_rows = []
                n_done += 1
                if n_done % 50 == 0 or n_done == len(gpt_work):
                    elapsed = time.time() - gpt_t0
                    print(f"  gpt progress: {n_done}/{len(gpt_work)} "
                          f"({100*n_done/max(1,len(gpt_work)):.0f}%, {elapsed:.0f}s)")
        append_jsonl(new_rows, sync_out, sync_lock)
        summary["gpt_wall_s"] = time.time() - gpt_t0
        summary["gpt_n_dispatched"] = len(gpt_work)

    # ============== Gemini 3 sync ==============
    if "gemini" in judges:
        gem = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"],
            vertexai=False,
        )
        gem_work = [w for w in work
                    if ("gemini", w[0], w[1], w[2], w[3]) not in already_done]
        print(f"\n--- Gemini sync judge: {len(gem_work)} items "
              f"(skipped {len(work) - len(gem_work)} already done) ---")
        gem_t0 = time.time()
        n_done = 0

        def process_gem(item):
            cond, sid, scen, gen, user_q, resp_text, stmt_text, rubric = item
            examples = get_examples(spec_by_id[sid])
            user_text = build_user_prompt(stmt_text, examples, rubric, user_q, resp_text)
            key = {"sid": sid, "scenario_idx": scen, "generator": gen,
                   "condition": cond, "round": args.round}
            try:
                data = call_gemini_json(log, gem, role=f"judge_gemini_{cond}",
                                        key=key, system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                                        user=user_text, max_tokens=1500)
                score = int(data["score"]) if data.get("score") is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
                return {
                    "judge": "gemini", "condition": cond,
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "score": score, "reasoning": data.get("reasoning"),
                    "spec_quotes": data.get("spec_quotes") or [],
                    "rubric_quotes": data.get("rubric_quotes") or [],
                    "example_refs": data.get("example_refs") or [],
                    "rubric_spec_tension": data.get("rubric_spec_tension"),
                    "round": args.round,
                }
            except Exception as exc:
                return {"judge": "gemini", "condition": cond,
                        "statement_id": sid, "scenario_idx": scen, "generator": gen,
                        "error": str(exc)[:300], "round": args.round}

        new_rows = []
        with ThreadPoolExecutor(max_workers=args.gemini_workers) as ex:
            futs = [ex.submit(process_gem, w) for w in gem_work]
            for fut in as_completed(futs):
                r = fut.result()
                new_rows.append(r)
                if len(new_rows) >= 25:
                    append_jsonl(new_rows, sync_out, sync_lock)
                    new_rows = []
                n_done += 1
                if n_done % 50 == 0 or n_done == len(gem_work):
                    elapsed = time.time() - gem_t0
                    print(f"  gemini progress: {n_done}/{len(gem_work)} "
                          f"({100*n_done/max(1,len(gem_work)):.0f}%, {elapsed:.0f}s)")
        append_jsonl(new_rows, sync_out, sync_lock)
        summary["gemini_wall_s"] = time.time() - gem_t0
        summary["gemini_n_dispatched"] = len(gem_work)

    # ============== Claude Sonnet 4.6 BATCH ==============
    batches_state: dict[str, Any] = {}
    if "claude" in judges:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        job_dir = Path("results/raw/e9_dart_iter_claude_batches") / f"round_{args.round}_{ts}"
        job_dir.mkdir(parents=True, exist_ok=True)

        for cond in conditions:
            cond_work = [w for w in work if w[0] == cond]
            if not cond_work:
                continue
            requests: list[dict[str, Any]] = []
            cells_meta: list[list[Any]] = []
            for item in cond_work:
                _cond, sid, scen, gen, user_q, resp_text, stmt_text, rubric = item
                examples = get_examples(spec_by_id[sid])
                user_text = build_user_prompt(stmt_text, examples, rubric, user_q, resp_text)
                cid = custom_id_for(sid, cond, scen, gen, args.round)
                requests.append(ba.build_request(
                    custom_id=cid,
                    model=ANTHROPIC_MODEL,
                    system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=600,
                    tools=[JUDGMENT_TOOL_1_5],
                    tool_choice={"type": "tool", "name": "submit_judgment"},
                    thinking={"type": "disabled"},
                    temperature=0,
                ))
                cells_meta.append([sid, cond, scen, gen])

            batch_name = f"dart_iter_round_{args.round}_{cond}"
            print(f"\n--- Claude batch submit cond={cond}: {len(requests)} requests ---")
            state = ba.submit(api_key, requests, job_dir, name=batch_name)
            print(f"  batch_id={state['batch_id']} (n={state['n_requests']})")
            id_map_path = job_dir / f"{batch_name}_custom_id_map.json"
            id_map_path.write_text(json.dumps(
                {r["custom_id"]: c for r, c in zip(requests, cells_meta)}, indent=2,
            ))
            batches_state[cond] = {
                "batch_id": state["batch_id"],
                "n_requests": state["n_requests"],
                "job_dir": str(job_dir),
                "name": batch_name,
                "custom_id_map": str(id_map_path),
                "model": ANTHROPIC_MODEL,
                "model_created_at": ANTHROPIC_MODEL_CREATED_AT,
                "submitted_at": time.time(),
            }

        summary["claude_batches"] = batches_state

        # Top-level pointer for Phase-4 fetch.
        batches_pointer.write_text(json.dumps({
            "round": args.round,
            "submitted_at": time.time(),
            "submitted_at_iso": ts,
            "conditions": conditions,
            "statements": target_sids,
            "job_dir": str(job_dir),
            "batches": batches_state,
        }, indent=2))
        print(f"\n  claude batches pointer: {batches_pointer}")

    summary["wall_s_total"] = time.time() - overall_t0
    print(f"\nDone in {summary['wall_s_total']:.0f}s")
    print(f"  sync output: {sync_out}")
    if "claude" in judges:
        print(f"  claude batches pointer: {batches_pointer}")
        print(f"  --> next: poll/fetch with the e9_phase4-style fetch script")
    return 0


if __name__ == "__main__":
    sys.exit(main())
