"""DART iterative validation — Round-N compile driver (calls all 3 compilers).

For each IMPROVING statement from prior round, builds a compiler prompt using
e9_dart_iter_compile.compile_for_statement (which inserts the cumulative-history
block), then dispatches sync calls to GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.6.

Outputs (analogous to dart_diagnoses*.jsonl from Run 1/2/3 but per round):
  - dart_iteration/dart_diagnoses_round_{N}.jsonl
  - dart_iteration/dart_diagnoses_gemini_round_{N}.jsonl
  - dart_iteration/dart_diagnoses_claude_round_{N}.jsonl

Usage:
    source .env && source .env2 && .venv/bin/python e9_dart_iter_round_n_compile.py --round 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import call_gpt_json
from e9_dart_compiler import COMPILER_SYSTEM, validate_diagnosis
from e9_dart_compiler_claude import call_claude_compiler, extract_tool_args
from e9_dart_compiler_gemini import call_gemini_json
from e9_dart_iter_compile import compile_for_statement
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"


def parse_anthropic_response(api_resp: dict) -> dict:
    """Wrap extract_tool_args to give a JSON-shaped diagnosis dict."""
    args = extract_tool_args(api_resp)
    return args or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--statements", default="")
    args = ap.parse_args()

    if args.round < 2:
        raise SystemExit("Round-N compile only meaningful for round >= 2")

    # Find IMPROVING statements from prior round
    prior_summary = ITER_DIR / f"round_{args.round - 1}_analysis_summary.json"
    if not prior_summary.exists():
        raise SystemExit(f"missing {prior_summary} — run analysis for round {args.round - 1} first")
    summary = json.loads(prior_summary.read_text())
    improving = [s for s, info in summary["per_statement"].items()
                 if info.get("verdict") == "improving"]
    if args.statements:
        wanted = set(args.statements.split(","))
        improving = [s for s in improving if s in wanted]
    print(f"Round {args.round} compile — eligible (IMPROVING): {len(improving)}")
    print(f"  {improving}\n")

    if not improving:
        print("No statements to compile. Exiting.")
        return 0

    # Build prompts
    prompts: dict[str, str] = {}
    for sid in improving:
        prompt, out_dir = compile_for_statement(sid, args.round)
        prompts[sid] = prompt
        (out_dir / "user_prompt.txt").write_text(prompt)
        print(f"  built prompt for {sid}: {len(prompt)} chars")

    # Initialize clients
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = genai.Client(api_key=(os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]))
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    log = RawAPILogger(f"e9_dart_iter_round_{args.round}_compile")
    print(f"\nraw log dir: {log.run_dir}\n")

    gpt_rows = []
    gem_rows = []
    cla_rows = []

    def call_one_gpt(sid: str):
        try:
            data = call_gpt_json(log, oai, role=f"dart_iter_round_{args.round}_compile_gpt",
                                 key={"statement_id": sid, "round": args.round},
                                 system=COMPILER_SYSTEM, user=prompts[sid], max_tokens=8000)
            return sid, data, None
        except Exception as e:
            return sid, None, str(e)[:300]

    def call_one_gem(sid: str):
        try:
            data = call_gemini_json(log, gem, role=f"dart_iter_round_{args.round}_compile_gem",
                                    key={"statement_id": sid, "round": args.round},
                                    system=COMPILER_SYSTEM, user=prompts[sid],
                                    thinking_budget=128)
            return sid, data, None
        except Exception as e:
            return sid, None, str(e)[:300]

    def call_one_cla(sid: str):
        try:
            api_resp = log.call(
                role=f"dart_iter_round_{args.round}_compile_cla",
                key={"statement_id": sid, "round": args.round},
                fn=lambda: call_claude_compiler(anthropic_key, system=COMPILER_SYSTEM, user=prompts[sid]),
            )
            data = parse_anthropic_response(api_resp)
            return sid, data, None
        except Exception as e:
            return sid, None, str(e)[:300]

    print("=== GPT-5.1 ===")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sid, data, err in (f.result() for f in [ex.submit(call_one_gpt, s) for s in improving]):
            if err:
                print(f"  {sid}: ERROR {err}")
                gpt_rows.append({"statement_id": sid, "error": err})
                continue
            ok, problems = validate_diagnosis(data, "", {})
            print(f"  {sid}: diag={data.get('diagnosis')} rec={data.get('recommendation')} "
                  f"rubric={len(data.get('rubric_edits') or [])} spec={len(data.get('spec_edits_for_author_review') or [])} "
                  f"({'ok' if ok else f'{len(problems)} warns'})")
            gpt_rows.append({
                "statement_id": sid,
                "round": args.round,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "diagnosis": data.get("diagnosis"),
                "evidence_summary": data.get("evidence_summary"),
                "rubric_edits": data.get("rubric_edits") or [],
                "spec_edits_for_author_review": data.get("spec_edits_for_author_review") or [],
                "recommendation": data.get("recommendation"),
                "validation_problems": problems,
            })

    print("\n=== Gemini 3 Pro ===")
    with ThreadPoolExecutor(max_workers=2) as ex:
        for sid, data, err in (f.result() for f in [ex.submit(call_one_gem, s) for s in improving]):
            if err:
                print(f"  {sid}: ERROR {err}")
                gem_rows.append({"statement_id": sid, "error": err})
                continue
            print(f"  {sid}: diag={data.get('diagnosis')} rec={data.get('recommendation')}")
            gem_rows.append({
                "statement_id": sid,
                "round": args.round,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "diagnosis": data.get("diagnosis"),
                "evidence_summary": data.get("evidence_summary"),
                "rubric_edits": data.get("rubric_edits") or [],
                "spec_edits_for_author_review": data.get("spec_edits_for_author_review") or [],
                "recommendation": data.get("recommendation"),
                "validation_problems": [],
            })

    print("\n=== Claude Sonnet 4.6 ===")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sid, data, err in (f.result() for f in [ex.submit(call_one_cla, s) for s in improving]):
            if err:
                print(f"  {sid}: ERROR {err}")
                cla_rows.append({"statement_id": sid, "error": err})
                continue
            print(f"  {sid}: diag={data.get('diagnosis')} rec={data.get('recommendation')}")
            cla_rows.append({
                "statement_id": sid,
                "round": args.round,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "diagnosis": data.get("diagnosis"),
                "evidence_summary": data.get("evidence_summary"),
                "rubric_edits": data.get("rubric_edits") or [],
                "spec_edits_for_author_review": data.get("spec_edits_for_author_review") or [],
                "recommendation": data.get("recommendation"),
                "validation_problems": [],
            })

    # Write outputs
    def write_jsonl(rows, path):
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write_jsonl(gpt_rows, ITER_DIR / f"dart_diagnoses_round_{args.round}.jsonl")
    write_jsonl(gem_rows, ITER_DIR / f"dart_diagnoses_gemini_round_{args.round}.jsonl")
    write_jsonl(cla_rows, ITER_DIR / f"dart_diagnoses_claude_round_{args.round}.jsonl")
    print(f"\nWrote dart_diagnoses_*_round_{args.round}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
