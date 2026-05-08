"""ONE single API call: verify Anthropic tool-use schema enforcement.

Anthropic does not expose OpenAI-style `response_format: {json_schema, strict: true}`.
The supported way to force a strict-schema JSON output is:

  1. Define a `tools` entry whose `input_schema` matches your desired output.
  2. Set `tool_choice = {"type": "tool", "name": "<tool_name>"}`.
  3. Read the structured args from the response's `tool_use` block's `input`.

This script replays the exact failure case `be_thorough_but_efficient/phase_4#1`
(scenario_idx=0, generator=gemini-3-flash-preview) — where the original sync
call returned 4468 chars of JavaScript code instead of any JSON — and asserts
that the tool-use forcing produces a schema-conformant judgment.

Cost: 1 call. ~$0.005 estimated (≈6k input + ≤600 output tokens).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, load_jsonl
from e9_claude_judge import DIR, GEN_KEYS, build_request

# -------------------- structured-output schema --------------------

JUDGMENT_TOOL = {
    "name": "submit_judgment",
    "description": (
        "Submit a single rubric-grounded compliance judgment. Call this exactly "
        "once with all fields populated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Anchored 1-5 score per the rubric.",
            },
            "reasoning": {
                "type": "string",
                "description": "1-3 sentence justification grounded in the rubric anchor for the score.",
            },
            "spec_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verbatim quotes from the spec statement that justify the score.",
            },
            "example_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "References to spec example IDs (or empty list).",
            },
            "rubric_spec_tension": {
                "type": "boolean",
                "description": "Does the rubric appear to conflict with the spec text on this case?",
            },
            "tension_description": {
                "type": "string",
                "description": "If rubric_spec_tension is true, describe the conflict; otherwise empty string.",
            },
        },
        "required": [
            "score",
            "reasoning",
            "spec_quotes",
            "example_refs",
            "rubric_spec_tension",
            "tension_description",
        ],
    },
}


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set in environment; aborting.", file=sys.stderr)
        return 1

    sid = "be_thorough_but_efficient"
    target_scenario_idx = 0
    target_generator = "gemini-3-flash-preview"

    spec_by_id: dict[str, Any] = {}
    for line in SPEC_PATH.open():
        if line.strip():
            row = json.loads(line)
            spec_by_id[row["id"]] = row
    stmt = spec_by_id[sid]

    rubrics_by_id = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics.jsonl") if "error" not in r}
    rubric = rubrics_by_id[sid]

    responses = load_jsonl(DIR / "e8_responses.jsonl")
    target_user_q: str | None = None
    target_response_text: str | None = None
    col_for_gen = {label: col for label, col in GEN_KEYS}
    for r in responses:
        if r.get("statement_id") != sid or r.get("scenario_idx") != target_scenario_idx:
            continue
        target_user_q = r["user_query"]
        target_response_text = r.get(col_for_gen[target_generator])
        break
    if target_user_q is None or target_response_text is None:
        print(f"Could not locate scenario {target_scenario_idx} for {target_generator}", file=sys.stderr)
        return 1

    body = build_request(
        stmt=stmt,
        rubric=rubric,
        user_q=target_user_q,
        response_text=target_response_text,
        condition="phase_4",
        max_tokens=600,
    )
    body["tools"] = [JUDGMENT_TOOL]
    body["tool_choice"] = {"type": "tool", "name": "submit_judgment"}

    print("== TEST CASE ==")
    print(f"  statement: {sid}")
    print(f"  scenario_idx: {target_scenario_idx}, generator: {target_generator}")
    print(f"  user_query (first 120): {target_user_q[:120]!r}")
    print(f"  response_text length: {len(target_response_text)} chars")
    print(f"  model: {body['model']}")
    print(f"  max_tokens: {body['max_tokens']}")
    print(f"  thinking: {body['thinking']}")
    print(f"  tools: {[t['name'] for t in body['tools']]}")
    print(f"  tool_choice: {body['tool_choice']}")
    print()

    t0 = time.time()
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=120.0,
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        # Redact any key from response text just in case.
        snippet = resp.text[:600].replace(api_key, "<REDACTED>")
        print(f"HTTP {resp.status_code}: {snippet}")
        return 2

    rj = resp.json()
    print("== RESPONSE ==")
    print(f"  wall_time: {elapsed:.2f}s")
    print(f"  stop_reason: {rj.get('stop_reason')}")
    print(f"  usage: {rj.get('usage')}")
    print(f"  content blocks: {[b.get('type') for b in rj.get('content', [])]}")

    tool_use = next((b for b in rj.get("content", []) if b.get("type") == "tool_use"), None)
    if tool_use is None:
        print("\nFAIL: no tool_use block in response — tool_choice did not force the call.")
        print(json.dumps(rj, indent=2)[:2000])
        return 3

    args = tool_use.get("input", {})
    print("\n== STRUCTURED OUTPUT ==")
    print(json.dumps(args, indent=2))

    required = ["score", "reasoning", "spec_quotes", "example_refs", "rubric_spec_tension", "tension_description"]
    missing = [k for k in required if k not in args]
    if missing:
        print(f"\nFAIL: missing required fields: {missing}")
        return 4

    score = args.get("score")
    if not isinstance(score, int) or not (1 <= score <= 5):
        print(f"\nFAIL: score not int in [1,5]: {score!r}")
        return 5

    print("\nPASS: tool-use forcing produced schema-conformant JSON output on the failed-case prompt.")
    print("This confirms Anthropic's structured-outputs equivalent works for future judge runs.")

    out = Path("results/raw/e9_claude_structured_outputs_test")
    out.mkdir(parents=True, exist_ok=True)
    proof_path = out / f"phase4_1_{int(time.time())}.json"
    with proof_path.open("w") as f:
        json.dump(
            {
                "case": {"sid": sid, "scenario_idx": target_scenario_idx, "generator": target_generator},
                "request": body,
                "response": rj,
                "extracted": args,
                "wall_time_s": elapsed,
            },
            f,
            indent=2,
        )
    print(f"  proof saved to {proof_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
