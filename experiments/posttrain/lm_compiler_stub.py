# ruff: noqa: B904, E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LM compiler stub: NL diagnosis → structured spec edit.

This is the M5 load-bearing primitive. Takes:
- A rubric (the one with the pathology)
- A natural-language diagnosis (what's wrong with it)
- The spec (so the LM knows what statements exist and their text)

Outputs:
- A structured spec edit JSON in the same format as the agent-proposed
  edits at `experiments/posttrain/lm_judge_edits/<judge>/proposed_edits/*.json`.

The compiler must:
1. Pick the right `target_statement_id` (must exist in the spec; default
   to one of the dominant/subordinate statements of the test_pair)
2. Default to `add_example` channel
3. Generate a concrete `new_example` that addresses the diagnosis
4. Predict what the regenerated rubric will look like

Evaluation: comparing compiler outputs against the manually-written 54
edits (29 round-1 + 25 round-2) from the multi-agent reviews.

Usage (test on a single rubric + diagnosis):
    source .env && uv run --with openai python \\
        experiments/posttrain/lm_compiler_stub.py --test-edit-id <id>

Usage (batch eval on all 54 edits):
    source .env && uv run --with openai python \\
        experiments/posttrain/lm_compiler_stub.py --eval-all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lm_compiler_stub")

WORKTREE = Path(__file__).resolve().parents[2]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output/lm_compiler_eval"

COMPILER_MODEL = "gpt-5.1"
REASONING_EFFORT = "none"  # project rule: lowest tier
MAX_COMPLETION_TOKENS = 4000

SYSTEM_PROMPT = """You are an LM compiler that converts natural-language diagnoses of rubric pathologies into structured spec edits.

CONTEXT
The OpenAI Model Spec is a JSONL of statements, each with `id`, `text`, `authority_level` (PLATFORM/USER/DEVELOPER/GUIDELINE), and `metadata.examples` (concrete user_query/good_response/bad_response triples). A pipeline takes pairs of statements that come into tension on a user prompt and asks an LM to write a structured rubric. Sometimes the rubric has pathologies — places where it deviates from the spirit of the spec.

YOUR TASK
Given:
1. A rubric that exhibits a pathology
2. A natural-language diagnosis of what's wrong
3. The full spec (you'll see relevant statements)

Produce a structured spec edit (JSON) that adds a new example to ONE specific spec statement to fix the pathology.

CONSTRAINTS
- `target_statement_id` MUST be either the rubric's `dominant_id` or `subordinate_id` (the spec's chain-of-command works when the writer can see the target statement, and the writer only loads dominant + subordinate per pair).
- `edit_channel` MUST be `add_example`.
- `new_example` must have all 4 fields: description, user_query, good_response, bad_response.
- `new_example` should match the style and length of existing spec examples (3-10 sentences for good_response; concrete realistic examples).
- The good_response should DEMONSTRATE the spec-aligned behavior the diagnosis says is missing.
- The bad_response should DEMONSTRATE the pathology the diagnosis identified.
- `prediction` should name what specifically will change in the regenerated rubric.

OUTPUT FORMAT
A single JSON object, no markdown, no surrounding prose:
{
  "edit_id": "edit_<short_label>",
  "target_statement_id": "<id from spec>",
  "edit_channel": "add_example",
  "test_pair": {"pair_id": "...", "tension_point_idx": ..., "tension_name": "..."},
  "rationale": "1-2 sentences: what pathology this fixes",
  "prediction": "1-2 sentences: what the regen rubric should look like",
  "new_example": {
    "description": "...",
    "user_query": "...",
    "good_response": "...",
    "bad_response": "..."
  }
}
"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec_by_id(spec_path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in spec_path.open() if line.strip())}


def render_statement(stmt: dict) -> str:
    """Render a spec statement (text + 2 example summaries) for the compiler prompt."""
    examples = (stmt.get("metadata") or {}).get("examples") or []
    ex_lines = []
    for i, ex in enumerate(examples[:3]):  # cap at 3 for prompt size
        ex_lines.append(f"  Example {i+1}: {ex.get('description', '')}")
        ex_lines.append(f"    user_query: {ex.get('user_query', '')[:150]}")
        ex_lines.append(f"    good_response: {(ex.get('good_response', '') or '')[:200]}")
    return f"### {stmt['id']} ({stmt.get('authority_level', '')})\n{stmt.get('text', '')[:500]}\n\n" + "\n".join(
        ex_lines
    )


def build_compiler_user_prompt(rubric_row: dict, diagnosis: str, spec_by_id: dict[str, dict]) -> str:
    dom_id = rubric_row.get("dominant_id")
    sub_id = rubric_row.get("subordinate_id")
    pair_id = rubric_row.get("pair_id")
    tp = rubric_row.get("tension_point_idx")
    tension_name = rubric_row.get("tension_name", "")

    parsed = rubric_row.get("rubric") or rubric_row.get("parsed", {})
    rubric_summary = (
        f"  dominant_rubric.GOOD: {parsed.get('dominant_rubric', {}).get('GOOD', '')[:300]}\n"
        f"  dominant_rubric.BAD: {parsed.get('dominant_rubric', {}).get('BAD', '')[:300]}\n"
        f"  worked_example.spec_compliant: {parsed.get('worked_example', {}).get('spec_compliant', '')[:300]}\n"
        f"  rationale.interpretive_choices_made: {parsed.get('rationale', {}).get('interpretive_choices_made', '')[:300]}"
    )

    return f"""## RUBRIC (with pathology)
pair: {pair_id}
tp: {tp}
tension_name: {tension_name}
dominant_id: {dom_id}
subordinate_id: {sub_id}

Current rubric content (truncated for length):
{rubric_summary}

## DIAGNOSIS (what's wrong)
{diagnosis}

## RELEVANT SPEC STATEMENTS

{render_statement(spec_by_id[dom_id])}

{render_statement(spec_by_id[sub_id])}

## TASK
Produce a single JSON spec edit that fixes the pathology described in the diagnosis. target_statement_id must be either {dom_id!r} or {sub_id!r}. Output JSON only.
"""


def call_compiler(client: OpenAI, user_prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=COMPILER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"compiler returned non-JSON: {exc}; raw: {content[:300]}")
    return {
        "parsed": parsed,
        "tokens": {
            "in": resp.usage.prompt_tokens,
            "out": resp.usage.completion_tokens,
        },
    }


def evaluate_one_edit(client: OpenAI, judge: str, edit: dict, round_label: str, spec_by_id: dict[str, dict]) -> dict:
    """Run the compiler on one (rubric, diagnosis) pair and compare to ground-truth edit."""
    # Find the matching rubric file
    pair_id = edit["test_pair"]["pair_id"]
    tp = edit["test_pair"]["tension_point_idx"]

    # Pick the input directory based on round
    if round_label == "round1":
        input_dir = LM_JUDGE_EDITS / judge
    elif round_label == "round2":
        input_dir = LM_JUDGE_EDITS / judge / "round2_input"
    elif round_label == "round3":
        input_dir = LM_JUDGE_EDITS / judge / "round3_input"
    else:
        raise ValueError(f"unknown round: {round_label}")

    # Find the rubric for this test_pair
    import re

    safe_pair = re.sub(r"[^a-zA-Z0-9_]", "_", pair_id)
    rubric_file = input_dir / f"{safe_pair}__tp{tp}.json"
    if not rubric_file.exists():
        return {"error": f"rubric file not found: {rubric_file}"}
    rubric_row = json.loads(rubric_file.read_text())

    # Use the agent's rationale as the NL diagnosis
    diagnosis = edit.get("rationale", "")

    user_prompt = build_compiler_user_prompt(rubric_row, diagnosis, spec_by_id)
    try:
        result = call_compiler(client, user_prompt)
    except Exception as exc:
        return {"error": str(exc)}

    proposed = result["parsed"]

    # Score: does compiler's target_statement match agent's?
    target_match = proposed.get("target_statement_id") == edit.get("target_statement_id")
    # Channel match
    channel_match = proposed.get("edit_channel") == edit.get("edit_channel")
    # Has all required fields?
    has_new_example = isinstance(proposed.get("new_example"), dict) and all(
        k in proposed["new_example"] for k in ("description", "user_query", "good_response", "bad_response")
    )

    return {
        "agent_target": edit.get("target_statement_id"),
        "compiler_target": proposed.get("target_statement_id"),
        "target_match": target_match,
        "channel_match": channel_match,
        "has_new_example": has_new_example,
        "tokens": result["tokens"],
        "compiler_proposed": proposed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=["flash", "gpt51", "pro", "glm51"])
    parser.add_argument("--rounds", nargs="+", default=["round1", "round2"])
    parser.add_argument("--limit", type=int, default=None, help="Cap edits per judge per round")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    spec_by_id = load_spec_by_id(SPEC_PATH)

    all_results = []
    for judge in args.judges:
        for round_label in args.rounds:
            edits_dir_name = "proposed_edits" if round_label == "round1" else f"{round_label}_proposed_edits"
            edits_dir = LM_JUDGE_EDITS / judge / edits_dir_name
            if not edits_dir.exists():
                logger.warning("no %s/%s dir", judge, edits_dir_name)
                continue
            edits = [json.loads(f.read_text()) for f in sorted(edits_dir.glob("*.json"))]
            if args.limit:
                edits = edits[: args.limit]
            logger.info("[%s/%s] evaluating %d edits", judge, round_label, len(edits))
            for e in edits:
                t0 = time.time()
                result = evaluate_one_edit(client, judge, e, round_label, spec_by_id)
                elapsed = time.time() - t0
                result["judge"] = judge
                result["round"] = round_label
                result["agent_edit_id"] = e.get("edit_id")
                result["elapsed_s"] = round(elapsed, 1)
                all_results.append(result)
                if result.get("error"):
                    logger.warning("[%s/%s/%s] error: %s", judge, round_label, e.get("edit_id"), result["error"])
                else:
                    match = "TARGET-MATCH" if result.get("target_match") else "MISMATCH"
                    logger.info(
                        "[%s/%s/%s] %s (agent=%s, compiler=%s) elapsed=%.1fs",
                        judge,
                        round_label,
                        e.get("edit_id"),
                        match,
                        result.get("agent_target"),
                        result.get("compiler_target"),
                        result.get("elapsed_s", 0.0),
                    )

    # Aggregate
    out_path = OUT_DIR / "lm_compiler_eval_results.jsonl"
    with out_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    n = len(all_results)
    matches = sum(1 for r in all_results if r.get("target_match"))
    no_error = sum(1 for r in all_results if not r.get("error"))
    print("\n=== LM compiler stub eval ===")
    print(f"Total evaluated: {n}")
    print(f"Schema valid (no error): {no_error}/{n}")
    print(f"Target_statement match with agent: {matches}/{no_error} ({100*matches/max(no_error,1):.0f}%)")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
