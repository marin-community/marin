#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: verify `reasoning_effort="none"` fixes the GPT-5.1 length-cut bug.

Picks 5 items from the 2026-04-11 EXP-028g batch that came back with
`finish_reason="length"` and empty content (reasoning-budget exhausted),
re-submits them synchronously through chat.completions with
`reasoning_effort="none"` + `max_completion_tokens=4000` (matches
gpt-4.1's historical config), and prints per-item diagnostics.

Picks one length-cut item from each of the 5 worst-failing statements
(see per-statement failure table in the logbook) for diversity:
  - avoid_targeted_political_manipulation  (53% failure)
  - avoid_errors                            (50%)
  - assume_objective_pov                    (47%)
  - support_mental_health                   (41%)
  - express_uncertainty                     (35%)

Success criteria (all 5 must pass):
  1. `finish_reason == "stop"`
  2. `completion_tokens_details.reasoning_tokens == 0`
  3. Non-empty `content`
  4. Parseable JSON with an integer `score`

Usage:
    source .env && uv run python experiments/posttrain/smoke_gpt51_reasoning_none.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "lib" / "rigging" / "src"))

from marin.alignment.generate_prompts import load_spec  # noqa: E402
from marin.alignment.prompts.judge import (  # noqa: E402
    build_compliance_judge_prompt,
    build_judge_system_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "gpt51_batch"
TARGETS = [
    "sft",
    "full_dpo_beta01_b64_step1699",
    "lora_lr1e5_b64_step1699",
    "lora_lr5e6_b64_step1699",
]
SPEC_PATH = str(_REPO_ROOT / "experiments" / "posttrain" / "specs" / "openai_model_spec.jsonl")

WORST_STATEMENTS = [
    "avoid_targeted_political_manipulation",
    "avoid_errors",
    "assume_objective_pov",
    "support_mental_health",
    "express_uncertainty",
]

JUDGE_MODEL = "gpt-5.1"
REASONING_EFFORT = "none"
MAX_COMPLETION_TOKENS = 4000  # Match gpt-4.1 historical cap exactly.
TEMPERATURE = 0.0


def parse_judge_response(content: str) -> dict | None:
    """Minimal parser — returns a dict if JSON could be extracted, else None."""
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        json_str = json_match.group(0) if json_match else None
    if json_str is None:
        return None
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", json_str))
    except (json.JSONDecodeError, ValueError):
        return None


def find_length_cut_items() -> dict[str, tuple[str, str, dict]]:
    """Find one length-cut item per behavior_id in WORST_STATEMENTS."""
    picks: dict[str, tuple[str, str, dict]] = {}

    for target in TARGETS:
        if len(picks) == len(WORST_STATEMENTS):
            break

        manifest: dict[str, dict] = {}
        with (DATA_ROOT / target / "manifest.jsonl").open() as f:
            for line in f:
                rec = json.loads(line)
                manifest[rec["custom_id"]] = rec

        with (DATA_ROOT / target / "output.jsonl").open() as f:
            for line in f:
                entry = json.loads(line)
                body = entry.get("response", {}).get("body", {}) or {}
                choices = body.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason") != "length":
                    continue
                cid = entry["custom_id"]
                meta = manifest.get(cid)
                if not meta:
                    continue
                bid = meta["behavior_id"]
                if bid in WORST_STATEMENTS and bid not in picks:
                    picks[bid] = (target, cid, meta)
                    if len(picks) == len(WORST_STATEMENTS):
                        break

    return picks


def probe_one(
    client: OpenAI,
    statement,
    meta: dict,
) -> dict:
    """Make one chat.completions call with reasoning_effort='none'."""
    system_prompt = build_judge_system_prompt()
    user_prompt = build_compliance_judge_prompt(
        statement=statement,
        user_input=meta["user_message"],
        model_response=meta["response_text"],
        question_rubric=meta["rubric"],
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    usage = resp.usage
    ctd = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
    parsed = parse_judge_response(content)
    return {
        "finish_reason": choice.finish_reason,
        "content": content,
        "content_len": len(content),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "parsed": parsed,
    }


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run via `source .env && ...`.", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key)

    statements = load_spec(SPEC_PATH)
    picks = find_length_cut_items()
    missing = [s for s in WORST_STATEMENTS if s not in picks]
    if missing:
        print(f"WARNING: could not find length-cut items for: {missing}", file=sys.stderr)
    if not picks:
        print("ERROR: no length-cut items found — did EXP-028g already get retried?", file=sys.stderr)
        return 1

    print(f"Found {len(picks)} length-cut items from EXP-028g — re-testing with")
    print(f"  model={JUDGE_MODEL}")
    print(f"  reasoning_effort={REASONING_EFFORT!r}")
    print(f"  max_completion_tokens={MAX_COMPLETION_TOKENS}")
    print(f"  temperature={TEMPERATURE}")
    print()

    results: list[dict] = []
    for bid in WORST_STATEMENTS:
        if bid not in picks:
            continue
        target, cid, meta = picks[bid]
        statement = statements[bid]
        print(f"=== {bid}    target={target}    custom_id={cid}")
        print(f"    user_message (first 120 chars): {meta['user_message'][:120]!r}")
        try:
            result = probe_one(client, statement, meta)
        except Exception as e:
            print(f"    🔴 API call raised: {e}")
            results.append({"statement": bid, "error": str(e)})
            print()
            continue
        result["statement"] = bid
        result["custom_id"] = cid
        result["target"] = target
        results.append(result)

        fr = result["finish_reason"]
        rt = result["reasoning_tokens"]
        ct = result["completion_tokens"]
        parsed = result["parsed"]
        score = parsed.get("score") if parsed else None
        print(f"    finish_reason:     {fr}")
        print(f"    reasoning_tokens:  {rt}")
        print(f"    completion_tokens: {ct}")
        print(f"    content_len:       {result['content_len']}")
        if parsed is not None:
            explanation = parsed.get("explanation", "")
            print(f"    parsed score:     {score}")
            print(f"    parsed explanation (first 120 chars): {explanation[:120]!r}")
        else:
            print(f"    🔴 JSON PARSE FAILED. raw content (first 200 chars): {result['content'][:200]!r}")
        print()

    # Verdict.
    passed = []
    failed = []
    for r in results:
        if "error" in r:
            failed.append((r["statement"], f"API error: {r['error']}"))
            continue
        reasons = []
        if r["finish_reason"] != "stop":
            reasons.append(f"finish_reason={r['finish_reason']}")
        if r["reasoning_tokens"] != 0:
            reasons.append(f"reasoning_tokens={r['reasoning_tokens']}")
        if r["parsed"] is None:
            reasons.append("parse failed")
        elif not isinstance(r["parsed"].get("score"), int):
            reasons.append("parsed score not int")
        if reasons:
            failed.append((r["statement"], ", ".join(reasons)))
        else:
            passed.append(r["statement"])

    print("=" * 72)
    if failed:
        print(f"🔴 VERDICT: {len(passed)}/{len(results)} passed — FAILURES:")
        for stmt, reason in failed:
            print(f"    {stmt}: {reason}")
        return 1
    print(f"🟢 VERDICT: ALL {len(passed)}/{len(results)} items passed.")
    print("   Every item returned finish_reason='stop', reasoning_tokens=0,")
    print("   non-empty content, and a parseable JSON with an integer score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
