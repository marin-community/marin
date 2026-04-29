# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini 3 Pro variant of v2 cross-tier rubric writer.

Same v2 system prompt as `write_cross_tier_rubrics_v2.py`, switched model:
`gemini-3-pro-preview` with `thinking_budget=0`. Synchronous generate_content
API only.

Part of the multi-model rubric writer comparison matrix. The prior
"no Pro at any stage" rule was lifted on 2026-04-26 evening (had been
scoped to a Codex overnight run only).

Usage:
    source .env && uv run --with google-genai python \\
        experiments/posttrain/write_cross_tier_rubrics_v2_pro.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

# Reuse v2 components — same system prompt, schema, sample selection
from write_cross_tier_rubrics_v2 import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cross_tier_records,
    load_spec,
    parse_json,
    response_text,
    usage_dict,
    validate_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("write_cross_tier_rubrics_v2_pro")

WORKTREE = Path(__file__).resolve().parents[2]
OUTPUT_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro.jsonl"

GEMINI_PRO_MODEL = "gemini-3-pro-preview"
# Gemini 3 Pro rejects thinking_budget=0 ("This model only works in thinking mode").
# Per Google's API, Pro's minimum is 128. Use the minimum to honor the project's
# "lowest reasoning tier" rule. This is the lowest allowed for Pro, not a value
# we'd want for production rubric writing.
THINKING_BUDGET = 128
MAX_OUTPUT_TOKENS = 8000


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return key


def make_client() -> genai.Client:
    return genai.Client(api_key=api_key(), vertexai=False)


def call_writer(
    client: genai.Client,
    user_prompt: str,
    candidate_idx: int,
    max_retries: int,
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        response_mime_type="application/json",
    )
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=user_prompt,
                config=config,
            )
            content = response_text(response)
            diag = {
                "candidate_idx": candidate_idx,
                "attempt": attempt,
                "elapsed_s": round(time.time() - t0, 2),
                "raw_content_len": len(content),
                "usage": usage_dict(response),
            }
            last_diag = diag
            last_content = content
            try:
                parsed = parse_json(content)
                diag["parse_ok"] = True
            except Exception as exc:
                parsed = None
                diag["parse_ok"] = False
                diag["parse_error"] = str(exc)
            schema_ok, missing = validate_schema(parsed)
            diag["schema_ok"] = schema_ok
            if missing:
                diag["schema_missing"] = missing
            if diag["parse_ok"] and schema_ok:
                return {"diag": diag, "parsed": parsed, "raw_content": None}
            logger.warning(
                "schema attempt failed candidate=%d attempt=%d diag=%s",
                candidate_idx,
                attempt,
                diag,
            )
        except Exception as exc:
            last_diag = {"attempt": attempt, "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}
            logger.warning("call attempt %d threw: %s", attempt + 1, exc)
        time.sleep(1 + attempt)
    raise RuntimeError(f"Gemini Pro rubric writer v2 failed after retries: diag={last_diag} raw={last_content[:500]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--spec-path", type=Path, default=None, help="Override default spec path (for forked-spec experiments)."
    )
    parser.add_argument(
        "--cross-cutting",
        action="store_true",
        help="Include 4 cross-cutting always-on statements (Option B / v3 architecture).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Run on first record only")
    args = parser.parse_args()

    spec = load_spec(args.spec_path)
    records = cross_tier_records(spec)
    if args.smoke:
        records = records[:1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("writing Gemini Pro v2 rubrics for %d cross-tier records", len(records))
    client = make_client()

    prompts = [build_user_prompt(record, spec, include_cross_cutting=args.cross_cutting) for record in records]
    results: dict[int, dict[str, Any]] = {}

    def worker(index: int) -> tuple[int, dict[str, Any]]:
        prompt, dominant_id, subordinate_id = prompts[index]
        result = call_writer(client, prompt, candidate_idx=0, max_retries=args.max_retries)
        record = records[index]
        return index, {
            "pair_id": record["pair_id"],
            "tension_point_idx": record["tension_point_idx"],
            "tension_name": record["tension_point"].get("tension_name", ""),
            "dominant_id": dominant_id,
            "subordinate_id": subordinate_id,
            "candidate_idx": 0,
            "writer_model": GEMINI_PRO_MODEL,
            "thinking_budget": THINKING_BUDGET,
            "writer_version": "v2",
            "diag": result["diag"],
            "parsed": result["parsed"],
            "raw_content": result["raw_content"],
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(worker, index) for index in range(len(records))]
        for future in as_completed(futures):
            index, row = future.result()
            results[index] = row
            d = row["diag"]
            r_count = len(row["parsed"].get("rationale", {}).get("spec_clauses_anchored_on", []))
            logger.info(
                "[%s tp=%d] schema_ok=%s rationale_clauses=%d elapsed=%.1fs",
                row["pair_id"],
                row["tension_point_idx"],
                d["schema_ok"],
                r_count,
                d["elapsed_s"],
            )

    rows = [results[index] for index in range(len(records))]
    args.output.write_text("\n".join(json.dumps(row) for row in rows))
    logger.info("wrote %d rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
