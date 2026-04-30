# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPT-5.1 variant of v2 cross-tier rubric writer.

Same v2 system prompt (rationale field, all spec examples, no topic-specific
REQUIREMENTS) as `write_cross_tier_rubrics_v2.py`. Different model + API:
GPT-5.1 with `reasoning_effort="minimal"` (lowest tier per project rule).
Synchronous chat completions API only.

Part of the multi-model rubric writer comparison matrix.

Usage:
    source .env && uv run --with openai python \\
        experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

# Reuse v2 components — same system prompt, schema, sample selection
from write_cross_tier_rubrics_v2 import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cross_tier_records,
    load_spec,
    parse_json,
    validate_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("write_cross_tier_rubrics_v2_gpt51")

WORKTREE = Path(__file__).resolve().parents[2]
OUTPUT_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51.jsonl"

GPT51_MODEL = "gpt-5.1"
REASONING_EFFORT = "none"
MAX_COMPLETION_TOKENS = 8000
TEMPERATURE = 0.2  # match flash/pro/glm51 (was previously unset → API default ≈1.0)


def call_writer(
    client: OpenAI,
    user_prompt: str,
    candidate_idx: int,
    max_retries: int,
    temperature: float = TEMPERATURE,
) -> dict[str, Any]:
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=GPT51_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = resp.usage
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            diag = {
                "candidate_idx": candidate_idx,
                "attempt": attempt,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": reasoning_tokens,
                "elapsed_s": round(time.time() - t0, 2),
                "raw_content_len": len(content),
            }
            last_diag = diag
            last_content = content
            try:
                parsed = parse_json(content) if content else None
                diag["parse_ok"] = parsed is not None
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
    raise RuntimeError(f"GPT-5.1 rubric writer v2 failed after retries: diag={last_diag} raw={last_content[:500]}")


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
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE, help="Sampling temperature (default 0.2; 0.0 for greedy)."
    )
    parser.add_argument("--smoke", action="store_true", help="Run on first record only")
    args = parser.parse_args()

    spec = load_spec(args.spec_path)
    records = cross_tier_records(spec)
    if args.smoke:
        records = records[:1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("writing GPT-5.1 v2 rubrics for %d cross-tier records", len(records))
    client = OpenAI()

    prompts = [build_user_prompt(record, spec, include_cross_cutting=args.cross_cutting) for record in records]
    results: dict[int, dict[str, Any]] = {}

    def worker(index: int) -> tuple[int, dict[str, Any]]:
        prompt, dominant_id, subordinate_id = prompts[index]
        result = call_writer(client, prompt, candidate_idx=0, max_retries=args.max_retries, temperature=args.temperature)
        record = records[index]
        return index, {
            "pair_id": record["pair_id"],
            "tension_point_idx": record["tension_point_idx"],
            "tension_name": record["tension_point"].get("tension_name", ""),
            "dominant_id": dominant_id,
            "subordinate_id": subordinate_id,
            "candidate_idx": 0,
            "writer_model": GPT51_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": TEMPERATURE,
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
