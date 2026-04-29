# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 variant of v2 cross-tier rubric writer (Together API, OpenAI-compat).

Same v2 system prompt as `write_cross_tier_rubrics_v2.py`. Different model:
`zai-org/GLM-5.1` via Together's OpenAI-compatible endpoint. No reasoning
toggle (open-weight chat model). Synchronous chat completions only.

Part of the multi-model rubric writer comparison matrix. GLM-5.1 was
identified in `.agents/logbooks/gpt5_correlation.md` as the best
open-weight surrogate for GPT-5.1 (Spearman 0.765 against GPT-5.1 on
judge tasks).

Usage:
    source .env && uv run --with openai python \\
        experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py [--smoke]

Env: TOGETHER_API_KEY in environment (loaded via `source .env`).
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

from openai import OpenAI

from write_cross_tier_rubrics_v2 import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cross_tier_records,
    load_spec,
    parse_json,
    validate_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("write_cross_tier_rubrics_v2_glm51")

WORKTREE = Path(__file__).resolve().parents[2]
OUTPUT_PATH = WORKTREE / "experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51.jsonl"

GLM51_MODEL = "zai-org/GLM-5.1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
MAX_TOKENS = 16000


def make_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("Missing TOGETHER_API_KEY in environment.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key)


def call_writer(
    client: OpenAI,
    user_prompt: str,
    candidate_idx: int,
    max_retries: int,
    temperature: float = 0.2,
) -> dict[str, Any]:
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=GLM51_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = resp.usage
            diag = {
                "candidate_idx": candidate_idx,
                "attempt": attempt,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
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
    raise RuntimeError(f"GLM-5.1 rubric writer v2 failed after retries: diag={last_diag} raw={last_content[:500]}")


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
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to the GLM-5.1 API (default 0.2; 0.0 for greedy).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run on first record only")
    args = parser.parse_args()

    spec = load_spec(args.spec_path)
    records = cross_tier_records(spec)
    if args.smoke:
        records = records[:1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("writing GLM-5.1 v2 rubrics for %d cross-tier records", len(records))
    client = make_client()

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
            "writer_model": GLM51_MODEL,
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
