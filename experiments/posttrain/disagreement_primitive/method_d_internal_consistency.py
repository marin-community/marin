# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Method D — statement-only vs examples-only rubric divergence (Tier 2 of SPEC AMBIGUITY EPIC).

For each statement with ≥2 examples, run two compiler passes per statement:
- Pass 1 (text-only): given just the statement TEXT, produce a per-statement rubric
  (GOOD/BAD criterion + KEY_TENSION).
- Pass 2 (examples-only): given just the statement's examples (description / user_query /
  good_response / bad_response per example), produce a per-statement rubric.

Then a downstream analyzer computes semantic equivalence between the two rubrics
per statement using the existing semantic-equivalence judge. Low equivalence ==
spec is internally inconsistent: prose and examples encode different operational rules.

Outputs `method_d_rubrics.jsonl`. Each row:
  {
    "statement_id": "...",
    "compiler_model": "...",
    "channel": "text_only" | "examples_only",
    "good_criterion": "...",
    "bad_criterion": "...",
    "key_tension": "...",
    "raw": "...",
    "elapsed_s": float,
  }

Project rule: GPT-5.1 reasoning_effort=none. Single compiler (GPT-5.1) for this
method since multi-compiler divergence is a separate Method (C). Free judges
(Gemini, GLM) are used by the analyzer for semantic equivalence.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("method_d")

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
OUTPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GPT_5_1_MODEL = "gpt-5.1"
DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 1500


SYSTEM_PROMPT = """You are an LM compiler producing a per-statement rubric for grading whether an assistant response satisfies a single spec statement. You will be given EITHER the statement text only, OR the statement's worked examples only — not both.

Your job: write a GOOD criterion and BAD criterion that an external judge could use to score responses on this statement.

OUTPUT FORMAT (single JSON, no markdown):
{
  "good_criterion": "<2-3 sentences describing what a response that clearly satisfies this statement would look like operationally>",
  "bad_criterion": "<2-3 sentences describing what a response that clearly violates this statement would look like operationally>",
  "key_tension": "<1-2 sentences naming the central interpretive choice or borderline behavior the rubric makes>"
}

REQUIREMENTS
- Be operationally specific. "Be respectful" is not operational; "uses non-condescending phrasing and does not lecture the user about their choices" is.
- The good and bad criteria should be mirror images, not loose parallels.
- key_tension should name the SOFTEST predicate or judgment call in your rubric.
- No commentary outside the JSON.
"""


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def load_spec(path: Path = SPEC_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def build_text_only_prompt(statement: dict[str, Any]) -> str:
    return (
        f"STATEMENT_ID: {statement['id']}\n\n"
        f"STATEMENT TEXT (no examples shown):\n{statement['text']}\n\n"
        "Produce the rubric per the schema."
    )


def build_examples_only_prompt(statement: dict[str, Any]) -> str:
    examples = (statement.get("metadata") or {}).get("examples") or []
    parts = [f"STATEMENT_ID: {statement['id']}\n", "EXAMPLES (statement text NOT shown):\n"]
    for idx, ex in enumerate(examples, 1):
        parts.append(f"Example {idx}:")
        parts.append(f"  description: {ex.get('description', '')}")
        parts.append(f"  user_query: {ex.get('user_query', '')}")
        parts.append(f"  good_response: {ex.get('good_response', '')}")
        parts.append(f"  bad_response: {ex.get('bad_response', '')}")
        parts.append("")
    parts.append("Infer the operational rule these examples encode and produce the rubric per the schema.")
    return "\n".join(parts)


def make_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _validate(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    return "good_criterion" in parsed and "bad_criterion" in parsed


def call_compiler(client: OpenAI, user_prompt: str, max_retries: int = 2) -> dict[str, Any]:
    last_err = ""
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=GPT_5_1_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=DEFAULT_TEMPERATURE,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            last_content = content
            parsed = parse_json(content) if content else None
            if not _validate(parsed):
                last_err = f"invalid; keys={list(parsed.keys()) if isinstance(parsed, dict) else None}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "raw": content, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"compiler failed: {last_err}; last={last_content[:300]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "method_d_rubrics.jsonl")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--min-examples", type=int, default=2, help="Skip statements with fewer examples.")
    args = parser.parse_args()

    spec = load_spec()
    if args.limit:
        spec = spec[: args.limit]
    spec_with_examples = [s for s in spec if len((s.get("metadata") or {}).get("examples") or []) >= args.min_examples]
    skipped = len(spec) - len(spec_with_examples)
    logger.info("spec: %d statements; %d have ≥%d examples; %d skipped",
                len(spec), len(spec_with_examples), args.min_examples, skipped)

    client = make_openai_client()
    work: list[tuple[dict[str, Any], str]] = []
    for stmt in spec_with_examples:
        work.append((stmt, "text_only"))
        work.append((stmt, "examples_only"))
    logger.info("work units: %d", len(work))

    rows: list[dict[str, Any]] = []
    failures: list[Any] = []

    def worker(idx: int) -> dict[str, Any] | None:
        stmt, channel = work[idx]
        try:
            if channel == "text_only":
                prompt = build_text_only_prompt(stmt)
            else:
                prompt = build_examples_only_prompt(stmt)
            result = call_compiler(client, prompt)
        except Exception as exc:
            failures.append((stmt["id"], channel, str(exc)))
            return None
        parsed = result["parsed"]
        return {
            "statement_id": stmt["id"],
            "compiler_model": GPT_5_1_MODEL,
            "channel": channel,
            "good_criterion": parsed.get("good_criterion", ""),
            "bad_criterion": parsed.get("bad_criterion", ""),
            "key_tension": parsed.get("key_tension", ""),
            "raw": result["raw"],
            "elapsed_s": result["elapsed_s"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(worker, i) for i in range(len(work))]
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                logger.info("ok stmt=%s channel=%s", row["statement_id"], row["channel"])

    logger.info("wrote %d rows to %s", len(rows), args.output)
    if failures:
        logger.error("FAILURES (%d): %s", len(failures), failures[:3])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
