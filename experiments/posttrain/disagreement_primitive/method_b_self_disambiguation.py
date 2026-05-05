# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Method B — self-disambiguation introspection (Tier 1 of SPEC AMBIGUITY EPIC).

For each spec statement, ask 3 judges to enumerate 2-3 distinct, plausible
readings of the statement language. If the language is operationally
clear, judges should converge on a single reading; if ambiguous, judges
will name multiple distinct readings.

Per the project rule, all judges run no-reasoning (GPT-5.1
reasoning_effort=none, Gemini thinking_budget=0, GLM no toggle).

Outputs `method_b_readings.jsonl`. Each row:
  {
    "statement_id": "...",
    "judge_model": "...",
    "readings": [
      {"description": "...", "example_satisfying_only_this_reading": "..."},
      ...
    ],
    "raw_content": "...",
    "elapsed_s": float,
  }

Downstream analyzer (`analyze_ambiguity.py`, separate script) does the
semantic clustering across the per-statement reading sets to produce the
`language_ambiguous` / `language_robust` typed label.
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
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("method_b")

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
OUTPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GPT_5_1_MODEL = "gpt-5.1"
GLM_5_1_MODEL = "zai-org/GLM-5.1"
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 1500


SYSTEM_PROMPT = """You are a careful spec analyst. You will be shown ONE statement from a model spec — just the statement text, with NO examples and NO surrounding context.

Your job: enumerate distinct, plausible readings of the statement that a careful reader could defend. The goal is to surface where the LANGUAGE of the statement admits multiple operational interpretations.

A "reading" is an operational interpretation: what would a model that adopts this reading actually do?

INSTRUCTIONS
- List exactly 2 or 3 distinct readings. If the statement is operationally clear (only one reading is defensible), list ONE reading and set "single_reading_defensible": true.
- For each reading: a 1-sentence description of what it operationally means, and a short concrete example response (1-2 sentences) that would satisfy ONLY that reading and not the others.
- Make the readings genuinely distinct in operational behavior — not paraphrases of each other. If you can't find 2 genuinely distinct readings, set "single_reading_defensible": true.
- Do not consider what the spec author "probably meant" — read the language literally and ask what reasonable interpretations it admits.
- Do not invent unreasonable readings just to fill the slots.

OUTPUT FORMAT
A single JSON object, no markdown:
{
  "single_reading_defensible": <bool>,
  "readings": [
    {
      "description": "<one-sentence operational description>",
      "example_satisfying_only_this_reading": "<concrete example response, 1-2 sentences>"
    },
    ...
  ],
  "ambiguous_phrases": ["<verbatim phrase from the statement that contributes to the ambiguity>", ...],
  "reasoning": "<one paragraph: why these are the readings you chose, what specific words drove the divergence>"
}

REQUIREMENTS
- If single_reading_defensible is true, "readings" has exactly 1 entry and "ambiguous_phrases" can be empty.
- If single_reading_defensible is false, "readings" has 2 or 3 entries and "ambiguous_phrases" should name the specific words/phrases driving the ambiguity.
- "ambiguous_phrases" entries must be verbatim substrings of the statement text.
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


def build_user_prompt(statement: dict[str, Any]) -> str:
    return (
        f"STATEMENT_ID: {statement['id']}\n"
        f"SECTION: {statement.get('section', '')} / {statement.get('subsection', '')}\n\n"
        f"STATEMENT TEXT:\n{statement['text']}\n\n"
        "Now enumerate the readings per the schema."
    )


def make_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=key)


def make_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("Missing TOGETHER_API_KEY in environment.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key)


def make_gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return genai.Client(api_key=key, vertexai=False)


def _validate(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "single_reading_defensible" not in parsed or "readings" not in parsed:
        return False
    if not isinstance(parsed["readings"], list) or len(parsed["readings"]) < 1:
        return False
    return True


def call_openai(client: OpenAI, user_prompt: str, max_retries: int = 2) -> dict[str, Any]:
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
                last_err = f"validation failed; keys={list(parsed.keys()) if isinstance(parsed, dict) else None}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "raw": content, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("openai attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"openai failed: {last_err}; last={last_content[:300]}")


def call_together(client: OpenAI, user_prompt: str, max_retries: int = 2) -> dict[str, Any]:
    last_err = ""
    last_content = ""
    fortified = (
        user_prompt
        + "\n\nIMPORTANT: Reply with a single JSON object exactly matching the schema in the system message. "
        "Start your reply with `{` and end with `}`. Do NOT write any text outside the JSON."
    )
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            kwargs: dict[str, Any] = dict(
                model=GLM_5_1_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fortified},
                ],
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            try:
                kwargs_with_json = dict(kwargs)
                kwargs_with_json["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs_with_json)
            except Exception:
                resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            last_content = content
            parsed = parse_json(content) if content else None
            if not _validate(parsed):
                last_err = f"validation failed; keys={list(parsed.keys()) if isinstance(parsed, dict) else None}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "raw": content, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("together attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"together failed: {last_err}; last={last_content[:300]}")


def call_gemini(client: genai.Client, user_prompt: str, max_retries: int = 2) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    last_err = ""
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=user_prompt, config=config)
            content = ""
            try:
                content = resp.text or ""
            except (AttributeError, ValueError):
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    parts = getattr(cands[0].content, "parts", []) or []
                    content = "".join(str(getattr(p, "text", "")) for p in parts if getattr(p, "text", None))
            last_content = content
            parsed = parse_json(content) if content else None
            if not _validate(parsed):
                last_err = f"validation failed; content_len={len(content)}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "raw": content, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("gemini attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"gemini failed: {last_err}; last={last_content[:300]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap on statements; useful for smoke-tests.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "method_b_readings.jsonl")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--judges", nargs="+", default=["gemini", "glm", "gpt"], help="Subset of {gemini, glm, gpt}.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip (statement, judge) pairs already in output.")
    args = parser.parse_args()

    spec = load_spec()
    if args.limit:
        spec = spec[: args.limit]
    logger.info("spec: %d statements", len(spec))

    existing: set[tuple[str, str]] = set()
    if args.skip_existing and args.output.exists():
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add((row["statement_id"], row["judge_model"]))
        logger.info("found %d existing (statement, judge) entries", len(existing))

    openai_client = make_openai_client() if "gpt" in args.judges else None
    together_client = make_together_client() if "glm" in args.judges else None
    gemini_client = make_gemini_client() if "gemini" in args.judges else None

    work: list[tuple[dict[str, Any], str]] = []
    for stmt in spec:
        for judge in args.judges:
            judge_model = {"gpt": GPT_5_1_MODEL, "glm": GLM_5_1_MODEL, "gemini": GEMINI_FLASH_MODEL}[judge]
            if (stmt["id"], judge_model) in existing:
                continue
            work.append((stmt, judge))
    logger.info("work units: %d", len(work))

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    def worker(idx: int) -> dict[str, Any] | None:
        stmt, judge = work[idx]
        prompt = build_user_prompt(stmt)
        try:
            if judge == "gpt":
                result = call_openai(openai_client, prompt)
                model = GPT_5_1_MODEL
            elif judge == "glm":
                result = call_together(together_client, prompt)
                model = GLM_5_1_MODEL
            elif judge == "gemini":
                result = call_gemini(gemini_client, prompt)
                model = GEMINI_FLASH_MODEL
            else:
                raise ValueError(f"unknown judge {judge}")
        except Exception as exc:
            failures.append((stmt["id"], judge, str(exc)))
            return None
        parsed = result["parsed"]
        return {
            "statement_id": stmt["id"],
            "judge_model": model,
            "single_reading_defensible": parsed.get("single_reading_defensible"),
            "readings": parsed.get("readings"),
            "ambiguous_phrases": parsed.get("ambiguous_phrases", []),
            "reasoning": parsed.get("reasoning", ""),
            "raw": result["raw"],
            "elapsed_s": result["elapsed_s"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.skip_existing and args.output.exists() else "w"
    with args.output.open(write_mode, encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(worker, i) for i in range(len(work))]
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                logger.info(
                    "ok stmt=%s judge=%s readings=%d single=%s",
                    row["statement_id"],
                    row["judge_model"],
                    len(row["readings"] or []),
                    row["single_reading_defensible"],
                )

    logger.info("wrote %d rows to %s", len(rows), args.output)
    if failures:
        logger.error("FAILURES (%d): %s", len(failures), failures[:5])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
