# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate per-statement Stage-1 understanding.json files using gpt-4.1.

Mirrors the MARIN alignment pipeline's understanding stage (system + user
prompt templates from `lib/marin/src/marin/alignment/prompts/understanding.py`)
but uses gpt-4.1 as the generator so the output matches the judge model used
in Stage 1 pair classification.

Usage:
    source .env && \\
        uv run python experiments/posttrain/generate_understandings_gpt41.py \\
            --spec experiments/posttrain/specs/openai_model_spec.jsonl \\
            --output-dir experiments/posttrain/stage1_inputs/understandings_gpt41 \\
            --model gpt-4.1 \\
            --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from marin.alignment.prompts.understanding import (
    STANDARD_DEMOGRAPHIC_AXES,
    make_behavior_understanding_prompt,
    make_understanding_system_prompt,
)

logger = logging.getLogger("generate_understandings")


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_variation_axes(text: str) -> list[dict]:
    m = re.search(r"<variation_axes>(.*?)</variation_axes>", text, re.DOTALL)
    if not m:
        raise ValueError("missing <variation_axes> block")
    payload = m.group(1).strip()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("variation_axes must be a list")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("axis"):
            raise ValueError(f"variation_axes[{i}] missing 'axis'")
        spectrum = item.get("spectrum")
        if not isinstance(spectrum, list) or len(spectrum) < 2:
            raise ValueError(f"variation_axes[{i}] needs spectrum of length >=2")
    return data


def generate_one(
    client: OpenAI,
    model: str,
    statement_id: str,
    statement_text: str,
    max_retries: int,
) -> dict:
    system = make_understanding_system_prompt()
    user = make_behavior_understanding_prompt(statement_id, statement_text)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                max_tokens=4000,
            )
            content = resp.choices[0].message.content or ""
            behavior_understanding = extract_tag(content, "behavior_understanding")
            scientific_motivation = extract_tag(content, "scientific_motivation")
            if not behavior_understanding:
                raise ValueError("missing <behavior_understanding>")
            if not scientific_motivation:
                raise ValueError("missing <scientific_motivation>")
            behavior_specific_axes = parse_variation_axes(content)
            variation_axes = behavior_specific_axes + [dict(a) for a in STANDARD_DEMOGRAPHIC_AXES]
            return {
                "behavior_name": statement_id,
                "understanding": behavior_understanding,
                "scientific_motivation": scientific_motivation,
                "variation_axes": variation_axes,
                "model": model,
            }
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"{statement_id}: failed after {max_retries} attempts: {last_err}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="Overwrite existing understanding.json files")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set. Run with `source .env && ...`.")
        return 2

    statements: list[tuple[str, str]] = []
    with open(args.spec) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            statements.append((row["id"], row["text"]))
    logger.info("Loaded %d statements from %s", len(statements), args.spec)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, str]] = []
    for sid, stext in statements:
        out_path = args.output_dir / sid / "understanding.json"
        if out_path.exists() and not args.force:
            continue
        todo.append((sid, stext))
    logger.info("To generate: %d (skipping %d already present)", len(todo), len(statements) - len(todo))

    client = OpenAI()
    t_start = time.time()
    n_ok, n_err = 0, 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_stmt = {
            pool.submit(generate_one, client, args.model, sid, stext, args.max_retries): (sid, stext)
            for sid, stext in todo
        }
        for fut in as_completed(future_to_stmt):
            sid, _ = future_to_stmt[fut]
            try:
                record = fut.result()
                out_dir = args.output_dir / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "understanding.json", "w") as f:
                    json.dump(record, f, indent=2)
                n_ok += 1
                logger.info(
                    "OK %s  (axes=%d  understanding=%dch  motivation=%dch)",
                    sid,
                    len(record["variation_axes"]),
                    len(record["understanding"]),
                    len(record["scientific_motivation"]),
                )
            except Exception as e:
                n_err += 1
                logger.warning("FAIL %s: %s", sid, e)

    logger.info("done: ok=%d err=%d elapsed=%.1fs out=%s", n_ok, n_err, time.time() - t_start, args.output_dir)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
