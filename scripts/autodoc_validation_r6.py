#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Round 6: Validate that package-level docs enable haiku to write correct code.

This is the formal validation of the package-based documentation system in
scripts/agent_docs/. Unlike R5, which varied MODULE_PROMPT across broad module
docs, R6 reads the already-generated sub-package doc and tests whether it gives
haiku enough signal to produce a correct fuzzy dedup script.

Key improvements over R5:
  - Sub-package granularity (marin.processing.classification.deduplication vs
    all of marin.processing with 40+ functions)
  - Reviewer gets explicit ground truth so it cannot hallucinate API judgments

Usage:
    ./scripts/autodoc_validation_r6.py --output-dir /tmp/autodoc-r6
"""

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

PACKAGE_DOC_PATH = REPO_ROOT / "docs" / "agent" / "packages" / "marin.processing.classification.deduplication.md"

ECOSYSTEM_OVERVIEW = """\
Marin is a data processing framework. Key libraries:
- marin: data processing pipelines (dedup, classification, tokenization)
- zephyr: lazy Dataset pipelines (Dataset.map/flat_map/group_by)
- dupekit: Rust FFI for fast hashing and MinHash/LSH
- fray: RPC actors and ResourceConfig
"""

TASK_PROMPT = """\
Write a Python script that performs fuzzy document deduplication on a set of Parquet files.

Requirements:
- Takes two CLI arguments: input glob pattern and output path.
- Uses Marin's built-in fuzzy dedup pipeline.
- Uses the correct default MinHash parameters.
- Complete and runnable with `if __name__ == "__main__"` and argparse.
- Write the script contents only — no explanation.

IMPORTANT: Use ONLY the documentation provided to write this script. Do not invent function names or parameters.\
"""

# The review rubric embeds ground truth extracted from the package doc so the
# reviewer never has to guess whether an API exists.
REVIEW_RUBRIC = """\
Review this Python script for correctness against the Marin codebase.

Score each criterion 0 (wrong) or 1 (correct). Use ONLY the ground truth below
to make your judgments — do not guess about APIs.

## Ground Truth

Valid import:
  from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

Valid APIs in this package (exhaustive list):
  - dedup_fuzzy_document (fuzzy.py) — entry point for fuzzy document dedup
  - dedup_exact_document (exact.py) — entry point for exact document dedup
  - dedup_exact_paragraph (exact.py) — entry point for exact paragraph dedup
  - connected_components (connected_components.py) — graph CC algorithm
  - group_files (dedup_commons.py) — file bucketing utility
  - make_document_dedup_aggregator (dedup_commons.py) — group_by reducer factory
  - finalize_dedup (dedup_commons.py) — stats aggregation
  - DedupMode (dedup_commons.py) — StrEnum with EXACT_PARAGRAPH, EXACT_DOCUMENT, FUZZY_DOCUMENT
  - ResourceConfig — imported from fray, used for worker_resources/coordinator_resources

dedup_fuzzy_document signature (all keyword-only):
  input_paths: str | list[str]
  output_path: str
  text_field: str = "text"
  filetypes: list[str] | None = None
  fuzzy_minhash_num_perms: int = 286
  fuzzy_minhash_num_bands: int = 26
  fuzzy_minhash_ngram_size: int = 5
  fuzzy_minhash_seed: int = 42
  max_parallelism: int          # REQUIRED, no default
  worker_resources: ResourceConfig | None = None
  coordinator_resources: ResourceConfig | None = None

## Criteria

1. IMPORT_PATH: Is `dedup_fuzzy_document` imported from `marin.processing.classification.deduplication.fuzzy`? (score 1 if yes)
2. FUNCTION_NAME: Is the function called `dedup_fuzzy_document`? (score 1 if yes)
3. NUM_PERMS: Is `fuzzy_minhash_num_perms` set to 286? (score 1 if 286 or if the default is used by omitting the arg)
4. NUM_BANDS: Is `fuzzy_minhash_num_bands` set to 26? (score 1 if 26 or if the default is used by omitting the arg)
5. NGRAM_SIZE: Is `fuzzy_minhash_ngram_size` set to 5? (score 1 if 5 or if the default is used by omitting the arg)
6. SEED: Is `fuzzy_minhash_seed` set to 42? (score 1 if 42 or if the default is used by omitting the arg)
7. MAX_PARALLELISM: Is `max_parallelism` provided as an argument? (score 1 if yes)
8. KEYWORD_ONLY: Are arguments passed as keyword arguments (name=value)? (score 1 if yes)
9. RUNNABLE: Has `if __name__ == "__main__"` and argparse? (score 1 if yes)
10. NO_HALLUCINATION: Does the script avoid inventing APIs not in the ground truth list above? (score 1 if no hallucinated APIs)

Output ONLY a JSON object with these 10 keys and 0/1 integer values, plus a "notes" string field. No other text.

## Script to review

```python
__SCRIPT__
```
"""

CRITERIA = [
    "IMPORT_PATH",
    "FUNCTION_NAME",
    "NUM_PERMS",
    "NUM_BANDS",
    "NGRAM_SIZE",
    "SEED",
    "MAX_PARALLELISM",
    "KEYWORD_ONLY",
    "RUNNABLE",
    "NO_HALLUCINATION",
]


def strip_markdown_fences(text: str) -> str:
    """Remove wrapping ```python ... ``` fences if present."""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def call_claude(prompt: str, *, model: str, system_prompt: str, budget: str = "0.50") -> dict:
    """Call claude CLI with --print --output-format json and return parsed response."""
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        budget,
        "--output-format",
        "json",
        "--system-prompt",
        system_prompt,
    ]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")
    return json.loads(result.stdout)


def generate_script(context: str, model: str) -> dict:
    """Ask the coding agent to write a fuzzy dedup script."""
    prompt = f"""\
You have access to the following documentation for the Marin monorepo.
Use ONLY this documentation to write the script. Do not invent function names or parameters.

{context}

---

{TASK_PROMPT}
"""
    data = call_claude(
        prompt,
        model=model,
        system_prompt="You are writing Python scripts for the Marin monorepo. Output only code.",
    )
    script = strip_markdown_fences(data.get("result", ""))
    return {
        "script": script,
        "cost_usd": data.get("total_cost_usd", 0),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
    }


def review_script(script_content: str, model: str) -> dict:
    """Review the generated script against ground-truth rubric."""
    prompt = REVIEW_RUBRIC.replace("__SCRIPT__", script_content)
    data = call_claude(
        prompt,
        model=model,
        system_prompt="You are a senior engineer reviewing code. Output only the requested JSON.",
    )
    review_text = data.get("result", "")
    review_cost = data.get("total_cost_usd", 0)

    json_match = re.search(r"\{[^{}]+\}", review_text, re.DOTALL)
    if json_match:
        scores = json.loads(json_match.group())
        return {**scores, "review_cost_usd": review_cost, "raw_review": review_text}

    return {"error": "Could not parse review JSON", "review_cost_usd": review_cost, "raw_review": review_text}


def print_summary(scores: dict[str, int], gen_result: dict, review: dict, context_chars: int) -> None:
    """Print a clear summary table."""
    total = sum(scores.values())
    print("\n" + "=" * 60)
    print("  R6 Validation: Package-Level Docs → Haiku → Fuzzy Dedup")
    print("=" * 60)
    print(f"  Context size:  {context_chars:,} chars")
    print(
        f"  Gen cost:      ${gen_result['cost_usd']:.4f} ({gen_result['input_tokens']} in / {gen_result['output_tokens']} out)"
    )
    print(f"  Review cost:   ${review.get('review_cost_usd', 0):.4f}")
    print(f"  Total score:   {total}/{len(CRITERIA)} ({100 * total / len(CRITERIA):.0f}%)")
    print("-" * 60)
    for c in CRITERIA:
        v = scores.get(c, 0)
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {c}")
    print("-" * 60)
    notes = review.get("notes", "")
    if notes:
        print(f"  Notes: {notes}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="R6 validation: package docs enable correct code generation")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output artifacts")
    parser.add_argument("--gen-model", type=str, default="haiku", help="Model for code generation")
    parser.add_argument("--review-model", type=str, default="sonnet", help="Model for review")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Read the pre-generated package doc
    if not PACKAGE_DOC_PATH.exists():
        logger.error("Package doc not found at %s — run the autodoc pipeline first", PACKAGE_DOC_PATH)
        raise SystemExit(1)

    package_doc = PACKAGE_DOC_PATH.read_text()
    logger.info("Read package doc: %d chars from %s", len(package_doc), PACKAGE_DOC_PATH)

    # Step 2: Build context
    context = f"{ECOSYSTEM_OVERVIEW}\n---\n\n# marin.processing.classification.deduplication\n\n{package_doc}"
    context_chars = len(context)
    (output_dir / "context.txt").write_text(context + "\n")
    logger.info("Total context: %d chars", context_chars)

    # Step 3: Generate script
    logger.info("Asking %s to write script...", args.gen_model)
    gen_result = generate_script(context, args.gen_model)
    (output_dir / "dedup_script.py").write_text(gen_result["script"] + "\n")
    logger.info(
        "Gen cost: $%.4f (%d in / %d out)",
        gen_result["cost_usd"],
        gen_result["input_tokens"],
        gen_result["output_tokens"],
    )

    # Step 4: Review
    logger.info("Reviewing with %s...", args.review_model)
    review = review_script(gen_result["script"], args.review_model)

    scores = {c: review.get(c, 0) for c in CRITERIA}
    total_correct = sum(scores.values())

    # Step 5: Save results
    result = {
        "round": 6,
        "package_doc": str(PACKAGE_DOC_PATH),
        "context_chars": context_chars,
        "gen_model": args.gen_model,
        "review_model": args.review_model,
        "generation_cost_usd": gen_result["cost_usd"],
        "review_cost_usd": review.get("review_cost_usd", 0),
        "total_cost_usd": gen_result["cost_usd"] + review.get("review_cost_usd", 0),
        "gen_input_tokens": gen_result["input_tokens"],
        "gen_output_tokens": gen_result["output_tokens"],
        "scores": scores,
        "total_correct": total_correct,
        "total_criteria": len(CRITERIA),
        "accuracy": total_correct / len(CRITERIA),
        "notes": review.get("notes", ""),
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n")

    # Step 6: Print summary
    print_summary(scores, gen_result, review, context_chars)

    # Also print the generated script
    print("Generated script:")
    print("-" * 60)
    print(gen_result["script"])
    print("-" * 60)


if __name__ == "__main__":
    main()
