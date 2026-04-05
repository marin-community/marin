#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["click"]
# ///
"""Validate that package-level docs enable haiku to write correct code.

Reads a pre-generated sub-package doc and tests whether it gives haiku enough
signal to produce a correct fuzzy dedup script. A reviewer model scores the
output against a ground-truth rubric.

Usage:
    ./scripts/agent_docs/validation.py --output-dir /tmp/autodoc-validation
"""

import json
import logging
import re
from pathlib import Path

import click

from agent_docs.claude_cli import generate_json, strip_markdown_fences

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent

PACKAGE_DOC_PATH = REPO_ROOT / "docs" / "agent" / "packages" / "marin.processing.classification.deduplication.md"

ECOSYSTEM_OVERVIEW = """\
Marin is a data processing framework. Key libraries:
- marin: data processing pipelines (dedup, classification, tokenization)
- zephyr: lazy Dataset pipelines (Dataset.map/flat_map/group_by)
- dupekit: Rust FFI for fast hashing and MinHash/LSH
- fray: RPC actors and ResourceConfig
"""

TASK_PROMPT = """\
Write a Python script that performs fuzzy document deduplication on a set of \
Parquet files.

Requirements:
- Takes two CLI arguments: input glob pattern and output path.
- Uses Marin's built-in fuzzy dedup pipeline.
- Uses the correct default MinHash parameters.
- Complete and runnable with `if __name__ == "__main__"` and argparse.
- Write the script contents only — no explanation.

IMPORTANT: Use ONLY the documentation provided to write this script. \
Do not invent function names or parameters."""

# The review rubric embeds ground truth extracted from the package doc so the
# reviewer never has to guess whether an API exists.
REVIEW_RUBRIC = """\
Review this Python script for correctness against the Marin codebase.

Score each criterion 0 (wrong) or 1 (correct). Use ONLY the ground truth below
to make your judgments — do not guess about APIs.

## Ground Truth

Valid import:
  from marin.processing.classification.deduplication.fuzzy \
import dedup_fuzzy_document

Valid APIs in this package (exhaustive list):
  - dedup_fuzzy_document (fuzzy.py) — entry point for fuzzy document dedup
  - dedup_exact_document (exact.py) — entry point for exact document dedup
  - dedup_exact_paragraph (exact.py) — entry point for exact paragraph dedup
  - connected_components (connected_components.py) — graph CC algorithm
  - group_files (dedup_commons.py) — file bucketing utility
  - make_document_dedup_aggregator (dedup_commons.py) — group_by reducer factory
  - finalize_dedup (dedup_commons.py) — stats aggregation
  - DedupMode (dedup_commons.py) — StrEnum: EXACT_PARAGRAPH, EXACT_DOCUMENT, \
FUZZY_DOCUMENT
  - ResourceConfig — imported from fray, used for worker/coordinator resources

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

1. IMPORT_PATH: Is `dedup_fuzzy_document` imported from \
`marin.processing.classification.deduplication.fuzzy`? (1 if yes)
2. FUNCTION_NAME: Is the function called `dedup_fuzzy_document`? (1 if yes)
3. NUM_PERMS: Is `fuzzy_minhash_num_perms` 286 or omitted (default)? (1 if yes)
4. NUM_BANDS: Is `fuzzy_minhash_num_bands` 26 or omitted (default)? (1 if yes)
5. NGRAM_SIZE: Is `fuzzy_minhash_ngram_size` 5 or omitted (default)? (1 if yes)
6. SEED: Is `fuzzy_minhash_seed` 42 or omitted (default)? (1 if yes)
7. MAX_PARALLELISM: Is `max_parallelism` provided as an argument? (1 if yes)
8. KEYWORD_ONLY: Are arguments passed as keyword arguments? (1 if yes)
9. RUNNABLE: Has `if __name__ == "__main__"` and argparse? (1 if yes)
10. NO_HALLUCINATION: Does the script avoid inventing unlisted APIs? (1 if yes)

Output ONLY a JSON object with these 10 keys and 0/1 integer values, plus a \
"notes" string field. No other text.

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


def generate_script(context: str, model: str) -> dict:
    """Ask the coding agent to write a fuzzy dedup script."""
    prompt = (
        "You have access to the following documentation for the Marin monorepo.\n"
        "Use ONLY this documentation to write the script. "
        "Do not invent function names or parameters.\n\n"
        f"{context}\n\n---\n\n{TASK_PROMPT}\n"
    )
    data = generate_json(
        prompt,
        model=model,
        system_prompt=("You are writing Python scripts for the Marin monorepo. " "Output only code."),
    )
    script = strip_markdown_fences(data.get("result", ""))
    return {
        "script": script,
        "cost_usd": data.get("total_cost_usd", 0),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
    }


def review_script(script_content: str, model: str) -> dict:
    """Review the generated script against the ground-truth rubric."""
    prompt = REVIEW_RUBRIC.replace("__SCRIPT__", script_content)
    data = generate_json(
        prompt,
        model=model,
        system_prompt=("You are a senior engineer reviewing code. " "Output only the requested JSON."),
    )
    review_text = data.get("result", "")
    review_cost = data.get("total_cost_usd", 0)

    json_match = re.search(r"\{[^{}]+\}", review_text, re.DOTALL)
    if not json_match:
        return {
            "error": "Could not parse review JSON",
            "review_cost_usd": review_cost,
            "raw_review": review_text,
        }

    scores = json.loads(json_match.group())
    return {**scores, "review_cost_usd": review_cost, "raw_review": review_text}


def print_summary(
    scores: dict[str, int],
    gen_result: dict,
    review: dict,
    context_chars: int,
) -> None:
    """Print a clear summary table."""
    total = sum(scores.values())
    pct = 100 * total / len(CRITERIA)
    separator = "=" * 60

    print(f"\n{separator}")
    print("  Validation: Package-Level Docs -> Haiku -> Fuzzy Dedup")
    print(separator)
    print(f"  Context size:  {context_chars:,} chars")
    print(
        f"  Gen cost:      ${gen_result['cost_usd']:.4f}"
        f" ({gen_result['input_tokens']} in / {gen_result['output_tokens']} out)"
    )
    print(f"  Review cost:   ${review.get('review_cost_usd', 0):.4f}")
    print(f"  Total score:   {total}/{len(CRITERIA)} ({pct:.0f}%)")
    print("-" * 60)
    for criterion in CRITERIA:
        mark = "PASS" if scores.get(criterion, 0) else "FAIL"
        print(f"  [{mark}] {criterion}")
    print("-" * 60)
    notes = review.get("notes", "")
    if notes:
        print(f"  Notes: {notes}")
    print(f"{separator}\n")


@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for output artifacts.",
)
@click.option(
    "--gen-model",
    default="haiku",
    help="Model for code generation.",
)
@click.option(
    "--review-model",
    default="sonnet",
    help="Model for review.",
)
@click.option("--verbose", "-v", is_flag=True)
def main(output_dir: str, gen_model: str, review_model: str, verbose: bool) -> None:
    """Validate that package docs enable correct code generation."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not PACKAGE_DOC_PATH.exists():
        raise click.ClickException(f"Package doc not found at {PACKAGE_DOC_PATH} " "-- run the autodoc pipeline first")

    package_doc = PACKAGE_DOC_PATH.read_text()
    logger.info(
        "Read package doc: %d chars from %s",
        len(package_doc),
        PACKAGE_DOC_PATH,
    )

    context = f"{ECOSYSTEM_OVERVIEW}\n---\n\n" f"# marin.processing.classification.deduplication\n\n{package_doc}"
    context_chars = len(context)
    (out / "context.txt").write_text(context + "\n")
    logger.info("Total context: %d chars", context_chars)

    logger.info("Asking %s to write script...", gen_model)
    gen_result = generate_script(context, gen_model)
    (out / "dedup_script.py").write_text(gen_result["script"] + "\n")
    logger.info(
        "Gen cost: $%.4f (%d in / %d out)",
        gen_result["cost_usd"],
        gen_result["input_tokens"],
        gen_result["output_tokens"],
    )

    logger.info("Reviewing with %s...", review_model)
    review = review_script(gen_result["script"], review_model)

    scores = {c: review.get(c, 0) for c in CRITERIA}
    total_correct = sum(scores.values())

    result = {
        "package_doc": str(PACKAGE_DOC_PATH),
        "context_chars": context_chars,
        "gen_model": gen_model,
        "review_model": review_model,
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

    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")

    print_summary(scores, gen_result, review, context_chars)
    print("Generated script:")
    print("-" * 60)
    print(gen_result["script"])
    print("-" * 60)


if __name__ == "__main__":
    main()
