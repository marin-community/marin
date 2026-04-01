#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Round 3 autodoc variation experiments.

Changes from round 2:
- Docs organized by folder/package, not by topic
- Examples show exact dedup (different from the task: fuzzy dedup)
- Coding agent is haiku (weaker model)
- Reviewer is sonnet
- 5 variations on semantic compression approach

Usage:
    uv run scripts/autodoc_variations_r3.py --output-dir /tmp/autodoc-r3
    uv run scripts/autodoc_variations_r3.py --variation 1 --output-dir /tmp/autodoc-r3
    uv run scripts/autodoc_variations_r3.py --report-only --output-dir /tmp/autodoc-r3
"""

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Ecosystem overview — same for all variations
# ---------------------------------------------------------------------------
ECOSYSTEM_OVERVIEW = """\
# Marin Ecosystem

Marin is a monorepo for training and curating large language models.

| Library | Purpose |
|---------|---------|
| **marin** | Data processing: classification, deduplication, decontamination, tokenization |
| **zephyr** | Lazy dataset pipelines (`Dataset.map/flat_map/group_by`), executed by `ZephyrContext` |
| **dupekit** | Rust FFI (PyO3): fast hashing (xxHash3, BLAKE3) and MinHash/LSH primitives |
| **iris** | Job orchestration: controller, autoscaling, gRPC workers |
| **fray** | RPC actors, `ResourceConfig(cpu, ram, disk)` for worker sizing |
| **rigging** | Storage abstraction (GCS, S3, local) via fsspec |
| **levanter** | LLM training (JAX): Llama, Gemma, Qwen with distributed training |
| **haliax** | Named tensors for JAX |

Dependency direction: `{iris, haliax}` → `{levanter, zephyr}` → `marin`.
"""

# ---------------------------------------------------------------------------
# V1: Package-level doc with full semantic overview, all functions documented
# inline, example shows exact dedup (not fuzzy)
# ---------------------------------------------------------------------------
V1_DOC = """\
# `marin.processing.classification.deduplication` — Package Reference

## Overview

This package removes duplicate content from large text corpora at two granularities
(document and paragraph) using two strategies (exact hash matching and fuzzy MinHash/LSH).
All functions produce vortex attribute files marking duplicates, leaving the original
data untouched. Internally they build Zephyr `Dataset` pipelines and execute them via
`ZephyrContext`.

The package has four modules:
- `exact.py` — Exact dedup via full-text xxHash3-128 hashing. Two entry points:
  `dedup_exact_document` and `dedup_exact_paragraph`.
- `fuzzy.py` — Fuzzy dedup via MinHash signatures + LSH banding. Entry point:
  `dedup_fuzzy_document`. Uses `dupekit` Rust FFI for the heavy computation.
- `connected_components.py` — Hash-to-Min connected-components algorithm over
  LSH bucket co-occurrences. Called internally by fuzzy dedup.
- `dedup_commons.py` — Shared helpers: file collection, wandb init, vortex writing,
  counter aggregation. Not called directly.

## `exact.py`

### `dedup_exact_document(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Exact document dedup. Hashes full document text with xxHash3-128, groups by hash,
marks all but one copy as duplicates. Returns counter dict.
Import: `from marin.processing.classification.deduplication.exact import dedup_exact_document`

### `dedup_exact_paragraph(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Paragraph-level exact dedup. Splits documents into paragraphs, hashes each, marks
duplicate paragraph spans. Returns counter dict.
Import: `from marin.processing.classification.deduplication.exact import dedup_exact_paragraph`

## `fuzzy.py`

### `dedup_fuzzy_document(*, input_paths, output_path, text_field="text", filetypes=None, fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Fuzzy document dedup using MinHash/LSH. Builds a dupekit pipeline (CleanText →
MinHash → MinHashLSH → SelectColumns), groups documents by LSH bucket, runs
`connected_components` to find near-duplicate clusters, then marks duplicates.
Constraint: `num_perms % num_bands == 0` or ValueError.
Import: `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document`

## `connected_components.py`

### `connected_components(ds, ctx, *, output_dir, max_iterations=10, preserve_singletons=True) -> tuple[bool, Sequence[str]]`
Hash-to-Min algorithm. Returns `(converged, vortex_file_paths)`. Non-convergence
only logs a warning — caller must check the bool.
Import: `from marin.processing.classification.deduplication.connected_components import connected_components`

## Common Parameters
- `input_paths: str | list[str]` — GCS glob or list of paths
- `output_path: str` — Where to write vortex attribute files
- `max_parallelism: int` — Number of Zephyr workers (required, no default)
- `worker_resources: ResourceConfig | None` — Default: `ResourceConfig(cpu=1, ram="32g", disk="5g")`
  Import ResourceConfig: `from fray.v2 import ResourceConfig`
- `filetypes: list[str] | None` — Default: `["jsonl", "jsonl.gz", "jsonl.zst", "parquet"]`

## dupekit — Rust FFI

Low-level primitives used internally. Key types:
- `dupekit.Transformation.CleanText(input_col, output_col)` — Normalize text
- `dupekit.Transformation.MinHash(input_col, output_col, num_perms, ngram_size, seed)` — MinHash signatures
- `dupekit.Transformation.MinHashLSH(input_col, output_col, num_bands)` — LSH banding
- `dupekit.Transformation.Hash(input_col, output_col, algo)` — Hashing (algo: `dupekit.HashAlgorithm.Xxh3_128`)
- `dupekit.Transformation.SelectColumns(columns)` — Column selection
- `dupekit.transform(batch: RecordBatch, steps: list[Transformation]) -> RecordBatch` — Run pipeline

## Example: Exact Document Dedup

```python
from marin.processing.classification.deduplication.exact import dedup_exact_document

result = dedup_exact_document(
    input_paths=["gs://my-bucket/data/shard-*.parquet"],
    output_path="gs://my-bucket/exact-dedup-out/",
    text_field="text",
    max_parallelism=64,
)
print(f"Total: {result['dedup/exact/document/total']}, Dups: {result['dedup/exact/document/dups']}")
```
"""

# ---------------------------------------------------------------------------
# V2: Same structure but more compressed — shorter descriptions, no dupekit
# internals, no example
# ---------------------------------------------------------------------------
V2_DOC = """\
# `marin.processing.classification.deduplication` — Package Reference

Removes duplicate content from text corpora. Produces vortex attribute files.

## Modules
- `exact.py` — Exact dedup via xxHash3-128
- `fuzzy.py` — Fuzzy dedup via MinHash/LSH (uses `dupekit` Rust FFI)
- `connected_components.py` — Hash-to-Min clustering (internal to fuzzy dedup)
- `dedup_commons.py` — Shared helpers (not called directly)

## Functions

`from marin.processing.classification.deduplication.exact import dedup_exact_document`
`dedup_exact_document(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Exact document dedup via full-text hash.

`from marin.processing.classification.deduplication.exact import dedup_exact_paragraph`
`dedup_exact_paragraph(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Paragraph-level exact dedup.

`from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document`
`dedup_fuzzy_document(*, input_paths, output_path, text_field="text", filetypes=None, fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict`
Fuzzy document dedup. `num_perms % num_bands` must equal 0.

`from marin.processing.classification.deduplication.connected_components import connected_components`
`connected_components(ds, ctx, *, output_dir, max_iterations=10, preserve_singletons=True) -> tuple[bool, Sequence[str]]`
Hash-to-Min clustering. Check returned bool for convergence.

## Common Parameters
- `max_parallelism: int` — Required. Number of Zephyr workers.
- `worker_resources` — Default: `ResourceConfig(cpu=1, ram="32g", disk="5g")` from `fray.v2`
"""

# ---------------------------------------------------------------------------
# V3: Semantic/conceptual doc — explains *how* dedup works, less focus on
# exact signatures, more on mental model
# ---------------------------------------------------------------------------
V3_DOC = """\
# `marin.processing.classification.deduplication` — Package Reference

## How Deduplication Works in Marin

Marin provides three dedup strategies, all following the same pattern:

1. **Exact document dedup** — Hash the full text of each document (xxHash3-128).
   Documents with identical hashes are duplicates. One is kept, rest are marked.
   Function: `dedup_exact_document` in `exact.py`.

2. **Exact paragraph dedup** — Split each document into paragraphs, hash each.
   Duplicate paragraphs across documents get their spans marked.
   Function: `dedup_exact_paragraph` in `exact.py`.

3. **Fuzzy document dedup** — MinHash/LSH to find near-duplicates:
   a. Clean text (lowercase, strip punctuation) via `dupekit.Transformation.CleanText`
   b. Compute MinHash signatures (character n-grams) via `dupekit.Transformation.MinHash`
   c. Apply LSH banding to group similar docs into buckets via `dupekit.Transformation.MinHashLSH`
   d. Run connected-components (Hash-to-Min algorithm) to cluster near-duplicates
   e. Mark all but one document per cluster as duplicates
   Function: `dedup_fuzzy_document` in `fuzzy.py`.

All three write **vortex attribute files** to `output_path` marking duplicates.
The original data is never modified.

## Calling Convention

Every dedup function uses keyword-only arguments:
```
dedup_*(
    input_paths="gs://..." or ["gs://..."],  # GCS globs or file list
    output_path="gs://...",                   # Where vortex files go
    text_field="text",                        # Column name in input data
    max_parallelism=64,                       # Required — number of Zephyr workers
)
```

For fuzzy dedup, additional MinHash parameters:
- `fuzzy_minhash_num_perms=286` — Number of hash permutations (must be divisible by num_bands)
- `fuzzy_minhash_num_bands=26` — Number of LSH bands
- `fuzzy_minhash_ngram_size=5` — Character n-gram size
- `fuzzy_minhash_seed=42` — Random seed

Worker resources default to `ResourceConfig(cpu=1, ram="32g", disk="5g")` from `fray.v2`.

## Import Paths

```python
from marin.processing.classification.deduplication.exact import dedup_exact_document
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from fray.v2 import ResourceConfig  # if customizing worker resources
```

## Example: Removing Exact Duplicates

```python
from marin.processing.classification.deduplication.exact import dedup_exact_document

result = dedup_exact_document(
    input_paths=["gs://my-bucket/data/"],
    output_path="gs://my-bucket/exact-dedup-out/",
    max_parallelism=64,
)
```
"""

# ---------------------------------------------------------------------------
# V4: Hierarchical — package-level overview is brief, then separate
# "module docs" for fuzzy.py and exact.py
# ---------------------------------------------------------------------------
V4_PACKAGE_DOC = """\
# `marin.processing.classification.deduplication` — Package Overview

Removes duplicate content from text corpora at document and paragraph level.
Two strategies: exact hash matching and fuzzy MinHash/LSH.

All functions produce vortex attribute files marking duplicates. They build
Zephyr `Dataset` pipelines internally.

## Modules
- `exact.py` — `dedup_exact_document`, `dedup_exact_paragraph` → `@docs/agent/modules/dedup_exact.md`
- `fuzzy.py` — `dedup_fuzzy_document` → `@docs/agent/modules/dedup_fuzzy.md`
- `connected_components.py` — Internal Hash-to-Min clustering (called by fuzzy dedup)
- `dedup_commons.py` — Shared helpers (not called directly)

Common parameters across all functions:
- `input_paths: str | list[str]` — GCS globs or file list
- `output_path: str` — Where to write vortex files
- `max_parallelism: int` — Required, no default. Number of Zephyr workers.
- `worker_resources` — Default: `ResourceConfig(cpu=1, ram="32g", disk="5g")` from `fray.v2`
"""

V4_EXACT_DOC = """\
# `deduplication.exact` — Module Reference

Exact deduplication via full-text xxHash3-128 hashing.

## `dedup_exact_document`
```python
from marin.processing.classification.deduplication.exact import dedup_exact_document

dedup_exact_document(
    *, input_paths, output_path, text_field="text", filetypes=None,
    max_parallelism, worker_resources=None, coordinator_resources=None,
) -> dict
```
Hashes full document text, groups by hash, marks all but one as duplicate.
Internally uses `dupekit.Transformation.Hash(algo=dupekit.HashAlgorithm.Xxh3_128)`.

## `dedup_exact_paragraph`
Same signature as `dedup_exact_document`. Splits documents into paragraphs first,
marks duplicate paragraph spans.

## Example
```python
result = dedup_exact_document(
    input_paths=["gs://bucket/data/"],
    output_path="gs://bucket/exact-out/",
    max_parallelism=64,
)
```
"""

V4_FUZZY_DOC = """\
# `deduplication.fuzzy` — Module Reference

Fuzzy document deduplication using MinHash/LSH via `dupekit` Rust FFI.

## `dedup_fuzzy_document`
```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

dedup_fuzzy_document(
    *, input_paths, output_path, text_field="text", filetypes=None,
    fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26,
    fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42,
    max_parallelism, worker_resources=None, coordinator_resources=None,
) -> dict
```

Pipeline stages: CleanText → MinHash → MinHashLSH → connected_components → mark duplicates.

**Constraint:** `fuzzy_minhash_num_perms` must be divisible by `fuzzy_minhash_num_bands` (286 % 26 == 0). Raises `ValueError` otherwise.

`connected_components` may not converge within `max_iterations=10` — it only warns, does not raise.

Worker resources: dedup map stage uses ~2 extra CPU cores for Rust thread pool beyond the Python thread. Consider `ResourceConfig(cpu=5, ram="32g")`.
"""

# ---------------------------------------------------------------------------
# V5: Maximally compressed — just signatures and imports, organized by file
# ---------------------------------------------------------------------------
V5_DOC = """\
# `marin.processing.classification.deduplication`

## exact.py
```python
from marin.processing.classification.deduplication.exact import dedup_exact_document, dedup_exact_paragraph

dedup_exact_document(*, input_paths: str | list[str], output_path: str, text_field: str = "text", filetypes: list[str] | None = None, max_parallelism: int, worker_resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None) -> dict
dedup_exact_paragraph(*, input_paths: str | list[str], output_path: str, text_field: str = "text", filetypes: list[str] | None = None, max_parallelism: int, worker_resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None) -> dict
```

## fuzzy.py
```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

dedup_fuzzy_document(*, input_paths: str | list[str], output_path: str, text_field: str = "text", filetypes: list[str] | None = None, fuzzy_minhash_num_perms: int = 286, fuzzy_minhash_num_bands: int = 26, fuzzy_minhash_ngram_size: int = 5, fuzzy_minhash_seed: int = 42, max_parallelism: int, worker_resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None) -> dict
```
Constraint: `num_perms % num_bands == 0`.

## connected_components.py
```python
from marin.processing.classification.deduplication.connected_components import connected_components

connected_components(ds: Dataset, ctx: ZephyrContext, *, output_dir: str, max_iterations: int = 10, preserve_singletons: bool = True) -> tuple[bool, Sequence[str]]
```

## ResourceConfig
```python
from fray.v2 import ResourceConfig
# Default for dedup workers: ResourceConfig(cpu=1, ram="32g", disk="5g")
```
"""

# ---------------------------------------------------------------------------
# Task: fuzzy dedup (but examples show exact dedup)
# ---------------------------------------------------------------------------
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

REVIEW_RUBRIC = """\
Review this Python script for correctness against the Marin codebase.

Score each criterion 0 (wrong) or 1 (correct):

1. IMPORT_PATH: Is `dedup_fuzzy_document` imported from `marin.processing.classification.deduplication.fuzzy`?
2. FUNCTION_NAME: Is the function called `dedup_fuzzy_document`?
3. NUM_PERMS: Is `fuzzy_minhash_num_perms` set to 286?
4. NUM_BANDS: Is `fuzzy_minhash_num_bands` set to 26?
5. NGRAM_SIZE: Is `fuzzy_minhash_ngram_size` set to 5?
6. SEED: Is `fuzzy_minhash_seed` set to 42?
7. MAX_PARALLELISM: Is `max_parallelism` provided?
8. KEYWORD_ONLY: Are arguments passed as keyword arguments?
9. RUNNABLE: Has `if __name__ == "__main__"` and argparse?
10. NO_HALLUCINATION: No invented APIs?

Output ONLY a JSON object with these keys and 0/1 values, plus a "notes" field.

Script to review:

```python
__SCRIPT__
```

Ground truth: `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document` with defaults num_perms=286, num_bands=26, ngram_size=5, seed=42. Keyword-only args. max_parallelism is required (no default).
"""

# ---------------------------------------------------------------------------
# Variation definitions
# ---------------------------------------------------------------------------
VARIATIONS = {
    1: {
        "description": "Full package doc + semantic overview + exact dedup example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V1_DOC,
    },
    2: {
        "description": "Compressed package doc (signatures + 1-liners, no example)",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V2_DOC,
    },
    3: {
        "description": "Conceptual/semantic doc + calling convention + exact example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V3_DOC,
    },
    4: {
        "description": "Hierarchical: package overview + separate module docs",
        "context": (
            ECOSYSTEM_OVERVIEW + "\n---\n\n" + V4_PACKAGE_DOC + "\n---\n\n" + V4_EXACT_DOC + "\n---\n\n" + V4_FUZZY_DOC
        ),
    },
    5: {
        "description": "Minimal: just signatures in code blocks, no prose",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V5_DOC,
    },
}


def ask_to_write_script(context: str, model: str) -> dict:
    """Ask the coding agent to write a fuzzy dedup script."""
    if context:
        prompt = f"""\
You have access to the following documentation for the Marin monorepo.
Use ONLY this documentation to write the script. Do not invent function names or parameters.

{context}

---

{TASK_PROMPT}
"""
    else:
        prompt = TASK_PROMPT

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        "0.50",
        "--output-format",
        "json",
        "--system-prompt",
        "You are writing Python scripts for the Marin monorepo. Output only code.",
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return {
            "script": f"# ERROR: {result.stderr[:500]}",
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "context_chars": len(context),
        }

    data = json.loads(result.stdout)
    script = data.get("result", "")

    if script.startswith("```"):
        lines = script.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        script = "\n".join(lines)

    return {
        "script": script,
        "cost_usd": data.get("total_cost_usd", 0),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "context_chars": len(context),
    }


def review_script(script_content: str, model: str) -> dict:
    """Senior engineer reviews the generated script."""
    prompt = REVIEW_RUBRIC.replace("__SCRIPT__", script_content)

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        "0.50",
        "--output-format",
        "json",
        "--system-prompt",
        "You are a senior engineer reviewing code. Output only the requested JSON.",
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("Review failed: %s", result.stderr[:500])
        return {"error": result.stderr[:500], "review_cost_usd": 0}

    data = json.loads(result.stdout)
    review_text = data.get("result", "")
    review_cost = data.get("total_cost_usd", 0)

    json_match = re.search(r"\{[^{}]+\}", review_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            return {**scores, "review_cost_usd": review_cost, "raw_review": review_text}
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse review JSON", "review_cost_usd": review_cost, "raw_review": review_text}


def run_variation(num: int, var: dict, output_dir: Path, gen_model: str, review_model: str) -> dict:
    """Run one variation: generate script with gen_model, review with review_model."""
    desc = var["description"]
    context = var["context"]

    var_dir = output_dir / f"variation_{num:02d}"
    var_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== V%d: %s ===", num, desc)
    logger.info("  Context: %d chars | Gen model: %s | Review model: %s", len(context), gen_model, review_model)

    gen_result = ask_to_write_script(context, gen_model)
    (var_dir / "dedup_script.py").write_text(gen_result["script"] + "\n")
    (var_dir / "context.txt").write_text(context + "\n")
    logger.info(
        "  Gen cost: $%.4f (%d in / %d out)",
        gen_result["cost_usd"],
        gen_result["input_tokens"],
        gen_result["output_tokens"],
    )

    logger.info("  Reviewing...")
    review = review_script(gen_result["script"], review_model)
    review_cost = review.get("review_cost_usd", 0)
    logger.info("  Review cost: $%.4f", review_cost)

    criteria = [
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
    scores = {c: review.get(c, 0) for c in criteria}
    total_correct = sum(scores.values())

    result = {
        "variation": num,
        "description": desc,
        "context_chars": len(context),
        "gen_model": gen_model,
        "review_model": review_model,
        "generation_cost_usd": gen_result["cost_usd"],
        "review_cost_usd": review_cost,
        "total_cost_usd": gen_result["cost_usd"] + review_cost,
        "gen_input_tokens": gen_result["input_tokens"],
        "gen_output_tokens": gen_result["output_tokens"],
        "scores": scores,
        "total_correct": total_correct,
        "total_criteria": len(criteria),
        "accuracy": total_correct / len(criteria),
        "notes": review.get("notes", ""),
    }

    (var_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (var_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n")

    logger.info(
        "  Score: %d/%d (%.0f%%) | Notes: %s",
        total_correct,
        len(criteria),
        100 * result["accuracy"],
        review.get("notes", ""),
    )
    return result


def compile_report(results: list[dict], output_dir: Path) -> str:
    lines = [
        "# Autodoc Variation Experiment — Round 3\n",
        "## Setup",
        "- **Coding agent**: haiku (weaker model)",
        "- **Reviewer**: sonnet",
        "- **Task**: Write a fuzzy dedup script (docs show exact dedup examples only)",
        "- **Docs**: Organized by package/folder, not by topic\n",
        "## Results\n",
        "| # | Variation | Context | Gen Cost | Score | Acc |",
        "|---|-----------|---------|----------|-------|-----|",
    ]

    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(
            f"| {r['variation']} | {r['description']} | {r['context_chars']:,} | ${r['generation_cost_usd']:.4f} | {r['total_correct']}/{r['total_criteria']} | {r['accuracy']:.0%} |"
        )

    lines.append("\n## Analysis\n")

    perfect = [r for r in results if r["accuracy"] == 1.0]
    if perfect:
        best = min(perfect, key=lambda r: r["generation_cost_usd"])
        lines.append(
            f"### Best perfect score\n**V{best['variation']}** ({best['description']}): ${best['generation_cost_usd']:.4f} at {best['context_chars']:,} chars\n"
        )

    good = [r for r in results if r["accuracy"] >= 0.8]
    if good:
        cheapest = min(good, key=lambda r: r["generation_cost_usd"])
        lines.append(
            f"### Best cost-accuracy tradeoff (≥80%)\n**V{cheapest['variation']}** ({cheapest['description']}): ${cheapest['generation_cost_usd']:.4f} at {cheapest['accuracy']:.0%}\n"
        )

    criteria = [
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
    lines.append("### Per-criterion pass rates\n")
    lines.append("| Criterion | Pass Rate |")
    lines.append("|-----------|-----------|")
    for c in criteria:
        passes = sum(1 for r in results if r["scores"].get(c, 0) == 1)
        lines.append(f"| {c} | {passes}/{len(results)} |")

    lines.append("\n## Scripts\n")
    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(f"### V{r['variation']}: {r['description']}")
        lines.append(
            f"*Context: {r['context_chars']:,} | Cost: ${r['generation_cost_usd']:.4f} | Score: {r['total_correct']}/{r['total_criteria']}*\n"
        )
        script_file = output_dir / f"variation_{r['variation']:02d}" / "dedup_script.py"
        if script_file.exists():
            lines.append("```python")
            lines.append(script_file.read_text().strip())
            lines.append("```\n")
        lines.append(f"**Review:** {r.get('notes', 'N/A')}\n")
        for c in criteria:
            v = r["scores"].get(c, 0)
            lines.append(f"- {'✓' if v else '✗'} {c}")
        lines.append("")

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--variation", type=int)
    parser.add_argument("--gen-model", type=str, default="haiku")
    parser.add_argument("--review-model", type=str, default="sonnet")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        results = [
            json.loads((output_dir / f"variation_{v:02d}" / "result.json").read_text())
            for v in range(1, 6)
            if (output_dir / f"variation_{v:02d}" / "result.json").exists()
        ]
        if results:
            print(compile_report(results, output_dir))
        return

    to_run = {args.variation: VARIATIONS[args.variation]} if args.variation else VARIATIONS

    results = []
    for num, var in sorted(to_run.items()):
        results.append(run_variation(num, var, output_dir, args.gen_model, args.review_model))

    all_results = [
        json.loads((output_dir / f"variation_{v:02d}" / "result.json").read_text())
        for v in range(1, 6)
        if (output_dir / f"variation_{v:02d}" / "result.json").exists()
    ]
    if len(all_results) > 1:
        print(compile_report(all_results, output_dir))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
