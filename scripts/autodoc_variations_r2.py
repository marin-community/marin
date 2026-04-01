#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Round 2 autodoc variation experiments.

Changes from round 1:
- Task is harder: write a working dedup script (not just answer a question)
- All docs start from a top-level overview of the Marin ecosystem
- Senior engineer agent reviews each generated script for correctness
- 10 new variations exploring the "overview + 1-sentence descriptions + refs" sweet spot

Usage:
    uv run scripts/autodoc_variations_r2.py --variation 1 --output-dir /tmp/autodoc-r2
    uv run scripts/autodoc_variations_r2.py --output-dir /tmp/autodoc-r2
    uv run scripts/autodoc_variations_r2.py --report-only --output-dir /tmp/autodoc-r2
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Top-level ecosystem overview (every variation gets this as the root)
# ---------------------------------------------------------------------------
ECOSYSTEM_OVERVIEW = """\
# Marin Ecosystem

Marin is a monorepo for training and curating large language models.

## Libraries

| Library | Purpose | Docs |
|---------|---------|------|
| **marin** | Data processing pipelines: classification, deduplication, decontamination, tokenization. | `@docs/agent/modules/marin.processing.md` |
| **zephyr** | Lazy dataset pipeline engine with typed operations (`map`, `flat_map`, `group_by`). Execution is deferred to `ZephyrContext`. | `@docs/agent/modules/zephyr.dataset.md` `@docs/agent/modules/zephyr.execution.md` |
| **dupekit** | Rust FFI (PyO3) for fast hashing (xxHash3, BLAKE3) and MinHash/LSH deduplication primitives. | `@docs/agent/modules/dupekit.md` |
| **iris** | Distributed job orchestration. Controller-based scheduler with autoscaling, gRPC workers, container management. | `@docs/agent/modules/iris.md` |
| **fray** | RPC framework: actors, async function invocation. Powers Iris and Zephyr coordinator-worker communication. | `@docs/agent/modules/fray.md` |
| **rigging** | Unified storage abstraction (GCS, S3, local) via fsspec. | `@docs/agent/modules/rigging.md` |
| **levanter** | LLM training framework (JAX). Supports Llama, Gemma, Qwen, etc. with distributed training and checkpointing. | `@docs/agent/modules/levanter.md` |
| **haliax** | Named tensor library for JAX. Foundation for Levanter. | `@docs/agent/modules/haliax.md` |

## Dependency Direction
`{iris, haliax}` → `{levanter, zephyr}` → `marin`. Each layer imports only from layers to its left.

## Key Conventions
- All data flows through Parquet or Vortex files on GCS.
- Zephyr pipelines are lazy: build a `Dataset`, then call `ZephyrContext.execute(dataset)`.
- `dupekit` provides the low-level MinHash/LSH; `marin.processing` wraps it into high-level dedup functions.
"""

# ---------------------------------------------------------------------------
# Module-level docs at different granularities
# ---------------------------------------------------------------------------

# Full-signature reference for dedup-related functions
DEDUP_FULL_SIGNATURES = """\
## marin.processing — Deduplication

### Functions
- `dedup_fuzzy_document(*, input_paths: str | list[str], output_path: str, text_field: str = "text", filetypes: list[str] | None = None, fuzzy_minhash_num_perms: int = 286, fuzzy_minhash_num_bands: int = 26, fuzzy_minhash_ngram_size: int = 5, fuzzy_minhash_seed: int = 42, max_parallelism: int, worker_resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None) -> dict`
  Fuzzy document dedup. Validates `num_perms % num_bands == 0`. Internally runs dupekit MinHash/LSH pipeline, then `connected_components` to cluster near-duplicates. Writes vortex attribute files. Import: `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document`

- `dedup_exact_document(*, input_paths: str | list[str], output_path: str, text_field: str = "text", filetypes: list[str] | None = None, max_parallelism: int, worker_resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None) -> dict`
  Exact hash dedup via xxHash3-128. Import: `from marin.processing.classification.deduplication.exact import dedup_exact_document`

- `connected_components(ds: Dataset, ctx: ZephyrContext, *, output_dir: str, max_iterations: int = 10, preserve_singletons: bool = True) -> tuple[bool, Sequence[str]]`
  Hash-to-Min connected components. Returns (converged, output_vortex_paths). Import: `from marin.processing.classification.deduplication.connected_components import connected_components`

## dupekit — Rust FFI

### Functions
- `clean_text(arr: StringArray) -> StringArray` — Normalize text (lowercase, strip punctuation).
- `compute_minhash(arr: StringArray, num_perms: int, ngram_size: int, seed: int) -> ListArray[uint64]` — MinHash signatures.
- `compute_lsh(input_col: ListArray, num_bands: int) -> ListArray[uint64]` — LSH band bucket hashes.
- `hash_xxh3_128(data: bytes) -> int` — XXH3-128 hash.
- `transform(batch: RecordBatch, steps: list[Transformation]) -> RecordBatch` — Execute pipeline steps on a batch.

### Pipeline Transformations (dupekit.Transformation)
- `CleanText(input_col, output_col)` — Text cleaning step.
- `MinHash(input_col, output_col, num_perms, ngram_size, seed)` — MinHash step.
- `MinHashLSH(input_col, output_col, num_bands)` — LSH banding step.
- `SelectColumns(columns: list[str])` — Column selection step.

## zephyr — Pipeline Engine

### Core Classes
- `Dataset[T]` — Lazy pipeline. Created via `Dataset.from_list(items)` or `Dataset.from_files(pattern)`. Operations: `.map(fn)`, `.flat_map(fn)`, `.group_by(key_fn, sort_by=, reducer=)`, `.load_parquet()`, `.load_vortex()`.
- `ZephyrContext(name: str, max_workers: int, resources: ResourceConfig | None = None, coordinator_resources: ResourceConfig | None = None)` — Executes a dataset pipeline. Call `ctx.execute(dataset)` to run.

### Key Pattern
```python
ctx = ZephyrContext(name="my-job", max_workers=64)
ds = Dataset.from_list(file_paths).load_parquet().map(transform_fn)
results = list(ctx.execute(ds))
```

## fray — Resources
- `ResourceConfig(cpu: int = 1, ram: str = "32g", disk: str = "5g")` — Worker resource spec. Import: `from fray.v2 import ResourceConfig`
"""

# 1-sentence summaries only (no signatures)
DEDUP_ONE_SENTENCE = """\
## marin.processing — Deduplication
- `dedup_fuzzy_document` — High-level fuzzy document dedup using MinHash/LSH. Defaults: num_perms=286, num_bands=26, ngram_size=5, seed=42. Import: `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document`
- `dedup_exact_document` — High-level exact document dedup using xxHash3-128. Import: `from marin.processing.classification.deduplication.exact import dedup_exact_document`
- `connected_components` — Hash-to-Min clustering over LSH buckets. Import: `from marin.processing.classification.deduplication.connected_components import connected_components`

## dupekit — Rust FFI
- `clean_text`, `compute_minhash`, `compute_lsh` — Low-level MinHash/LSH primitives. Usually called via `dedup_fuzzy_document`.
- `Transformation.CleanText/MinHash/MinHashLSH/SelectColumns` — Pipeline steps for `dupekit.transform(batch, steps)`.

## zephyr — Pipeline Engine
- `Dataset[T]` — Lazy pipeline: `from_list()`, `.map()`, `.flat_map()`, `.group_by()`. Import: `from zephyr.dataset import Dataset`
- `ZephyrContext(name, max_workers, resources=)` — Executes pipelines. Import: `from zephyr import ZephyrContext`

## fray — Resources
- `ResourceConfig(cpu=1, ram="32g", disk="5g")` — Worker resources. Import: `from fray.v2 import ResourceConfig`
"""

# Grep-friendly index (one function per line, tab-separated)
DEDUP_GREP_INDEX = """\
# Function Index (grep-friendly: NAME | SIGNATURE | IMPORT)
dedup_fuzzy_document | (*, input_paths, output_path, text_field="text", filetypes=None, fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict | from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
dedup_exact_document | (*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict | from marin.processing.classification.deduplication.exact import dedup_exact_document
connected_components | (ds, ctx, *, output_dir, max_iterations=10, preserve_singletons=True) -> tuple[bool, Sequence[str]] | from marin.processing.classification.deduplication.connected_components import connected_components
Dataset | Lazy pipeline: from_list(), .map(), .flat_map(), .group_by(), .load_parquet() | from zephyr.dataset import Dataset
ZephyrContext | (name: str, max_workers: int, resources: ResourceConfig | None = None) — executes datasets | from zephyr import ZephyrContext
ResourceConfig | (cpu: int = 1, ram: str = "32g", disk: str = "5g") | from fray.v2 import ResourceConfig
clean_text | (arr: StringArray) -> StringArray — normalize text | import dupekit
compute_minhash | (arr: StringArray, num_perms: int, ngram_size: int, seed: int) -> ListArray | import dupekit
compute_lsh | (input_col: ListArray, num_bands: int) -> ListArray | import dupekit
Transformation.CleanText | (input_col, output_col) — pipeline step | dupekit.Transformation.CleanText
Transformation.MinHash | (input_col, output_col, num_perms, ngram_size, seed) — pipeline step | dupekit.Transformation.MinHash
Transformation.MinHashLSH | (input_col, output_col, num_bands) — pipeline step | dupekit.Transformation.MinHashLSH
"""

# Usage example (demonstrates the pattern)
DEDUP_EXAMPLE = """\
## Example: Fuzzy Document Deduplication

```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

result = dedup_fuzzy_document(
    input_paths=["gs://my-bucket/data/shard-*.parquet"],
    output_path="gs://my-bucket/dedup-output/",
    text_field="text",
    fuzzy_minhash_num_perms=286,
    fuzzy_minhash_num_bands=26,
    fuzzy_minhash_ngram_size=5,
    fuzzy_minhash_seed=42,
    max_parallelism=64,
)
# result is a dict with dedup counters; output vortex files at output_path
```

Key constraints:
- `fuzzy_minhash_num_perms` must be divisible by `fuzzy_minhash_num_bands` (286 % 26 == 0).
- `connected_components` may not converge — check the `converged` bool in the return tuple.
- Worker resources default to 1 CPU, 32GB RAM, 5GB disk; dedup map stage uses ~2 extra cores for Rust thread pool.
"""

# ---------------------------------------------------------------------------
# The task and evaluation
# ---------------------------------------------------------------------------
TASK_PROMPT = """\
Write a Python script that fuzzy-deduplicates a set of Parquet files.

Requirements:
- The script takes two arguments: an input glob pattern (e.g. "gs://bucket/data/*.parquet") and an output path.
- Use Marin's built-in dedup pipeline (not a custom implementation).
- Use appropriate defaults for MinHash parameters.
- The script should be complete and runnable.
- Write the script to: {output_file}
- Include proper imports and a `if __name__ == "__main__"` block.
- Parse arguments with argparse.

IMPORTANT: Use ONLY the documentation provided above to write this script. Do not invent function names or parameters.\
"""

REVIEW_RUBRIC = """\
Review this Python script for correctness against the Marin codebase.

Score each criterion 0 (wrong) or 1 (correct):

1. IMPORT_PATH: Is `dedup_fuzzy_document` imported from the correct path? (`marin.processing.classification.deduplication.fuzzy`)
2. FUNCTION_NAME: Is the function called `dedup_fuzzy_document` (not dedup_documents, fuzzy_dedup, etc.)?
3. NUM_PERMS: Is `fuzzy_minhash_num_perms` set to 286 (not 128, 256, etc.)?
4. NUM_BANDS: Is `fuzzy_minhash_num_bands` set to 26?
5. NGRAM_SIZE: Is `fuzzy_minhash_ngram_size` set to 5?
6. SEED: Is `fuzzy_minhash_seed` set to 42?
7. MAX_PARALLELISM: Is `max_parallelism` provided as a required argument?
8. KEYWORD_ONLY: Are the arguments passed as keyword arguments (not positional)?
9. RUNNABLE: Does the script have `if __name__ == "__main__"` and argparse?
10. NO_HALLUCINATION: Does the script avoid inventing non-existent APIs (e.g. `Deduplicator()`, `marin.dedup()`, etc.)?

Output ONLY a JSON object with these keys and 0/1 values, plus a "notes" field with a one-sentence summary.

Here is the script to review:

```python
__SCRIPT_PLACEHOLDER__
```

Ground truth: the correct import is `from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document` with defaults num_perms=286, num_bands=26, ngram_size=5, seed=42. The function takes keyword-only arguments including max_parallelism (required, no default).
"""


# ---------------------------------------------------------------------------
# 10 Variations
# ---------------------------------------------------------------------------
def _v(num: int, desc: str, context: str) -> dict:
    return {"num": num, "description": desc, "context": context}


def build_variations(full_module_docs: dict[str, str]) -> list[dict]:
    """Build 10 variations. full_module_docs has keys like 'marin.processing', 'dupekit', etc."""

    def _extract_section(doc: str, section: str) -> str:
        import re

        pattern = rf"^## {re.escape(section)}\s*\n(.*?)(?=^## |\Z)"
        match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    variations = []

    # V1: Overview + full signatures (no example)
    variations.append(_v(1, "Overview + full signatures", ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_FULL_SIGNATURES))

    # V2: Overview + 1-sentence descriptions (no signatures)
    variations.append(_v(2, "Overview + 1-sentence descriptions", ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_ONE_SENTENCE))

    # V3: Overview + grep-friendly index
    variations.append(_v(3, "Overview + grep-friendly index", ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_GREP_INDEX))

    # V4: Overview + 1-sentence + example
    variations.append(
        _v(
            4,
            "Overview + 1-sentence + example",
            ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_ONE_SENTENCE + "\n---\n\n" + DEDUP_EXAMPLE,
        )
    )

    # V5: Overview + full signatures + example
    variations.append(
        _v(
            5,
            "Overview + full signatures + example",
            ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_FULL_SIGNATURES + "\n---\n\n" + DEDUP_EXAMPLE,
        )
    )

    # V6: Overview only (no module docs) — can the agent figure it out from just the overview?
    variations.append(_v(6, "Overview only (no module docs)", ECOSYSTEM_OVERVIEW))

    # V7: Overview + generated module docs for marin.processing only
    mp_doc = full_module_docs.get("marin.processing", "")
    variations.append(
        _v(
            7,
            "Overview + generated marin.processing doc",
            ECOSYSTEM_OVERVIEW + "\n---\n\n# Module: marin.processing\n\n" + mp_doc,
        )
    )

    # V8: Overview + generated docs for marin.processing + dupekit + zephyr
    parts = []
    for mod in ["marin.processing", "dupekit", "zephyr.dataset"]:
        doc = full_module_docs.get(mod, "")
        if doc:
            parts.append(f"# Module: {mod}\n\n{doc}")
    variations.append(
        _v(8, "Overview + generated 3-module docs", ECOSYSTEM_OVERVIEW + "\n---\n\n" + "\n\n---\n\n".join(parts))
    )

    # V9: Overview + 1-sentence + grep index (combo of V2 and V3)
    variations.append(
        _v(
            9,
            "Overview + 1-sentence + grep index",
            ECOSYSTEM_OVERVIEW + "\n---\n\n" + DEDUP_ONE_SENTENCE + "\n---\n\n" + DEDUP_GREP_INDEX,
        )
    )

    # V10: No context baseline
    variations.append(_v(10, "No context baseline", ""))

    return variations


def ask_to_write_script(context: str, output_file: str, model: str) -> dict:
    """Ask claude to write a dedup script given the doc context."""
    task = TASK_PROMPT.replace("{output_file}", output_file)

    if context:
        prompt = f"""\
You have access to the following documentation for the Marin monorepo.
Use ONLY this documentation to write the script. Do not invent function names or parameters.

{context}

---

{task}

Output ONLY the Python script, no explanation.\
"""
    else:
        prompt = f"""\
{task}

You are writing against the Marin monorepo. Use your best knowledge of the codebase.
Output ONLY the Python script, no explanation.\
"""

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

    # Strip markdown fences if present
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
    """Have a senior engineer review the generated script."""
    prompt = REVIEW_RUBRIC.replace("__SCRIPT_PLACEHOLDER__", script_content)

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
        "You are a senior engineer reviewing code for correctness. Output only the requested JSON.",
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("Review failed: %s", result.stderr[:500])
        return {"error": result.stderr[:500], "review_cost_usd": 0}

    data = json.loads(result.stdout)
    review_text = data.get("result", "")
    review_cost = data.get("total_cost_usd", 0)

    # Parse the JSON scores from the review
    # Find JSON in the response
    import re

    json_match = re.search(r"\{[^{}]+\}", review_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            return {**scores, "review_cost_usd": review_cost, "raw_review": review_text}
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse review JSON", "review_cost_usd": review_cost, "raw_review": review_text}


def run_variation(variation: dict, output_dir: Path, model: str) -> dict:
    """Run a single variation: generate script, then review it."""
    num = variation["num"]
    desc = variation["description"]
    context = variation["context"]

    var_dir = output_dir / f"variation_{num:02d}"
    var_dir.mkdir(parents=True, exist_ok=True)

    script_path = var_dir / "dedup_script.py"

    logger.info("=== Variation %d: %s ===", num, desc)
    logger.info("  Context size: %d chars", len(context))

    # Step 1: Generate the script
    gen_result = ask_to_write_script(context, str(script_path), model)
    script_path.write_text(gen_result["script"] + "\n")
    (var_dir / "context.txt").write_text(context + "\n")

    logger.info(
        "  Generation cost: $%.4f (%d in / %d out tokens)",
        gen_result["cost_usd"],
        gen_result["input_tokens"],
        gen_result["output_tokens"],
    )

    # Step 2: Review the script
    logger.info("  Reviewing script...")
    review = review_script(gen_result["script"], model)

    review_cost = review.get("review_cost_usd", 0)
    logger.info("  Review cost: $%.4f", review_cost)

    # Calculate scores
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
    total_criteria = len(criteria)

    result = {
        "variation": num,
        "description": desc,
        "context_chars": len(context),
        "generation_cost_usd": gen_result["cost_usd"],
        "review_cost_usd": review_cost,
        "total_cost_usd": gen_result["cost_usd"] + review_cost,
        "gen_input_tokens": gen_result["input_tokens"],
        "gen_output_tokens": gen_result["output_tokens"],
        "scores": scores,
        "total_correct": total_correct,
        "total_criteria": total_criteria,
        "accuracy": total_correct / total_criteria if total_criteria else 0,
        "notes": review.get("notes", ""),
    }

    (var_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (var_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n")

    logger.info(
        "  Score: %d/%d (%.0f%%) | Gen: $%.4f | Review: $%.4f | Notes: %s",
        total_correct,
        total_criteria,
        100 * result["accuracy"],
        gen_result["cost_usd"],
        review_cost,
        review.get("notes", ""),
    )

    return result


def compile_report(results: list[dict], output_dir: Path) -> str:
    """Compile markdown report."""
    lines = [
        "# Autodoc Variation Experiment — Round 2\n",
        "## Objective",
        "Minimize agent token cost while producing a correct, runnable dedup script.",
        "Agent starts from a top-level Marin ecosystem overview and must navigate to the right APIs.\n",
        "## Task",
        "> Write a Python script that fuzzy-deduplicates Parquet files using Marin's built-in pipeline.",
        "> Script must include proper imports, argparse, correct function name and defaults.\n",
        "## Scoring Criteria",
        "| Criterion | Description |",
        "|-----------|-------------|",
        "| IMPORT_PATH | Correct import path for `dedup_fuzzy_document` |",
        "| FUNCTION_NAME | Uses `dedup_fuzzy_document` (not invented names) |",
        "| NUM_PERMS | Default 286 (not 128) |",
        "| NUM_BANDS | Default 26 |",
        "| NGRAM_SIZE | Default 5 |",
        "| SEED | Default 42 |",
        "| MAX_PARALLELISM | Required arg provided |",
        "| KEYWORD_ONLY | Keyword arguments used |",
        "| RUNNABLE | Has `__main__` block + argparse |",
        "| NO_HALLUCINATION | No invented APIs |",
        "",
        "## Results\n",
        "| # | Variation | Context | Gen Cost | Score | Accuracy |",
        "|---|-----------|---------|----------|-------|----------|",
    ]

    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(
            f"| {r['variation']} | {r['description']} | {r['context_chars']:,} chars | "
            f"${r['generation_cost_usd']:.4f} | {r['total_correct']}/{r['total_criteria']} | "
            f"{r['accuracy']:.0%} |"
        )

    lines.append("\n## Analysis\n")

    # Best tradeoff
    correct_enough = [r for r in results if r["accuracy"] >= 0.8]
    if correct_enough:
        cheapest = min(correct_enough, key=lambda r: r["generation_cost_usd"])
        lines.append("### Best cost-accuracy tradeoff")
        lines.append(
            f"**V{cheapest['variation']}** ({cheapest['description']}): "
            f"${cheapest['generation_cost_usd']:.4f} at {cheapest['accuracy']:.0%} accuracy "
            f"({cheapest['context_chars']:,} chars)\n"
        )

    perfect = [r for r in results if r["accuracy"] == 1.0]
    if perfect:
        cheapest_perfect = min(perfect, key=lambda r: r["generation_cost_usd"])
        lines.append("### Cheapest perfect score")
        lines.append(
            f"**V{cheapest_perfect['variation']}** ({cheapest_perfect['description']}): "
            f"${cheapest_perfect['generation_cost_usd']:.4f} ({cheapest_perfect['context_chars']:,} chars)\n"
        )

    # Per-criterion pass rates
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
        lines.append(f"| {c} | {passes}/{len(results)} ({100*passes/len(results):.0f}%) |")

    # Detailed results
    lines.append("\n## Detailed Results\n")
    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(f"### V{r['variation']}: {r['description']}")
        lines.append(
            f"*Context: {r['context_chars']:,} chars | Gen cost: ${r['generation_cost_usd']:.4f} | Score: {r['total_correct']}/{r['total_criteria']}*\n"
        )

        script_file = output_dir / f"variation_{r['variation']:02d}" / "dedup_script.py"
        if script_file.exists():
            script = script_file.read_text().strip()
            lines.append("```python")
            lines.append(script)
            lines.append("```\n")

        lines.append(f"**Review notes:** {r.get('notes', 'N/A')}\n")

        lines.append("**Scores:**")
        for c in criteria:
            val = r["scores"].get(c, 0)
            mark = "✓" if val else "✗"
            lines.append(f"- {mark} {c}")
        lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "report.md"
    report_path.write_text(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Round 2 autodoc variation experiments")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--variation", type=int, help="Run only this variation (1-10)")
    parser.add_argument("--model", type=str, default="sonnet")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        results = []
        for v in range(1, 11):
            rp = output_dir / f"variation_{v:02d}" / "result.json"
            if rp.exists():
                results.append(json.loads(rp.read_text()))
        if results:
            report = compile_report(results, output_dir)
            print(report)
        return

    # Load pre-generated module docs from round 1
    r1_docs_dir = Path("/tmp/autodoc-variations/generated_docs/modules")
    full_module_docs = {}
    if r1_docs_dir.exists():
        for md in r1_docs_dir.glob("*.md"):
            full_module_docs[md.stem] = md.read_text()
    else:
        logger.warning("No round 1 docs found at %s; V7 and V8 will have empty module docs", r1_docs_dir)

    all_variations = build_variations(full_module_docs)

    if args.variation:
        to_run = [v for v in all_variations if v["num"] == args.variation]
    else:
        to_run = all_variations

    results = []
    for v in to_run:
        result = run_variation(v, output_dir, args.model)
        results.append(result)

    # Compile full report
    all_results = []
    for v in range(1, 11):
        rp = output_dir / f"variation_{v:02d}" / "result.json"
        if rp.exists():
            all_results.append(json.loads(rp.read_text()))

    if len(all_results) > 1:
        report = compile_report(all_results, output_dir)
        print(report)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
