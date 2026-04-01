#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Round 4 autodoc variation experiments.

Changes from round 3:
- NO examples from the dedup domain at all (exact dedup was still cheating)
- Examples (when present) come from completely unrelated domains: tokenization,
  Zephyr Dataset API, iris jobs
- Still haiku as coding agent, sonnet as reviewer
- Same task: write a fuzzy dedup script

Usage:
    uv run scripts/autodoc_variations_r4.py --output-dir /tmp/autodoc-r4
    uv run scripts/autodoc_variations_r4.py --variation 1 --output-dir /tmp/autodoc-r4
    uv run scripts/autodoc_variations_r4.py --report-only --output-dir /tmp/autodoc-r4
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
# Dedup doc: conceptual + calling convention (V3/R3 winning format)
# but NO dedup example of any kind
# ---------------------------------------------------------------------------
DEDUP_DOC_NO_EXAMPLE = """\
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
"""

# ---------------------------------------------------------------------------
# Unrelated examples from other domains
# ---------------------------------------------------------------------------
TOKENIZATION_EXAMPLE = """\
## Example: Tokenizing a Dataset (unrelated domain)

```python
from marin.processing.tokenize.tokenize_dolma import tokenize_dataset, TokenizeConfig

config = TokenizeConfig(
    tokenizer_path="gs://bucket/tokenizers/llama3",
    output_path="gs://bucket/tokenized-out/",
    max_seq_len=8192,
)
tokenize_dataset(
    input_paths=["gs://bucket/cleaned-data/"],
    config=config,
    max_parallelism=32,
)
```
"""

ZEPHYR_EXAMPLE = """\
## Example: Zephyr Dataset Pipeline (unrelated domain)

```python
from zephyr.dataset import Dataset
from zephyr import ZephyrContext

ctx = ZephyrContext(name="my-job", max_workers=16)
result = list(ctx.execute(
    Dataset.from_list(["gs://bucket/file1.parquet", "gs://bucket/file2.parquet"])
    .flat_map(lambda path: load_records(path))
    .map(lambda r: {**r, "length": len(r["text"])})
    .group_by(lambda r: r["source"], reducer=aggregate_stats),
    verbose=True,
))
```
"""

IRIS_EXAMPLE = """\
## Example: Iris Job (unrelated domain)

```python
from iris.controller import IrisController

controller = IrisController(
    job_name="my-processing-job",
    worker_image="us-docker.pkg.dev/marin/workers/processor:latest",
    max_workers=64,
    worker_resources={"cpu": 4, "memory": "16Gi"},
)
controller.submit(input_manifest="gs://bucket/manifest.json")
controller.wait()
```
"""

# ---------------------------------------------------------------------------
# V1: R3-V3 format, no example at all
# ---------------------------------------------------------------------------
V1_DOC = DEDUP_DOC_NO_EXAMPLE

# ---------------------------------------------------------------------------
# V2: R3-V3 format + tokenization example (completely unrelated domain)
# ---------------------------------------------------------------------------
V2_DOC = DEDUP_DOC_NO_EXAMPLE + "\n" + TOKENIZATION_EXAMPLE

# ---------------------------------------------------------------------------
# V3: R3-V3 format + Zephyr pipeline example (related infra, but not dedup)
# ---------------------------------------------------------------------------
V3_DOC = DEDUP_DOC_NO_EXAMPLE + "\n" + ZEPHYR_EXAMPLE

# ---------------------------------------------------------------------------
# V4: R3-V3 format + iris job example (completely unrelated)
# ---------------------------------------------------------------------------
V4_DOC = DEDUP_DOC_NO_EXAMPLE + "\n" + IRIS_EXAMPLE

# ---------------------------------------------------------------------------
# V5: R3-V3 format, but the calling convention section is beefed up with
# a pseudo-example inline (shows calling pattern without being a real example)
# ---------------------------------------------------------------------------
V5_DOC = """\
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

Every dedup function uses keyword-only arguments. Here is the complete call
signature for fuzzy dedup — all other dedup functions follow the same pattern
minus the minhash-specific parameters:

```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from fray.v2 import ResourceConfig

result = dedup_fuzzy_document(
    input_paths=["gs://..."],         # str | list[str] — GCS globs or file list
    output_path="gs://...",           # str — where to write vortex attribute files
    text_field="text",                # str — column name in input data
    fuzzy_minhash_num_perms=286,      # int — must be divisible by num_bands
    fuzzy_minhash_num_bands=26,       # int — LSH bands
    fuzzy_minhash_ngram_size=5,       # int — character n-gram size
    fuzzy_minhash_seed=42,            # int — random seed
    max_parallelism=64,               # int — required, number of Zephyr workers
    worker_resources=ResourceConfig(cpu=1, ram="32g", disk="5g"),  # optional
)
# result is a dict with counter keys like "dedup/fuzzy/document/total"
```

Other dedup imports:
```python
from marin.processing.classification.deduplication.exact import dedup_exact_document
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
```
"""

# ---------------------------------------------------------------------------
# Task and review — same as R3
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
        "description": "Conceptual + calling convention, no example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V1_DOC,
    },
    2: {
        "description": "Conceptual + calling convention + tokenization example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V2_DOC,
    },
    3: {
        "description": "Conceptual + calling convention + Zephyr pipeline example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V3_DOC,
    },
    4: {
        "description": "Conceptual + calling convention + Iris job example",
        "context": ECOSYSTEM_OVERVIEW + "\n---\n\n" + V4_DOC,
    },
    5: {
        "description": "Conceptual + annotated call-site (pseudo-example in docs)",
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
    logger.info(
        "  Context: %d chars | Gen model: %s | Review model: %s",
        len(context),
        gen_model,
        review_model,
    )

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
        "# Autodoc Variation Experiment — Round 4\n",
        "## Setup",
        "- **Coding agent**: haiku (weaker model)",
        "- **Reviewer**: sonnet",
        "- **Task**: Write a fuzzy dedup script",
        "- **Key change from R3**: No dedup examples at all. Examples (when present) come from",
        "  completely unrelated domains (tokenization, Zephyr pipelines, Iris jobs).\n",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

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
