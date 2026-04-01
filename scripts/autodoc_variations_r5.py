#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tree-sitter==0.24.0",
#     "tree-sitter-python==0.23.6",
#     "tree-sitter-rust==0.23.2",
#     "pyyaml>=6.0",
# ]
# ///
"""Round 5 autodoc variation experiments.

Key constraint: documentation is generated at the module/package level by the
actual doc generator pipeline. NO task-specific docs allowed. The only thing
that varies is the MODULE_PROMPT — how much prose vs how code is described.

Variations:
  V1: Current baseline prompt (Purpose/API/Dependencies/Abstractions/Gotchas, <150 lines)
  V2: Minimal — signatures only, no prose sections, <80 lines
  V3: Heavy prose — conceptual overview of how the module works, then API, <200 lines
  V4: Code-focused — signatures in code blocks with type annotations, brief prose, <120 lines
  V5: Hybrid — 2-sentence conceptual overview per API group, then signatures, <150 lines

Usage:
    ./scripts/autodoc_variations_r5.py --output-dir /tmp/autodoc-r5
    ./scripts/autodoc_variations_r5.py --variation 1 --output-dir /tmp/autodoc-r5
    ./scripts/autodoc_variations_r5.py --report-only --output-dir /tmp/autodoc-r5
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_docs.graph import ClassInfo, FunctionInfo, RepoGraph, build_repo_graph
from agent_docs.tier2 import _format_sources, _get_callee_context

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# Modules relevant to a dedup task — but the docs don't know that.
# We generate module-level docs for these, as the pipeline would.
TARGET_MODULES = ["marin.processing", "dupekit", "zephyr.dataset"]

# ---------------------------------------------------------------------------
# Ecosystem overview — always prepended (same as MAP.md would provide)
# ---------------------------------------------------------------------------
ECOSYSTEM_OVERVIEW = """\
# Marin Ecosystem

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
# Five MODULE_PROMPT variants — these replace prompts.MODULE_PROMPT
# ---------------------------------------------------------------------------
PROMPTS = {
    1: {
        "description": "Baseline: Purpose/API/Deps/Abstractions/Gotchas (<150 lines)",
        "prompt": (
            """\
You are generating a module reference card for an AI coding agent working on \
the Marin monorepo. The agent will read this doc to understand the module's API \
before reading source code. It needs to know WHAT exists, WHERE it is, and \
what will trip it up — not HOW things work internally.

Given the source code below for module `{module_name}`, produce a markdown \
document with EXACTLY these sections:

## Purpose
2-3 sentences: what this module does and when an agent would touch it.

## Public API
For each public function/class, one line in this exact format:
- `name(signature)` — one-sentence description. Default values matter — \
include them for key parameters. `file_path:line`

Group logically (e.g., "Configuration", "Execution", "Utilities").
For classes, list public methods indented under the class entry.
Skip private/internal items (underscore-prefixed).
Include default parameter values when they encode important choices \
(e.g., `num_perms=286`, `ngram_size=5`).

## Dependencies
Which other Marin modules this imports from, as a flat list:
- `module_name` — what it uses from there.

Only list cross-module dependencies within the monorepo (marin, levanter, \
haliax, fray, rigging, iris, zephyr, dupekit). Skip stdlib and third-party.

## Key Abstractions
The 3-5 most important types/classes and what they represent. One line each.

## Gotchas
Things an agent would get wrong without being told. Non-obvious behaviors, \
implicit contracts, common mistakes. 2-5 bullet points. Be specific — name \
the function or parameter that bites you.

Rules:
- Total output MUST be under 150 lines. Truncate the API surface to the most \
important items if the module has too many public symbols.
- No examples, no code blocks, no tutorials.
- Be specific and actionable, not vague.
- Every API entry MUST include `file_path:line_number` so the agent can jump \
to source.
- If a function is a thin wrapper, say what it wraps instead of redescribing it.
- For functions with important default values, include them in the signature.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""
        ),
    },
    2: {
        "description": "Minimal: signatures only, no prose (<80 lines)",
        "prompt": (
            """\
You are generating a minimal API index for module `{module_name}` in the \
Marin monorepo. An AI coding agent will use this to find function names, \
signatures, and import paths. Keep it extremely concise.

Given the source code below, produce a markdown document with ONLY:

## Signatures
For each public function/class, list it as:
- `import_path.function_name(full_signature) -> return_type` — `file:line`

Include ALL default parameter values in the signature. Skip private items.
For classes, list the class then indent its public methods.

## Constraints
List any parameter constraints or validation rules (e.g., "X must be \
divisible by Y"). One bullet per constraint.

Rules:
- Total output MUST be under 80 lines.
- NO prose, NO explanations, NO examples.
- Signatures must include full type annotations and default values.
- Every entry MUST include `file_path:line_number`.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""
        ),
    },
    3: {
        "description": "Heavy prose: conceptual overview then API (<200 lines)",
        "prompt": (
            """\
You are generating a module reference for an AI coding agent working on the \
Marin monorepo. The agent needs to understand HOW this module works \
conceptually so it can use the APIs correctly without reading source.

Given the source code below for module `{module_name}`, produce a markdown \
document with EXACTLY these sections:

## Overview
A 1-2 paragraph conceptual explanation of what this module does and how its \
pieces fit together. Explain the processing pipeline or data flow if there is \
one. Describe the relationship between the key functions — which ones are \
entry points vs internal helpers, what order they run in, what data flows \
between them. This section should give the agent a mental model of the module.

## Public API
For each public function/class:
- `name(signature)` — 2-3 sentence description explaining what it does, when \
to use it, and what it returns. Include all default parameter values that \
encode important choices. `file_path:line`

Group logically. For classes, list public methods indented under the class.
Skip private/internal items.

## Calling Patterns
For each major entry-point function, describe the typical calling pattern \
in prose (not code): what arguments are required, which have important \
defaults, and what the return value contains. Describe how parameter choices \
affect behavior.

## Dependencies
Which other Marin modules this imports from:
- `module_name` — what it uses and why.

## Gotchas
Non-obvious behaviors, implicit contracts, common mistakes. 3-5 bullets. \
Be specific — name the function or parameter.

Rules:
- Total output MUST be under 200 lines.
- No code blocks or examples — describe everything in prose.
- Be conceptual and explanatory, not just a reference list.
- Every API entry MUST include `file_path:line_number`.
- Include all important default values in signatures.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""
        ),
    },
    4: {
        "description": "Code-focused: signatures in fenced code blocks (<120 lines)",
        "prompt": (
            """\
You are generating a code-oriented reference for module `{module_name}` in \
the Marin monorepo. An AI coding agent will use this to write correct \
function calls. Prioritize showing exact signatures in code format.

Given the source code below, produce a markdown document with these sections:

## Purpose
1 sentence: what this module does.

## API
For each logical group of public functions/classes, use a fenced Python code \
block showing the import and call signature with full type annotations and \
defaults:

```python
from module.submodule import function_name

function_name(
    *,
    param1: type = default,
    param2: type = default,
    ...
) -> return_type
```

After each code block, a 1-sentence description of what the function does. \
`file_path:line`

## Constraints
Parameter validation rules and important relationships between parameters. \
One bullet each.

Rules:
- Total output MUST be under 120 lines.
- Signatures MUST be in fenced Python code blocks with full types and defaults.
- Skip private items (underscore-prefixed).
- Every entry MUST include `file_path:line_number`.
- Minimal prose — let the code speak.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""
        ),
    },
    5: {
        "description": "Hybrid: conceptual group intros + compact signatures (<150 lines)",
        "prompt": (
            """\
You are generating a module reference for an AI coding agent working on the \
Marin monorepo. For each logical group of APIs, the agent needs a brief \
conceptual explanation followed by the signatures.

Given the source code below for module `{module_name}`, produce a markdown \
document with these sections:

## Purpose
2 sentences: what this module does and when an agent would use it.

## API

Organize the public API into logical groups (e.g., by feature area or \
processing stage). For each group:

### Group Name
2-3 sentences explaining what this group of functions/classes does, how they \
relate to each other, and what the typical usage pattern is. Explain any \
important parameter defaults and why they have those values.

Then list each function/class:
- `name(signature)` — one-sentence description. `file_path:line`

Include all default parameter values in signatures.
Skip private/internal items.

## Dependencies
- `module_name` — what it uses from there.

## Gotchas
2-5 bullets on non-obvious behaviors. Be specific.

Rules:
- Total output MUST be under 150 lines.
- No code blocks or examples.
- Every API entry MUST include `file_path:line_number`.
- The group introductions are the most important part — they should give \
the agent enough context to use the APIs correctly.
- Include important default values in signatures.

{callee_context}

## Source code for module `{module_name}`:

{sources}
"""
        ),
    },
}

# ---------------------------------------------------------------------------
# Task and review — same as R3/R4
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
# Doc generation using the actual pipeline
# ---------------------------------------------------------------------------


def generate_module_doc_with_prompt(
    graph: RepoGraph,
    mod_name: str,
    module_prompt: str,
    existing_docs: dict[str, str],
    model: str = "sonnet",
) -> str:
    """Generate a single module doc using a custom prompt template."""
    mod = graph.modules[mod_name]
    public_items: list[FunctionInfo | ClassInfo] = [f for f in mod.functions if f.is_public] + [
        c for c in mod.classes if c.is_public
    ]

    if not public_items:
        return f"# {mod_name}\n\nNo public API.\n"

    callee_context = _get_callee_context(mod, existing_docs)
    sources = _format_sources(public_items)

    # Truncate sources if very large (marin.processing is huge)
    if len(sources) > 80_000:
        # Keep only the most relevant items — dedup-related for marin.processing
        sources = _format_sources(public_items[:40])
        if len(sources) > 80_000:
            sources = sources[:80_000] + "\n... (truncated)"

    prompt = module_prompt.format(
        module_name=mod_name,
        callee_context=callee_context,
        sources=sources,
    )

    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        "1.00",
        "--system-prompt",
        "You are a precise documentation generator. Output only what is requested.",
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("Doc generation failed for %s: %s", mod_name, result.stderr[:500])
        return f"# {mod_name}\n\nGeneration failed.\n"

    text = result.stdout.strip()
    # Strip markdown fences if wrapped
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text


def generate_all_module_docs(
    graph: RepoGraph,
    modules: list[str],
    module_prompt: str,
    output_dir: Path,
    model: str = "sonnet",
) -> dict[str, str]:
    """Generate docs for all target modules, returns {mod_name: doc_text}."""
    existing_docs: dict[str, str] = {}
    docs: dict[str, str] = {}

    for mod_name in modules:
        if mod_name not in graph.modules:
            logger.warning("Module %s not found in graph", mod_name)
            continue

        logger.info("  Generating doc for %s...", mod_name)
        doc = generate_module_doc_with_prompt(graph, mod_name, module_prompt, existing_docs, model)
        docs[mod_name] = doc
        existing_docs[mod_name] = doc

        # Write to disk
        doc_path = output_dir / f"{mod_name}.md"
        doc_path.write_text(doc.rstrip() + "\n")
        logger.info("  Wrote %s (%d chars)", doc_path, len(doc))

    return docs


# ---------------------------------------------------------------------------
# Ask agent to write script using generated docs
# ---------------------------------------------------------------------------


def ask_to_write_script(context: str, model: str) -> dict:
    """Ask the coding agent to write a fuzzy dedup script."""
    prompt = f"""\
You have access to the following documentation for the Marin monorepo.
Use ONLY this documentation to write the script. Do not invent function names or parameters.

{context}

---

{TASK_PROMPT}
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


# ---------------------------------------------------------------------------
# Run a variation
# ---------------------------------------------------------------------------


def run_variation(
    num: int,
    graph: RepoGraph,
    output_dir: Path,
    gen_model: str,
    review_model: str,
    doc_gen_model: str,
) -> dict:
    """Run one variation end-to-end: generate docs, build context, write script, review."""
    var = PROMPTS[num]
    desc = var["description"]
    module_prompt = var["prompt"]

    var_dir = output_dir / f"variation_{num:02d}"
    docs_dir = var_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== V%d: %s ===", num, desc)

    # Step 1: Generate module docs using the pipeline
    logger.info("  Generating module docs (model=%s)...", doc_gen_model)
    docs = generate_all_module_docs(graph, TARGET_MODULES, module_prompt, docs_dir, model=doc_gen_model)

    # Step 2: Build context from generated docs
    context_parts = [ECOSYSTEM_OVERVIEW]
    for mod_name in TARGET_MODULES:
        if mod_name in docs:
            context_parts.append(f"---\n\n{docs[mod_name]}")

    context = "\n\n".join(context_parts)
    (var_dir / "context.txt").write_text(context + "\n")
    logger.info("  Total context: %d chars from %d modules", len(context), len(docs))

    # Step 3: Ask haiku to write the script
    logger.info("  Asking %s to write script...", gen_model)
    gen_result = ask_to_write_script(context, gen_model)
    (var_dir / "dedup_script.py").write_text(gen_result["script"] + "\n")
    logger.info(
        "  Gen cost: $%.4f (%d in / %d out)",
        gen_result["cost_usd"],
        gen_result["input_tokens"],
        gen_result["output_tokens"],
    )

    # Step 4: Review
    logger.info("  Reviewing with %s...", review_model)
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
        "doc_gen_model": doc_gen_model,
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
        "per_module_doc_chars": {m: len(d) for m, d in docs.items()},
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
        "# Autodoc Variation Experiment — Round 5\n",
        "## Setup",
        "- **Doc generator model**: sonnet (generates module-level docs from source)",
        "- **Coding agent**: haiku (writes script from generated docs)",
        "- **Reviewer**: sonnet",
        "- **Task**: Write a fuzzy dedup script",
        "- **Key constraint**: Docs are generated at module/package level by the actual",
        "  doc pipeline. NO task-specific documentation. Only the MODULE_PROMPT varies.",
        "- **Modules documented**: marin.processing, dupekit, zephyr.dataset\n",
        "## Results\n",
        "| # | Variation | Context | Gen Cost | Score | Acc |",
        "|---|-----------|---------|----------|-------|-----|",
    ]

    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(
            f"| {r['variation']} | {r['description']} | {r['context_chars']:,} | ${r['generation_cost_usd']:.4f} | {r['total_correct']}/{r['total_criteria']} | {r['accuracy']:.0%} |"
        )

    lines.append("\n## Per-module doc sizes\n")
    lines.append("| # | marin.processing | dupekit | zephyr.dataset | Total |")
    lines.append("|---|-----------------|---------|----------------|-------|")
    for r in sorted(results, key=lambda x: x["variation"]):
        sizes = r.get("per_module_doc_chars", {})
        mp = sizes.get("marin.processing", 0)
        dk = sizes.get("dupekit", 0)
        zd = sizes.get("zephyr.dataset", 0)
        lines.append(f"| {r['variation']} | {mp:,} | {dk:,} | {zd:,} | {mp + dk + zd:,} |")

    lines.append("\n## Analysis\n")

    perfect = [r for r in results if r["accuracy"] == 1.0]
    if perfect:
        best = min(perfect, key=lambda r: r["generation_cost_usd"])
        lines.append(
            f"### Best perfect score\n**V{best['variation']}** ({best['description']}): "
            f"${best['generation_cost_usd']:.4f} at {best['context_chars']:,} chars\n"
        )

    good = [r for r in results if r["accuracy"] >= 0.8]
    if good:
        cheapest = min(good, key=lambda r: r["generation_cost_usd"])
        lines.append(
            f"### Best cost-accuracy tradeoff (≥80%)\n**V{cheapest['variation']}** "
            f"({cheapest['description']}): ${cheapest['generation_cost_usd']:.4f} at {cheapest['accuracy']:.0%}\n"
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

    lines.append("\n## Generated Docs (V1 baseline sample)\n")
    v1_docs_dir = output_dir / "variation_01" / "docs"
    if v1_docs_dir.exists():
        for md_file in sorted(v1_docs_dir.glob("*.md")):
            lines.append(f"### {md_file.stem}")
            lines.append(f"```markdown\n{md_file.read_text().strip()}\n```\n")

    lines.append("\n## Scripts\n")
    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(f"### V{r['variation']}: {r['description']}")
        lines.append(
            f"*Context: {r['context_chars']:,} | Cost: ${r['generation_cost_usd']:.4f} | "
            f"Score: {r['total_correct']}/{r['total_criteria']}*\n"
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
    parser.add_argument("--gen-model", type=str, default="haiku", help="Coding agent model")
    parser.add_argument("--review-model", type=str, default="sonnet", help="Reviewer model")
    parser.add_argument("--doc-gen-model", type=str, default="sonnet", help="Doc generation model")
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

    logger.info("Building repo graph...")
    graph = build_repo_graph(REPO_ROOT)

    to_run = [args.variation] if args.variation else list(range(1, 6))

    results = []
    for num in to_run:
        results.append(run_variation(num, graph, output_dir, args.gen_model, args.review_model, args.doc_gen_model))

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
