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
"""Run autodoc experiment across 10 structural variations.

Each variation constructs a different context format from the same underlying
module docs, then asks claude to answer the test question using only that context.
Measures cost, tokens, and answer correctness.

Usage:
    ./scripts/autodoc_variations.py --output-dir /tmp/autodoc-variations
    ./scripts/autodoc_variations.py --variation 3 --output-dir /tmp/autodoc-variations
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_docs.cache import DocCache
from agent_docs.graph import build_repo_graph
from agent_docs.tier2 import generate_module_docs

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

TEST_QUESTION = """\
Explain how to compute duplicate documents from a list of parquet files using \
Marin's zephyr pipeline. Be concise but specific: which functions to call, \
what parameters matter (including defaults), and how the pipeline stages work. \
Include the import paths. Keep it under 200 words.\
"""

# Ground truth facts for scoring answers.
GROUND_TRUTH = {
    "dedup_fuzzy_document": "function name",
    "num_perms=286": "correct default (not 128)",
    "num_bands=26": "correct default",
    "ngram_size=5": "correct default",
    "seed=42": "correct default",
    "connected_components": "clustering step",
    "dupekit": "Rust FFI library",
    "marin.processing": "import path",
}

# 3-module set (focused)
MODULES_3 = ["marin.processing", "dupekit", "zephyr.dataset"]

# 7-module set (broad)
MODULES_7 = [
    "marin.processing",
    "dupekit",
    "zephyr.dataset",
    "zephyr.execution",
    "zephyr.plan",
    "zephyr.readers",
    "zephyr.writers",
]

# Source files for raw-source variation
DEDUP_SOURCE_FILES = [
    "lib/marin/src/marin/processing/classification/deduplication/fuzzy.py",
    "lib/marin/src/marin/processing/classification/deduplication/connected_components.py",
    "rust/dupekit/dupekit/__init__.pyi",
]


def _ensure_module_docs(modules: list[str], docs_dir: Path, model: str) -> dict[str, str]:
    """Generate module docs if not already present. Returns {mod_name: doc_text}."""
    modules_dir = docs_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    missing = []
    for mod in modules:
        md_path = modules_dir / f"{mod}.md"
        if md_path.exists():
            existing[mod] = md_path.read_text()
        else:
            missing.append(mod)

    if missing:
        logger.info("Generating docs for %d missing modules: %s", len(missing), missing)
        graph = build_repo_graph(REPO_ROOT)
        cache = DocCache()
        available = {m for m in missing if m in graph.modules}

        import agent_docs.tier2 as tier2_mod

        original = tier2_mod.OUTPUT_DIR
        tier2_mod.OUTPUT_DIR = str(modules_dir)
        try:
            generate_module_docs(graph, cache, available, REPO_ROOT, model=model)
        finally:
            tier2_mod.OUTPUT_DIR = original

        for mod in missing:
            md_path = modules_dir / f"{mod}.md"
            if md_path.exists():
                existing[mod] = md_path.read_text()

    return existing


def _extract_section(doc: str, section: str) -> str:
    """Extract a specific ## section from a markdown doc."""
    pattern = rf"^## {re.escape(section)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, doc, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _build_context_variation_1(docs: dict[str, str]) -> str:
    """Full module docs, 3 modules."""
    parts = [f"# Module: {name}\n\n{doc}" for name, doc in sorted(docs.items()) if name in MODULES_3]
    return "\n\n---\n\n".join(parts)


def _build_context_variation_2(docs: dict[str, str]) -> str:
    """Full module docs, 7 modules."""
    parts = [f"# Module: {name}\n\n{doc}" for name, doc in sorted(docs.items())]
    return "\n\n---\n\n".join(parts)


def _build_context_variation_3(docs: dict[str, str]) -> str:
    """API-only: strip everything except Public API section."""
    parts = []
    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        api = _extract_section(docs[name], "Public API")
        if api:
            parts.append(f"# Module: {name}\n\n## Public API\n{api}")
    return "\n\n---\n\n".join(parts)


def _build_context_variation_4(docs: dict[str, str]) -> str:
    """Signature index: one line per function, no prose."""
    lines = []
    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        api = _extract_section(docs[name], "Public API")
        if not api:
            continue
        lines.append(f"## {name}")
        for line in api.splitlines():
            line = line.strip()
            if line.startswith("- `"):
                lines.append(line)
    return "\n".join(lines)


def _build_context_variation_5(docs: dict[str, str]) -> str:
    """Compressed: Purpose + first 30 API lines + Gotchas only."""
    parts = []
    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        purpose = _extract_section(docs[name], "Purpose")
        api = _extract_section(docs[name], "Public API")
        gotchas = _extract_section(docs[name], "Gotchas")
        api_lines = api.splitlines()[:30]
        section = f"# {name}\n{purpose}\n\n## API\n" + "\n".join(api_lines)
        if gotchas:
            section += f"\n\n## Gotchas\n{gotchas}"
        parts.append(section)
    return "\n\n---\n\n".join(parts)


def _build_context_variation_6(docs: dict[str, str]) -> str:
    """Header + references: short summary per module, pointer to 'read file X'."""
    lines = ["# Module Index\n"]
    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        purpose = _extract_section(docs[name], "Purpose")
        first_sentence = purpose.split(".")[0] + "." if purpose else "No description."
        lines.append(f"- **{name}**: {first_sentence} → `docs/agent/modules/{name}.md`")

    lines.append("\n# Key Details\n")
    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        gotchas = _extract_section(docs[name], "Gotchas")
        deps = _extract_section(docs[name], "Dependencies")
        if gotchas:
            lines.append(f"## {name} Gotchas\n{gotchas}\n")
        if deps:
            lines.append(f"## {name} Dependencies\n{deps}\n")
    return "\n".join(lines)


def _build_context_variation_7(docs: dict[str, str]) -> str:
    """Curated snippet: hand-picked dedup-relevant functions only."""
    curated = """# Deduplication API Reference

## marin.processing — Fuzzy Dedup
- `dedup_fuzzy_document(*, input_paths, output_path, text_field="text", filetypes=None, fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, fuzzy_minhash_seed=42, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict` — Full fuzzy document dedup pipeline. Validates num_perms % num_bands == 0, runs dupekit MinHash/LSH, then connected_components. `classification/deduplication/fuzzy.py:27`
- `dedup_exact_document(*, input_paths, output_path, text_field="text", filetypes=None, max_parallelism, worker_resources=None, coordinator_resources=None) -> dict` — Exact hash dedup via xxHash3-128. `classification/deduplication/exact.py:186`
- `connected_components(ds, ctx, *, output_dir, max_iterations=10, preserve_singletons=True) -> tuple[bool, Sequence[str]]` — Hash-to-Min connected components over LSH buckets. Returns (converged, output_files). `classification/deduplication/connected_components.py:51`

## dupekit — Rust FFI
- `clean_text(arr: StringArray) -> StringArray` — Lowercase, strip punctuation, normalize whitespace. `rust/dupekit/src/minhash_ops.rs:15`
- `compute_minhash(arr: StringArray, num_perms: int, ngram_size: int, seed: int) -> ListArray[uint64]` — MinHash signatures using character n-grams. `rust/dupekit/src/minhash_ops.rs:40`
- `compute_lsh(input_col: ListArray, num_bands: int) -> ListArray[uint64]` — LSH band bucket hashes. num_perms must be divisible by num_bands. `rust/dupekit/src/minhash_ops.rs:99`
- `Transformation.CleanText(input_col, output_col)` — Pipeline step for text cleaning.
- `Transformation.MinHash(input_col, output_col, num_perms, ngram_size, seed)` — Pipeline step for MinHash.
- `Transformation.MinHashLSH(input_col, output_col, num_bands)` — Pipeline step for LSH banding.
- `transform(batch: RecordBatch, steps: list[Transformation]) -> RecordBatch` — Execute pipeline steps on a batch. GIL-released. `rust/dupekit/src/pipeline.rs:172`

## Pipeline Flow
1. `dedup_fuzzy_document` collects input files, creates ZephyrContext
2. Builds dupekit pipeline: CleanText → MinHash → MinHashLSH → SelectColumns
3. `Dataset.from_list(file_groups).flat_map(compute_minhash_lsh_batches)` yields {bucket, id} pairs
4. `connected_components` runs Hash-to-Min iterations writing vortex files
5. Results grouped by file, written as vortex attribute files marking duplicates
6. `finalize_dedup` aggregates counters
"""
    return curated


def _build_context_variation_8() -> str:
    """Raw source only: actual Python source files, no generated docs."""
    parts = []
    for rel_path in DEDUP_SOURCE_FILES:
        full_path = REPO_ROOT / rel_path
        if full_path.exists():
            content = full_path.read_text()
            parts.append(f"# File: {rel_path}\n```python\n{content}\n```")
    return "\n\n---\n\n".join(parts)


def _build_context_variation_9(docs: dict[str, str]) -> str:
    """Single merged doc: all 3 modules merged into one cohesive document."""
    all_purposes = []
    all_apis = []
    all_gotchas = []
    all_deps = []
    all_abstractions = []

    for name in sorted(docs):
        if name not in MODULES_3:
            continue
        purpose = _extract_section(docs[name], "Purpose")
        api = _extract_section(docs[name], "Public API")
        gotchas = _extract_section(docs[name], "Gotchas")
        deps = _extract_section(docs[name], "Dependencies")
        abstractions = _extract_section(docs[name], "Key Abstractions")

        if purpose:
            all_purposes.append(f"**{name}**: {purpose}")
        if api:
            all_apis.append(f"### {name}\n{api}")
        if gotchas:
            all_gotchas.append(f"### {name}\n{gotchas}")
        if deps:
            all_deps.append(f"### {name}\n{deps}")
        if abstractions:
            all_abstractions.append(f"### {name}\n{abstractions}")

    sections = [
        "# Marin Module Reference\n",
        "## Purpose\n" + "\n\n".join(all_purposes),
        "## Public API\n" + "\n\n".join(all_apis),
        "## Key Abstractions\n" + "\n\n".join(all_abstractions),
        "## Dependencies\n" + "\n\n".join(all_deps),
        "## Gotchas\n" + "\n\n".join(all_gotchas),
    ]
    return "\n\n".join(sections)


def _build_context_variation_10() -> str:
    """No context baseline."""
    return ""


VARIATION_DESCRIPTIONS = {
    1: "Full module docs (3 modules)",
    2: "Full module docs (7 modules)",
    3: "API-only (3 modules, no prose/gotchas)",
    4: "Signature index (one line per function)",
    5: "Compressed (purpose + 30 API lines + gotchas)",
    6: "Header + references (summary + gotchas only)",
    7: "Curated snippet (hand-picked dedup functions)",
    8: "Raw source only (no generated docs)",
    9: "Single merged doc (3 modules combined)",
    10: "No context baseline",
}


def build_context(variation: int, docs_3: dict[str, str], docs_7: dict[str, str]) -> str:
    """Build the context string for a given variation number."""
    match variation:
        case 1:
            return _build_context_variation_1(docs_3)
        case 2:
            return _build_context_variation_2(docs_7)
        case 3:
            return _build_context_variation_3(docs_3)
        case 4:
            return _build_context_variation_4(docs_3)
        case 5:
            return _build_context_variation_5(docs_3)
        case 6:
            return _build_context_variation_6(docs_3)
        case 7:
            return _build_context_variation_7(docs_3)
        case 8:
            return _build_context_variation_8()
        case 9:
            return _build_context_variation_9(docs_3)
        case 10:
            return _build_context_variation_10()
        case _:
            raise ValueError(f"Unknown variation: {variation}")


def ask_question(context: str, question: str, model: str) -> dict:
    """Ask claude a question with the given context. Returns answer + cost info."""
    if context:
        prompt = f"""\
You have access to the following documentation for the Marin monorepo. \
Use ONLY this documentation to answer the question. Do not make up function \
names or parameters that are not in the docs.

{context}

---

Question: {question}

Answer concisely in one paragraph. Be specific about function names, parameters, \
and import paths.\
"""
    else:
        prompt = f"""\
Question: {question}

Answer concisely in one paragraph. Be specific about function names, parameters, \
and import paths. If you don't know specific details, say so.\
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
        "You are answering questions about the Marin codebase. Be precise and specific.",
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return {
            "answer": f"ERROR: {result.stderr[:500]}",
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "context_chars": len(context),
        }

    data = json.loads(result.stdout)
    return {
        "answer": data.get("result", ""),
        "cost_usd": data.get("total_cost_usd", 0),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "cache_creation_tokens": data.get("usage", {}).get("cache_creation_input_tokens", 0),
        "cache_read_tokens": data.get("usage", {}).get("cache_read_input_tokens", 0),
        "context_chars": len(context),
    }


def score_answer(answer: str) -> dict:
    """Score an answer against ground truth facts. Returns {fact: found_bool}."""
    answer_lower = answer.lower()
    scores = {}
    for fact, description in GROUND_TRUTH.items():
        # Flexible matching: check for the key value even in different formats
        if "=" in fact:
            key, val = fact.split("=")
            found = val in answer_lower or f"{key}={val}" in answer_lower
        else:
            found = fact.lower() in answer_lower
        scores[fact] = {"found": found, "description": description}
    return scores


def run_variation(
    variation: int,
    docs_3: dict[str, str],
    docs_7: dict[str, str],
    output_dir: Path,
    model: str,
) -> dict:
    """Run a single variation experiment."""
    var_dir = output_dir / f"variation_{variation:02d}"
    var_dir.mkdir(parents=True, exist_ok=True)

    desc = VARIATION_DESCRIPTIONS[variation]
    logger.info("=== Variation %d: %s ===", variation, desc)

    context = build_context(variation, docs_3, docs_7)
    logger.info("Context size: %d chars", len(context))

    answer_info = ask_question(context, TEST_QUESTION, model)
    scores = score_answer(answer_info["answer"])

    facts_correct = sum(1 for s in scores.values() if s["found"])
    facts_total = len(scores)

    result = {
        "variation": variation,
        "description": desc,
        "context_chars": len(context),
        **answer_info,
        "facts_correct": facts_correct,
        "facts_total": facts_total,
        "accuracy": facts_correct / facts_total,
        "scores": scores,
    }

    (var_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (var_dir / "answer.txt").write_text(answer_info["answer"] + "\n")
    (var_dir / "context.txt").write_text(context + "\n")

    logger.info(
        "  Cost: $%.4f | Context: %d chars | Tokens: %d in / %d out | Accuracy: %d/%d (%.0f%%)",
        answer_info["cost_usd"],
        len(context),
        answer_info["input_tokens"],
        answer_info["output_tokens"],
        facts_correct,
        facts_total,
        100 * facts_correct / facts_total,
    )

    return result


def compile_report(results: list[dict], output_dir: Path) -> str:
    """Compile a markdown report from all variation results."""
    lines = [
        "# Autodoc Variation Experiment Report\n",
        "## Objective",
        "Minimize agent token cost while retaining correct answers about",
        "Marin's zephyr deduplication pipeline.\n",
        "## Test Question",
        f"> {TEST_QUESTION}\n",
        "## Ground Truth Facts Checked",
    ]
    for fact, desc in GROUND_TRUTH.items():
        lines.append(f"- `{fact}` — {desc}")

    lines.append("\n## Results\n")
    lines.append("| # | Variation | Context | Cost | In Tokens | Out Tokens | Accuracy | Score |")
    lines.append("|---|-----------|---------|------|-----------|------------|----------|-------|")

    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(
            f"| {r['variation']} | {r['description']} | {r['context_chars']:,} chars | "
            f"${r['cost_usd']:.4f} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
            f"{r['accuracy']:.0%} | {r['facts_correct']}/{r['facts_total']} |"
        )

    # Find best cost-accuracy tradeoff
    lines.append("\n## Analysis\n")

    correct_results = [r for r in results if r["accuracy"] >= 0.75]
    if correct_results:
        cheapest = min(correct_results, key=lambda r: r["cost_usd"])
        lines.append("### Best cost-accuracy tradeoff")
        lines.append(
            f"**Variation {cheapest['variation']}** ({cheapest['description']}): "
            f"${cheapest['cost_usd']:.4f} at {cheapest['accuracy']:.0%} accuracy "
            f"({cheapest['context_chars']:,} chars context)\n"
        )

    most_accurate = max(results, key=lambda r: (r["accuracy"], -r["cost_usd"]))
    lines.append("### Most accurate")
    lines.append(
        f"**Variation {most_accurate['variation']}** ({most_accurate['description']}): "
        f"{most_accurate['accuracy']:.0%} accuracy at ${most_accurate['cost_usd']:.4f}\n"
    )

    cheapest_overall = min(results, key=lambda r: r["cost_usd"])
    lines.append("### Cheapest")
    lines.append(
        f"**Variation {cheapest_overall['variation']}** ({cheapest_overall['description']}): "
        f"${cheapest_overall['cost_usd']:.4f} at {cheapest_overall['accuracy']:.0%} accuracy\n"
    )

    # Cost-efficiency ratio
    lines.append("### Cost per correct fact\n")
    lines.append("| # | Variation | $/fact | Accuracy |")
    lines.append("|---|-----------|--------|----------|")
    for r in sorted(results, key=lambda x: x["variation"]):
        cost_per_fact = r["cost_usd"] / max(r["facts_correct"], 1)
        lines.append(f"| {r['variation']} | {r['description']} | ${cost_per_fact:.4f} | {r['accuracy']:.0%} |")

    lines.append("\n## Detailed Answers\n")
    for r in sorted(results, key=lambda x: x["variation"]):
        lines.append(f"### Variation {r['variation']}: {r['description']}")
        lines.append(
            f"*Context: {r['context_chars']:,} chars | Cost: ${r['cost_usd']:.4f} | Accuracy: {r['accuracy']:.0%}*\n"
        )

        answer_file = output_dir / f"variation_{r['variation']:02d}" / "answer.txt"
        if answer_file.exists():
            lines.append(f"> {answer_file.read_text().strip()}\n")

        lines.append("**Fact check:**")
        for fact, info in r["scores"].items():
            mark = "pass" if info["found"] else "MISS"
            lines.append(f"- [{mark}] `{fact}` — {info['description']}")
        lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "report.md"
    report_path.write_text(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run autodoc variation experiments")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--variation", type=int, help="Run only this variation (1-10)")
    parser.add_argument("--model", type=str, default="sonnet")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Compile report from existing results")
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
            result_path = output_dir / f"variation_{v:02d}" / "result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
        report = compile_report(results, output_dir)
        print(report)
        return

    # Generate docs for all needed modules
    docs_dir = output_dir / "generated_docs"
    logger.info("Ensuring module docs exist...")
    docs_3 = _ensure_module_docs(MODULES_3, docs_dir, args.model)
    docs_7 = _ensure_module_docs(MODULES_7, docs_dir, args.model)

    variations = [args.variation] if args.variation else list(range(1, 11))

    results = []
    for v in variations:
        result = run_variation(v, docs_3, docs_7, output_dir, args.model)
        results.append(result)

    # Load any existing results for the full report
    all_results = []
    for v in range(1, 11):
        result_path = output_dir / f"variation_{v:02d}" / "result.json"
        if result_path.exists():
            all_results.append(json.loads(result_path.read_text()))

    if len(all_results) > 1:
        report = compile_report(all_results, output_dir)
        print(report)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
