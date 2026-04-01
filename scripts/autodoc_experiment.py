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
"""Run a single autodoc experiment iteration.

1. Generate docs for specified modules using the current pipeline
2. Ask claude (with only the generated docs as context) to answer a test question
3. Record the answer, token usage, and cost

Usage:
    ./scripts/autodoc_experiment.py --iteration 1 --output-dir /tmp/autodoc-exp
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_docs.cache import DocCache
from agent_docs.graph import build_repo_graph
from agent_docs.tier2 import generate_module_docs

logger = logging.getLogger(__name__)

# Modules relevant to the zephyr dedup question.
# Reduced set for iterations 4+: only the modules needed to answer correctly.
TARGET_MODULES = [
    "marin.processing",
    "dupekit",
    "zephyr.dataset",
]

TEST_QUESTION = """\
Explain how to compute duplicate documents from a list of parquet files using \
Marin's zephyr pipeline. Be concise but specific: which functions to call, \
what parameters matter (including defaults), and how the pipeline stages work. \
Include the import paths. Keep it under 200 words.
"""


def generate_docs_for_modules(
    repo_root: Path,
    modules: list[str],
    output_dir: Path,
    model: str = "sonnet",
) -> dict:
    """Generate docs for specified modules and return cost info."""
    graph = build_repo_graph(repo_root)
    cache = DocCache()

    # Filter to only requested modules that exist
    available = {m for m in modules if m in graph.modules}
    missing = set(modules) - available
    if missing:
        logger.warning("Modules not found in graph: %s", missing)

    # Redirect output to experiment dir
    import agent_docs.tier2 as tier2_mod

    original_output_dir = tier2_mod.OUTPUT_DIR
    tier2_mod.OUTPUT_DIR = str(output_dir / "modules")

    try:
        updated = generate_module_docs(graph, cache, available, repo_root, model=model)
    finally:
        tier2_mod.OUTPUT_DIR = original_output_dir

    return {"modules_generated": sorted(updated), "modules_requested": sorted(modules)}


def ask_question_with_docs(
    docs_dir: Path,
    question: str,
    model: str = "sonnet",
) -> dict:
    """Ask claude a question with only the generated docs as context.

    Returns the answer, token usage, and cost.
    """
    # Collect all generated module docs
    modules_dir = docs_dir / "modules"
    context_parts: list[str] = []
    if modules_dir.exists():
        for md_file in sorted(modules_dir.glob("*.md")):
            content = md_file.read_text()
            if content.strip():
                mod_name = md_file.stem
                context_parts.append(f"# Module: {mod_name}\n\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    context_size = len(context)

    prompt = f"""\
You have access to the following module documentation for the Marin monorepo. \
Use ONLY this documentation to answer the question. Do not make up function \
names or parameters that are not in the docs.

{context}

---

Question: {question}

Answer in exactly one paragraph. Be specific about function names, parameters, \
and import paths.
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
        "You are answering questions about the Marin codebase using only the provided documentation. Be precise and specific.",
    ]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        return {
            "answer": f"ERROR: {result.stderr[:500]}",
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "context_chars": context_size,
        }

    data = json.loads(result.stdout)
    return {
        "answer": data.get("result", ""),
        "cost_usd": data.get("total_cost_usd", 0),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "cache_creation_tokens": data.get("usage", {}).get("cache_creation_input_tokens", 0),
        "cache_read_tokens": data.get("usage", {}).get("cache_read_input_tokens", 0),
        "context_chars": context_size,
    }


def run_experiment(
    iteration: int,
    repo_root: Path,
    output_dir: Path,
    model: str = "sonnet",
) -> dict:
    """Run a single experiment iteration."""
    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Iteration %d ===", iteration)

    # Step 1: Generate docs
    logger.info("Generating docs for %d modules...", len(TARGET_MODULES))
    gen_info = generate_docs_for_modules(repo_root, TARGET_MODULES, iter_dir, model=model)

    # Step 2: Ask question with only generated docs as context
    logger.info("Asking test question...")
    answer_info = ask_question_with_docs(iter_dir, TEST_QUESTION, model=model)

    # Combine results
    result = {
        "iteration": iteration,
        **gen_info,
        **answer_info,
    }

    # Write result
    result_path = iter_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    # Write answer separately for easy reading
    answer_path = iter_dir / "answer.txt"
    answer_path.write_text(answer_info["answer"] + "\n")

    logger.info(
        "Cost: $%.4f | Input tokens: %d | Output tokens: %d",
        answer_info["cost_usd"],
        answer_info["input_tokens"],
        answer_info["output_tokens"],
    )
    logger.info("Answer: %s", answer_info["answer"][:200])

    return result


def main():
    parser = argparse.ArgumentParser(description="Run autodoc experiment iteration")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model to use")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)

    result = run_experiment(args.iteration, repo_root, output_dir, model=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
