#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tree-sitter==0.24.0",
#     "tree-sitter-python==0.23.6",
#     "tree-sitter-rust==0.23.2",
#     "pyyaml>=6.0",
# ]
# ///
"""Generate agent-optimized documentation for Marin.

Produces a two-tier documentation hierarchy in docs/agent/:
  - MAP.md (module index, ~4KB, auto-loaded via @docs/agent/MAP.md in AGENTS.md)
  - modules/*.md (per-module reference cards, ~2-8KB each, read on demand)

Agents navigate: MAP.md → module doc → source code. Two hops.

Uses the claude CLI for LLM generation and content-addressed caching
to skip unchanged modules.

Usage:
    ./scripts/generate_agent_docs.py                    # incremental update
    ./scripts/generate_agent_docs.py --full              # regenerate everything
    ./scripts/generate_agent_docs.py --dry-run           # show what would change
    ./scripts/generate_agent_docs.py --module marin.execution  # single module
    ./scripts/generate_agent_docs.py --stats             # print graph stats only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts/ to path so agent_docs package is importable
sys.path.insert(0, str(Path(__file__).parent))

from agent_docs.cache import DocCache, compute_stale_modules, load_cache, save_cache
from agent_docs.graph import build_repo_graph, print_graph_stats
from agent_docs.tier1 import generate_map
from agent_docs.tier2 import generate_module_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agent-optimized docs for Marin")
    parser.add_argument("--full", action="store_true", help="Ignore cache, regenerate everything")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be regenerated")
    parser.add_argument("--module", type=str, help="Regenerate only a specific module")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model to use")
    parser.add_argument("--stats", action="store_true", help="Print graph stats and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent
    logging.info("Building dependency graph...")
    graph = build_repo_graph(repo_root)

    if args.stats:
        print_graph_stats(graph)
        return

    cache = DocCache() if args.full else load_cache(repo_root)

    if args.module:
        if args.module not in graph.modules:
            print(f"Module '{args.module}' not found. Available: {sorted(graph.modules.keys())}")
            sys.exit(1)
        stale = {args.module}
    else:
        stale = compute_stale_modules(graph, cache)

    if not stale:
        logging.info("Nothing to regenerate — all modules up to date.")
        return

    logging.info("Stale modules (%d): %s", len(stale), ", ".join(sorted(stale)))

    # Two-step pipeline: module docs → MAP.md
    updated_modules = generate_module_docs(graph, cache, stale, repo_root, model=args.model, dry_run=args.dry_run)
    generate_map(graph, cache, updated_modules, repo_root, model=args.model, dry_run=args.dry_run)

    if not args.dry_run:
        save_cache(cache, repo_root)
        logging.info("Done. Updated %d module docs.", len(updated_modules))


if __name__ == "__main__":
    main()
