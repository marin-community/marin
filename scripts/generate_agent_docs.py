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
  - MAP.md (package index, ~4KB, auto-loaded via @docs/agent/MAP.md in AGENTS.md)
  - packages/*.md (per-package reference cards, ~2-4KB each, read on demand)

Packages are grouped by filesystem directory (not depth-2 modules), giving
natural sub-package boundaries like marin.processing.classification.deduplication
instead of the overly broad marin.processing.

Agents navigate: MAP.md → package doc → source code. Two hops.

Uses the claude CLI for LLM generation and content-addressed caching
to skip unchanged packages.

Usage:
    ./scripts/generate_agent_docs.py                    # incremental update
    ./scripts/generate_agent_docs.py --full              # regenerate everything
    ./scripts/generate_agent_docs.py --dry-run           # show what would change
    ./scripts/generate_agent_docs.py --package marin.processing.classification.deduplication
    ./scripts/generate_agent_docs.py --stats             # print package stats only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts/ to path so agent_docs package is importable
sys.path.insert(0, str(Path(__file__).parent))

from agent_docs.cache import DocCache, load_cache, save_cache
from agent_docs.packages import PackageInfo, compute_stale_packages, discover_packages
from agent_docs.tier1 import generate_map
from agent_docs.tier2 import generate_package_docs


def print_package_stats(packages: dict[str, PackageInfo]) -> None:
    """Print statistics about discovered packages."""
    total_funcs = 0
    total_classes = 0
    print(f"\n{'Package':<60} {'Funcs':>6} {'Classes':>8} {'Files':>6}")
    print("-" * 84)
    for name in sorted(packages):
        pkg = packages[name]
        pub_funcs = sum(1 for f in pkg.functions if f.is_public)
        pub_classes = sum(1 for c in pkg.classes if c.is_public)
        total_funcs += pub_funcs
        total_classes += pub_classes
        if pub_funcs + pub_classes > 0:
            print(f"{name:<60} {pub_funcs:>6} {pub_classes:>8} {len(pkg.file_paths):>6}")
    print("-" * 84)
    print(f"{'TOTAL':<60} {total_funcs:>6} {total_classes:>8}")
    print(f"\n{len(packages)} packages discovered")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agent-optimized docs for Marin")
    parser.add_argument("--full", action="store_true", help="Ignore cache, regenerate everything")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be regenerated")
    parser.add_argument("--package", type=str, help="Regenerate only a specific package")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model to use")
    parser.add_argument("--stats", action="store_true", help="Print package stats and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent
    logging.info("Discovering packages...")
    packages = discover_packages(repo_root)

    if args.stats:
        print_package_stats(packages)
        return

    cache = DocCache() if args.full else load_cache(repo_root)

    if args.package:
        if args.package not in packages:
            print(f"Package '{args.package}' not found. Available:")
            for name in sorted(packages):
                pkg = packages[name]
                pub = sum(1 for f in pkg.functions if f.is_public) + sum(1 for c in pkg.classes if c.is_public)
                if pub > 0:
                    print(f"  {name} ({pub} public items)")
            sys.exit(1)
        stale = {args.package}
    else:
        stale = compute_stale_packages(packages, cache)

    if not stale:
        logging.info("Nothing to regenerate — all packages up to date.")
        return

    logging.info("Stale packages (%d): %s", len(stale), ", ".join(sorted(stale)))

    updated = generate_package_docs(packages, cache, stale, repo_root, model=args.model, dry_run=args.dry_run)
    generate_map(None, cache, updated, repo_root, model=args.model, dry_run=args.dry_run)

    if not args.dry_run:
        save_cache(cache, repo_root)
        logging.info("Done. Updated %d package docs.", len(updated))


if __name__ == "__main__":
    main()
