#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click>=8.0",
#     "tree-sitter==0.24.0",
#     "tree-sitter-python==0.23.6",
#     "tree-sitter-rust==0.23.2",
#     "pyyaml>=6.0",
# ]
# ///
"""Generate agent-optimized documentation for Marin.

Produces a two-tier documentation hierarchy in docs/agent/:
  - packages/*.md (per-package reference cards, ~2-4KB each, read on demand)
  - MAP.md (package index, regenerated from package docs)

Packages are grouped by filesystem directory, giving natural sub-package
boundaries like marin.processing.classification.deduplication.
"""

import logging
import sys
from pathlib import Path

import click

# Add scripts/ to path so agent_docs package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.generate_index import generate_map
from agent_docs.generate_package import generate_package_docs
from agent_docs.packages import PackageInfo, discover_packages

logger = logging.getLogger(__name__)


def _print_package_stats(packages: dict[str, PackageInfo]) -> None:
    total_funcs = 0
    total_classes = 0
    click.echo(f"\n{'Package':<60} {'Funcs':>6} {'Classes':>8} {'Files':>6}")
    click.echo("-" * 84)
    for name in sorted(packages):
        pkg = packages[name]
        pub_funcs = sum(1 for f in pkg.functions if f.is_public)
        pub_classes = sum(1 for c in pkg.classes if c.is_public)
        total_funcs += pub_funcs
        total_classes += pub_classes
        if pub_funcs + pub_classes > 0:
            click.echo(f"{name:<60} {pub_funcs:>6} {pub_classes:>8} {len(pkg.file_paths):>6}")
    click.echo("-" * 84)
    click.echo(f"{'TOTAL':<60} {total_funcs:>6} {total_classes:>8}")
    click.echo(f"\n{len(packages)} packages discovered")


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would be regenerated.")
@click.option("--package", type=str, default=None, help="Generate only a specific package.")
@click.option("--model", type=str, default="sonnet", help="Model for aggregation and direct generation.")
@click.option("--summary-model", type=str, default="haiku", help="Model for per-file summaries (large packages).")
@click.option("--stats", is_flag=True, help="Print package stats and exit.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    dry_run: bool,
    package: str | None,
    model: str,
    summary_model: str,
    stats: bool,
    verbose: bool,
) -> None:
    """Generate agent-optimized docs for Marin."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent.parent
    logger.info("Discovering packages...")
    packages = discover_packages(repo_root)

    if stats:
        _print_package_stats(packages)
        return

    if package:
        if package not in packages:
            click.echo(f"Package '{package}' not found. Available:")
            for name in sorted(packages):
                pkg = packages[name]
                pub = sum(1 for f in pkg.functions if f.is_public) + sum(1 for c in pkg.classes if c.is_public)
                if pub > 0:
                    click.echo(f"  {name} ({pub} public items)")
            raise SystemExit(1)
        targets = {package}
    else:
        targets = {
            name
            for name, pkg in packages.items()
            if any(f.is_public for f in pkg.functions) or any(c.is_public for c in pkg.classes)
        }

    if not targets:
        logger.info("No packages with public items found.")
        return

    logger.info("Target packages (%d): %s", len(targets), ", ".join(sorted(targets)))

    updated = generate_package_docs(
        packages, targets, repo_root, model=model, summary_model=summary_model, dry_run=dry_run
    )
    generate_map(updated, repo_root, model=model, dry_run=dry_run)

    if not dry_run:
        logger.info("Done. Generated %d package docs.", len(updated))


if __name__ == "__main__":
    main()
