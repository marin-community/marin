# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MAP.md generator: the top-level package index loaded into every agent conversation."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_docs.cache import DocCache, hash_text
from agent_docs.claude_cli import generate
from agent_docs.prompts import MAP_PROMPT

logger = logging.getLogger(__name__)

PACKAGES_DIR = "docs/agent/packages"
OUTPUT_FILE = "docs/agent/MAP.md"


def generate_map(
    cache: DocCache,
    updated_packages: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> bool:
    """Generate MAP.md from all package docs.

    Returns True if MAP.md was regenerated.
    """
    if not updated_packages:
        logger.info("No package docs updated, skipping MAP.md generation")
        return False

    packages_dir = repo_root / PACKAGES_DIR
    if not packages_dir.exists():
        logger.warning("No packages directory found at %s", packages_dir)
        return False

    summaries: list[str] = []
    for md_file in sorted(packages_dir.glob("*.md")):
        content = md_file.read_text()
        if content.strip():
            pkg_name = md_file.stem
            summaries.append(f"# {pkg_name}\n\n{content}")

    if not summaries:
        logger.warning("No package docs found")
        return False

    combined = "\n\n---\n\n".join(summaries)

    if dry_run:
        logger.info("[dry-run] Would generate MAP.md from %d package summaries", len(summaries))
        return False

    logger.info("Generating MAP.md from %d package summaries", len(summaries))

    prompt = MAP_PROMPT.format(package_summaries=combined)

    response = generate(prompt, model=model, max_budget_usd=0.50)
    output_path = repo_root / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.rstrip() + "\n")

    doc_hash = hash_text(response)
    source_hash = hash_text(combined)
    cache.update("MAP.md", source_hash, doc_hash, tier=1)

    return True
