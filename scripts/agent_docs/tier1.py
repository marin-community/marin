# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 generator: MAP.md — the top-level module index."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_docs.cache import DocCache, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import RepoGraph
from agent_docs.prompts import TIER1_PROMPT

logger = logging.getLogger(__name__)

MODULES_DIR = "docs/agent/modules"
OUTPUT_FILE = "docs/agent/MAP.md"


def generate_tier1(
    graph: RepoGraph,
    cache: DocCache,
    updated_t2: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> bool:
    """Generate MAP.md from all Tier 2 module summaries.

    Returns True if MAP.md was regenerated.
    """
    if not updated_t2:
        logger.info("No Tier 2 updates, skipping Tier 1 generation")
        return False

    modules_dir = repo_root / MODULES_DIR
    if not modules_dir.exists():
        logger.warning("No modules directory found at %s", modules_dir)
        return False

    # Collect all Tier 2 summaries
    summaries: list[str] = []
    for md_file in sorted(modules_dir.glob("*.md")):
        content = md_file.read_text()
        if content.strip():
            mod_name = md_file.stem
            summaries.append(f"# {mod_name}\n\n{content}")

    if not summaries:
        logger.warning("No Tier 2 summaries found")
        return False

    combined = "\n\n---\n\n".join(summaries)

    if dry_run:
        logger.info("[dry-run] Would generate Tier 1 MAP.md from %d module summaries", len(summaries))
        return False

    logger.info("Generating Tier 1 MAP.md from %d module summaries", len(summaries))

    prompt = TIER1_PROMPT.format(module_summaries=combined)

    try:
        response = generate(prompt, model=model, max_budget_usd=0.10)
        output_path = repo_root / OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response)

        doc_hash = hash_text(response)
        source_hash = hash_text(combined)
        cache.update("MAP.md", source_hash, doc_hash, tier=1)

        return True
    except Exception:
        logger.error("Failed to generate Tier 1 MAP.md", exc_info=True)
        return False
