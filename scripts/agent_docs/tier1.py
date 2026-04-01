# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MAP.md generator: the top-level module index loaded into every agent conversation."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_docs.cache import DocCache, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import RepoGraph
from agent_docs.prompts import MAP_PROMPT

logger = logging.getLogger(__name__)

MODULES_DIR = "docs/agent/modules"
OUTPUT_FILE = "docs/agent/MAP.md"


def generate_map(
    graph: RepoGraph,
    cache: DocCache,
    updated_modules: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> bool:
    """Generate MAP.md from all module docs.

    Returns True if MAP.md was regenerated.
    """
    if not updated_modules:
        logger.info("No module docs updated, skipping MAP.md generation")
        return False

    modules_dir = repo_root / MODULES_DIR
    if not modules_dir.exists():
        logger.warning("No modules directory found at %s", modules_dir)
        return False

    # Collect all module doc summaries
    summaries: list[str] = []
    for md_file in sorted(modules_dir.glob("*.md")):
        content = md_file.read_text()
        if content.strip():
            mod_name = md_file.stem
            summaries.append(f"# {mod_name}\n\n{content}")

    if not summaries:
        logger.warning("No module docs found")
        return False

    combined = "\n\n---\n\n".join(summaries)

    if dry_run:
        logger.info("[dry-run] Would generate MAP.md from %d module summaries", len(summaries))
        return False

    logger.info("Generating MAP.md from %d module summaries", len(summaries))

    prompt = MAP_PROMPT.format(module_summaries=combined)

    try:
        response = generate(prompt, model=model, max_budget_usd=0.50)
        output_path = repo_root / OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response.rstrip() + "\n")

        doc_hash = hash_text(response)
        source_hash = hash_text(combined)
        cache.update("MAP.md", source_hash, doc_hash, tier=1)

        return True
    except Exception:
        logger.error("Failed to generate MAP.md", exc_info=True)
        return False
