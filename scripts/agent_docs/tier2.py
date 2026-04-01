# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 generator: per-module reference card markdown docs."""

from __future__ import annotations

import logging
from pathlib import Path


from agent_docs.cache import DocCache, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import RepoGraph
from agent_docs.prompts import TIER2_PROMPT

logger = logging.getLogger(__name__)

OUTPUT_DIR = "docs/agent/modules"
API_DIR = "docs/agent/api"


def generate_tier2(
    graph: RepoGraph,
    cache: DocCache,
    stale_modules: set[str],
    updated_t3: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> set[str]:
    """Generate Tier 2 markdown docs for modules whose Tier 3 changed.

    Returns the set of module names that were updated.
    """
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    api_dir = repo_root / API_DIR

    # A module needs Tier 2 regen if its Tier 3 was updated
    needs_regen = stale_modules & updated_t3
    updated: set[str] = set()

    for mod_name in sorted(needs_regen):
        mod = graph.modules.get(mod_name)
        if mod is None:
            continue

        # Load this module's Tier 3 YAML
        yaml_path = api_dir / f"{mod_name}.yaml"
        if not yaml_path.exists():
            logger.info("Skipping Tier 2 for %s (no Tier 3 YAML)", mod_name)
            continue

        api_docs = yaml_path.read_text()
        if not api_docs.strip():
            continue

        if dry_run:
            logger.info("[dry-run] Would generate Tier 2 for %s", mod_name)
            continue

        logger.info("Generating Tier 2 for %s", mod_name)

        prompt = TIER2_PROMPT.format(
            module_name=mod_name,
            api_docs=api_docs,
        )

        try:
            response = generate(prompt, model=model)
            md_path = output_dir / f"{mod_name}.md"
            md_path.write_text(response)
            updated.add(mod_name)

            doc_hash = hash_text(response)
            # Use the Tier 3 doc hash as the source hash for Tier 2 cache
            t3_hash = hash_text(api_docs)
            cache.update(mod_name + ":tier2", t3_hash, doc_hash, tier=2)

        except Exception:
            logger.error("Failed to generate Tier 2 for %s", mod_name, exc_info=True)

    return updated
