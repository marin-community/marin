# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Module doc generator: produces per-module markdown reference cards
directly from source code via the claude CLI."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_docs.cache import DocCache, _combined_source_hash, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import ClassInfo, FunctionInfo, ModuleInfo, RepoGraph
from agent_docs.prompts import MODULE_PROMPT

logger = logging.getLogger(__name__)

OUTPUT_DIR = "docs/agent/modules"
MAX_SOURCE_PER_BATCH = 50_000  # chars of source per LLM call


def _format_sources(items: list[FunctionInfo | ClassInfo]) -> str:
    """Format function/class sources for the prompt."""
    parts: list[str] = []
    for item in items:
        header = f"### {item.qualified_name} ({item.file_path}:{item.line_number})"
        parts.append(f"{header}\n```{item.language}\n{item.source}\n```\n")
    return "\n".join(parts)


def _get_callee_context(mod: ModuleInfo, existing_docs: dict[str, str]) -> str:
    """Build callee context from already-generated docs of imported modules."""
    context_parts: list[str] = []
    for dep_mod_name in sorted(mod.imports_from):
        if dep_mod_name in existing_docs:
            # Include just the Purpose and Key Abstractions sections
            doc = existing_docs[dep_mod_name]
            # Truncate to keep prompt reasonable
            if len(doc) > 2000:
                doc = doc[:2000] + "\n... (truncated)"
            context_parts.append(f"### {dep_mod_name}\n{doc}")

    if not context_parts:
        return ""

    return "## Context from imported modules:\n\n" + "\n\n".join(context_parts)


def _topo_sort_modules(graph: RepoGraph) -> list[str]:
    """Topological sort of modules by import dependencies (leaves first)."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        mod = graph.modules.get(name)
        if mod is None:
            return
        for dep in mod.imports_from:
            for mod_name in graph.modules:
                if mod_name == dep or mod_name.startswith(f"{dep}."):
                    visit(mod_name)
        order.append(name)

    for name in graph.modules:
        visit(name)

    return order


def generate_module_docs(
    graph: RepoGraph,
    cache: DocCache,
    stale_modules: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> set[str]:
    """Generate module reference card markdown docs for all stale modules.

    Processes modules in topological order so that callee docs are available
    as context when generating caller docs.

    Returns the set of module names that were updated.
    """
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    module_order = _topo_sort_modules(graph)

    # Pre-load existing docs for non-stale modules (callee context)
    existing_docs: dict[str, str] = {}
    for mod_name in module_order:
        if mod_name not in stale_modules:
            md_path = output_dir / f"{mod_name}.md"
            if md_path.exists():
                existing_docs[mod_name] = md_path.read_text()

    updated: set[str] = set()

    for mod_name in module_order:
        mod = graph.modules[mod_name]
        if mod_name not in stale_modules:
            continue

        public_items: list[FunctionInfo | ClassInfo] = [f for f in mod.functions if f.is_public] + [
            c for c in mod.classes if c.is_public
        ]

        if not public_items:
            logger.info("Skipping %s (no public items)", mod_name)
            continue

        if dry_run:
            logger.info("[dry-run] Would generate module doc for %s (%d items)", mod_name, len(public_items))
            continue

        logger.info("Generating module doc for %s (%d items)", mod_name, len(public_items))

        sources = _format_sources(public_items)
        callee_context = _get_callee_context(mod, existing_docs)

        # If source is too large, truncate (the LLM prompt has limits)
        if len(sources) > MAX_SOURCE_PER_BATCH:
            logger.warning("Truncating source for %s (%d chars -> %d)", mod_name, len(sources), MAX_SOURCE_PER_BATCH)
            sources = sources[:MAX_SOURCE_PER_BATCH] + "\n\n... (source truncated, additional items omitted)"

        prompt = MODULE_PROMPT.format(
            module_name=mod_name,
            callee_context=callee_context,
            sources=sources,
        )

        try:
            response = generate(prompt, model=model)
            md_path = output_dir / f"{mod_name}.md"
            md_path.write_text(response.rstrip() + "\n")

            existing_docs[mod_name] = response
            updated.add(mod_name)

            combined_hash = _combined_source_hash(public_items)
            doc_hash = hash_text(response)
            cache.update(mod_name, combined_hash, doc_hash, tier=2)

        except Exception:
            logger.error("Failed to generate module doc for %s", mod_name, exc_info=True)

    return updated
