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


MERGE_PROMPT = """\
You are merging multiple partial module reference cards into one cohesive \
document for module `{module_name}`. Each partial doc covers a subset of the \
module's API.

Combine them into a single document with the same sections as the originals \
(Purpose, Public API, Dependencies, Key Abstractions, Gotchas). Deduplicate \
and merge — do not repeat entries. Keep total output under 8KB. Preserve all \
`file_path:line` references exactly.

## Partial docs to merge:

{partial_docs}
"""


def _generate_module_doc(
    mod_name: str,
    public_items: list[FunctionInfo | ClassInfo],
    callee_context: str,
    model: str,
) -> str:
    """Generate a module doc, batching large modules into multiple LLM calls."""
    sources = _format_sources(public_items)

    if len(sources) <= MAX_SOURCE_PER_BATCH:
        # Small enough for one call
        prompt = MODULE_PROMPT.format(
            module_name=mod_name,
            callee_context=callee_context,
            sources=sources,
        )
        return generate(prompt, model=model)

    # Large module: split items into batches, generate partial docs, merge
    logger.info("Module %s is large (%d chars), splitting into batches", mod_name, len(sources))
    batches = _batch_items(public_items)
    partial_docs: list[str] = []

    for i, batch in enumerate(batches):
        batch_sources = _format_sources(batch)
        prompt = MODULE_PROMPT.format(
            module_name=mod_name,
            callee_context=callee_context if i == 0 else "",
            sources=batch_sources,
        )
        logger.info("  Batch %d/%d (%d items, %d chars)", i + 1, len(batches), len(batch), len(batch_sources))
        partial = generate(prompt, model=model)
        partial_docs.append(partial)

    if len(partial_docs) == 1:
        return partial_docs[0]

    # Merge partial docs
    logger.info("  Merging %d partial docs", len(partial_docs))
    combined = "\n\n---\n\n".join(f"## Partial {i + 1}\n\n{doc}" for i, doc in enumerate(partial_docs))
    merge_prompt = MERGE_PROMPT.format(module_name=mod_name, partial_docs=combined)
    return generate(merge_prompt, model=model)


def _batch_items(items: list[FunctionInfo | ClassInfo]) -> list[list[FunctionInfo | ClassInfo]]:
    """Split items into batches that fit within MAX_SOURCE_PER_BATCH."""
    batches: list[list[FunctionInfo | ClassInfo]] = []
    current: list[FunctionInfo | ClassInfo] = []
    current_size = 0

    for item in items:
        item_size = len(item.source)
        if current and current_size + item_size > MAX_SOURCE_PER_BATCH:
            batches.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += item_size

    if current:
        batches.append(current)
    return batches


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

        callee_context = _get_callee_context(mod, existing_docs)

        try:
            response = _generate_module_doc(mod_name, public_items, callee_context, model)
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
