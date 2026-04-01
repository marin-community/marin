# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 generator: function-level structured YAML docs.

Generates docs bottom-up by topological order — leaf modules first so that
callers get richer context from already-documented callees.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from agent_docs.cache import DocCache, _combined_source_hash, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import ClassInfo, FunctionInfo, ModuleInfo, RepoGraph
from agent_docs.prompts import TIER3_PROMPT

logger = logging.getLogger(__name__)

OUTPUT_DIR = "docs/agent/api"
MAX_SOURCE_PER_BATCH = 30_000  # chars of source per LLM call


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
            # Resolve dep to actual module names in the graph
            for mod_name in graph.modules:
                if mod_name == dep or mod_name.startswith(f"{dep}."):
                    visit(mod_name)
        order.append(name)

    for name in graph.modules:
        visit(name)

    return order


def _format_sources(items: list[FunctionInfo | ClassInfo]) -> str:
    """Format function/class sources for the prompt."""
    parts: list[str] = []
    for item in items:
        header = f"### {item.qualified_name} ({item.file_path}:{item.line_number})"
        parts.append(f"{header}\n```{item.language}\n{item.source}\n```\n")
    return "\n".join(parts)


def _batch_items(items: list[FunctionInfo | ClassInfo]) -> list[list[FunctionInfo | ClassInfo]]:
    """Split items into batches that fit within the source size limit."""
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


def _load_existing_yaml(path: Path) -> dict:
    """Load existing YAML docs if present."""
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def _get_callee_context(mod: ModuleInfo, all_docs: dict[str, dict]) -> str:
    """Build callee context from already-generated docs of imported modules."""
    callee_entries: dict[str, dict] = {}
    for dep_mod_name in mod.imports_from:
        for key, doc in all_docs.items():
            if key.startswith(dep_mod_name):
                callee_entries[key] = doc

    if not callee_entries:
        return ""

    # Trim to keep prompt reasonable — only include summaries, not full entries
    lines: list[str] = []
    for key, doc in sorted(callee_entries.items()):
        summary = doc.get("summary", "")
        sig = doc.get("signature", "")
        if summary:
            lines.append(f"- `{key}`: {sig} — {summary}")

    if not lines:
        return ""

    return "## Already-documented callees (for context):\n\n" + "\n".join(lines[:100])


def generate_tier3(
    graph: RepoGraph,
    cache: DocCache,
    stale_modules: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> set[str]:
    """Generate Tier 3 YAML docs for all stale modules.

    Returns the set of module names that were updated.
    """
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    module_order = _topo_sort_modules(graph)
    all_docs: dict[str, dict] = {}

    # Pre-load existing docs for non-stale modules (callee context)
    for mod_name in module_order:
        if mod_name not in stale_modules:
            yaml_path = output_dir / f"{mod_name}.yaml"
            existing = _load_existing_yaml(yaml_path)
            all_docs.update(existing)

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
            logger.info("[dry-run] Would generate Tier 3 for %s (%d items)", mod_name, len(public_items))
            continue

        logger.info("Generating Tier 3 for %s (%d items)", mod_name, len(public_items))

        batches = _batch_items(public_items)
        module_docs: dict[str, dict] = {}

        for batch_idx, batch in enumerate(batches):
            callee_context = _get_callee_context(mod, all_docs)
            sources = _format_sources(batch)

            prompt = TIER3_PROMPT.format(
                callee_context=callee_context,
                sources=sources,
            )

            try:
                response = generate(prompt, model=model)
                parsed = yaml.safe_load(response)
                if isinstance(parsed, dict):
                    module_docs.update(parsed)
                else:
                    logger.warning("Tier 3 response for %s batch %d was not a dict", mod_name, batch_idx)
            except Exception:
                logger.error("Failed to generate Tier 3 for %s batch %d", mod_name, batch_idx, exc_info=True)

        if module_docs:
            yaml_path = output_dir / f"{mod_name}.yaml"
            # Merge with existing (preserve docs for items we didn't regenerate)
            existing = _load_existing_yaml(yaml_path)
            existing.update(module_docs)
            yaml_path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=True, width=120))

            all_docs.update(module_docs)
            updated.add(mod_name)

            # Update cache
            combined_hash = _combined_source_hash(public_items)
            doc_hash = hash_text(yaml.dump(module_docs, sort_keys=True))
            cache.update(mod_name, combined_hash, doc_hash, tier=3)

    return updated
