# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package doc generator: produces per-package markdown reference cards
directly from source code via the claude CLI.

Two-phase architecture for large packages:
1. Per-file summaries: each file's public items are summarized by an LLM call
2. Package aggregation: file summaries are combined into the final doc

Small packages (under MAX_SOURCE_DIRECT chars) skip phase 1 and go direct.

Also supports legacy module-level docs via generate_module_docs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from agent_docs.cache import DocCache, _combined_source_hash, hash_text
from agent_docs.claude_cli import generate
from agent_docs.graph import ClassInfo, FunctionInfo, ModuleInfo, RepoGraph
from agent_docs.packages import PackageInfo, _package_source_hash
from agent_docs.prompts import FILE_SUMMARY_PROMPT, PACKAGE_PROMPT

logger = logging.getLogger(__name__)

OUTPUT_DIR = "docs/agent/packages"
MAX_SOURCE_DIRECT = 30_000  # chars of formatted source; below this, skip file summaries


def _format_sources(items: list[FunctionInfo | ClassInfo]) -> str:
    """Format function/class sources for the prompt."""
    parts: list[str] = []
    for item in items:
        header = f"### {item.qualified_name} ({item.file_path}:{item.line_number})"
        parts.append(f"{header}\n```{item.language}\n{item.source}\n```\n")
    return "\n".join(parts)


def _group_items_by_file(
    items: list[FunctionInfo | ClassInfo],
) -> dict[str, list[FunctionInfo | ClassInfo]]:
    """Group public items by their source file path."""
    by_file: dict[str, list[FunctionInfo | ClassInfo]] = defaultdict(list)
    for item in items:
        by_file[item.file_path].append(item)
    return dict(by_file)


def _generate_file_summary(
    package_name: str,
    file_path: str,
    items: list[FunctionInfo | ClassInfo],
    model: str,
) -> str:
    """Generate a concise structured summary for one file's public items."""
    sources = _format_sources(items)
    prompt = FILE_SUMMARY_PROMPT.format(
        file_path=file_path,
        package_name=package_name,
        sources=sources,
    )
    return generate(prompt, model=model, max_budget_usd=0.25)


def _generate_file_summaries(
    name: str,
    public_items: list[FunctionInfo | ClassInfo],
    model: str,
) -> str:
    """Phase 1: generate per-file summaries and concatenate them."""
    by_file = _group_items_by_file(public_items)
    summaries: list[str] = []

    for file_path, items in sorted(by_file.items()):
        logger.info("  Summarizing %s (%d items)", file_path, len(items))
        summary = _generate_file_summary(name, file_path, items, model)
        summaries.append(summary)

    return "\n\n---\n\n".join(summaries)


def _generate_direct(
    name: str,
    public_items: list[FunctionInfo | ClassInfo],
    callee_context: str,
    model: str,
) -> str:
    """Small-package path: send raw source directly to the package prompt."""
    sources = _format_sources(public_items)
    prompt = PACKAGE_PROMPT.format(
        package_name=name,
        input_description=f"Given the source code below for package `{name}`,",
        callee_context=callee_context,
        input_section_header=f"Source code for package `{name}`",
        sources=sources,
    )
    return generate(prompt, model=model)


def _aggregate_package_doc(
    name: str,
    file_summaries: str,
    callee_context: str,
    model: str,
) -> str:
    """Phase 2: aggregate file summaries into the final package doc."""
    prompt = PACKAGE_PROMPT.format(
        package_name=name,
        input_description=f"Given the file-level summaries below for package `{name}`,",
        callee_context=callee_context,
        input_section_header=f"File summaries for package `{name}`",
        sources=file_summaries,
    )
    return generate(prompt, model=model)


def _generate_doc(
    name: str,
    pkg: PackageInfo,
    callee_context: str,
    model: str,
) -> str:
    """Generate a package doc via file-level summaries or direct generation."""
    public_items = [f for f in pkg.functions if f.is_public] + [c for c in pkg.classes if c.is_public]
    sources = _format_sources(public_items)

    if len(sources) <= MAX_SOURCE_DIRECT:
        return _generate_direct(name, public_items, callee_context, model)

    logger.info("Package %s is large (%d chars), using file-summary pipeline", name, len(sources))
    file_summaries = _generate_file_summaries(name, public_items, model)
    return _aggregate_package_doc(name, file_summaries, callee_context, model)


def _get_callee_context_for_package(pkg: PackageInfo, existing_docs: dict[str, str]) -> str:
    """Build callee context from already-generated docs of imported packages."""
    context_parts: list[str] = []
    for dep_name in sorted(pkg.imports_from):
        if dep_name in existing_docs:
            doc = existing_docs[dep_name]
            if len(doc) > 2000:
                doc = doc[:2000] + "\n... (truncated)"
            context_parts.append(f"### {dep_name}\n{doc}")

    if not context_parts:
        return ""

    return "## Context from imported packages:\n\n" + "\n\n".join(context_parts)


def _topo_sort_packages(packages: dict[str, PackageInfo]) -> list[str]:
    """Topological sort of packages by import dependencies (leaves first)."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        pkg = packages.get(name)
        if pkg is None:
            return
        for dep in pkg.imports_from:
            visit(dep)
        order.append(name)

    for name in packages:
        visit(name)

    return order


def generate_package_docs(
    packages: dict[str, PackageInfo],
    cache: DocCache,
    stale_packages: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> set[str]:
    """Generate package reference cards for all stale packages.

    Processes packages in topological order so that callee docs are available
    as context when generating caller docs.

    Returns the set of package names that were updated.
    """
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pkg_order = _topo_sort_packages(packages)

    # Pre-load existing docs for non-stale packages (callee context)
    existing_docs: dict[str, str] = {}
    for pkg_name in pkg_order:
        if pkg_name not in stale_packages:
            md_path = output_dir / f"{pkg_name}.md"
            if md_path.exists():
                existing_docs[pkg_name] = md_path.read_text()

    updated: set[str] = set()

    for pkg_name in pkg_order:
        if pkg_name not in stale_packages:
            continue

        pkg = packages[pkg_name]
        public_items = [f for f in pkg.functions if f.is_public] + [c for c in pkg.classes if c.is_public]

        if not public_items:
            logger.info("Skipping %s (no public items)", pkg_name)
            continue

        if dry_run:
            sources = _format_sources(public_items)
            path = "file-summary" if len(sources) > MAX_SOURCE_DIRECT else "direct"
            logger.info("[dry-run] Would generate doc for %s (%d items, %s path)", pkg_name, len(public_items), path)
            continue

        logger.info("Generating doc for %s (%d public items)", pkg_name, len(public_items))

        callee_context = _get_callee_context_for_package(pkg, existing_docs)

        try:
            response = _generate_doc(pkg_name, pkg, callee_context, model)
            md_path = output_dir / f"{pkg_name}.md"
            md_path.write_text(response.rstrip() + "\n")

            existing_docs[pkg_name] = response
            updated.add(pkg_name)

            source_hash = _package_source_hash(pkg)
            doc_hash = hash_text(response)
            cache.update(pkg_name, source_hash, doc_hash, tier=2)

        except Exception:
            logger.error("Failed to generate doc for %s", pkg_name, exc_info=True)

    return updated


# ---------------------------------------------------------------------------
# Legacy module-level API (kept for backward compatibility with experiments)
# ---------------------------------------------------------------------------


def _get_callee_context(mod: ModuleInfo, existing_docs: dict[str, str]) -> str:
    """Build callee context from already-generated docs of imported modules."""
    context_parts: list[str] = []
    for dep_mod_name in sorted(mod.imports_from):
        if dep_mod_name in existing_docs:
            doc = existing_docs[dep_mod_name]
            if len(doc) > 2000:
                doc = doc[:2000] + "\n... (truncated)"
            context_parts.append(f"### {dep_mod_name}\n{doc}")

    if not context_parts:
        return ""

    return "## Context from imported modules:\n\n" + "\n\n".join(context_parts)


def _generate_legacy_doc(
    name: str,
    public_items: list[FunctionInfo | ClassInfo],
    callee_context: str,
    model: str,
) -> str:
    """Generate a doc for the legacy module-level path (direct only)."""
    sources = _format_sources(public_items)
    prompt = PACKAGE_PROMPT.format(
        package_name=name,
        input_description=f"Given the source code below for package `{name}`,",
        callee_context=callee_context,
        input_section_header=f"Source code for package `{name}`",
        sources=sources,
    )
    return generate(prompt, model=model)


def generate_module_docs(
    graph: RepoGraph,
    cache: DocCache,
    stale_modules: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    dry_run: bool = False,
) -> set[str]:
    """Legacy: generate module docs using depth-2 grouping from RepoGraph."""
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

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

    existing_docs: dict[str, str] = {}
    for mod_name in order:
        if mod_name not in stale_modules:
            md_path = output_dir / f"{mod_name}.md"
            if md_path.exists():
                existing_docs[mod_name] = md_path.read_text()

    updated: set[str] = set()

    for mod_name in order:
        mod = graph.modules[mod_name]
        if mod_name not in stale_modules:
            continue

        public_items: list[FunctionInfo | ClassInfo] = [f for f in mod.functions if f.is_public] + [
            c for c in mod.classes if c.is_public
        ]

        if not public_items:
            continue

        if dry_run:
            logger.info("[dry-run] Would generate doc for %s (%d items)", mod_name, len(public_items))
            continue

        logger.info("Generating doc for %s (%d items)", mod_name, len(public_items))
        callee_context = _get_callee_context(mod, existing_docs)

        try:
            response = _generate_legacy_doc(mod_name, public_items, callee_context, model)
            md_path = output_dir / f"{mod_name}.md"
            md_path.write_text(response.rstrip() + "\n")
            existing_docs[mod_name] = response
            updated.add(mod_name)
            combined_hash = _combined_source_hash(public_items)
            cache.update(mod_name, combined_hash, hash_text(response), tier=2)
        except Exception:
            logger.error("Failed to generate doc for %s", mod_name, exc_info=True)

    return updated
