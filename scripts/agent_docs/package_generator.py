# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package doc generator: produces per-package markdown reference cards
directly from source code via the claude CLI.

Two-phase architecture for large packages:
1. Per-file summaries: each file's public items are summarized by an LLM call
2. Package aggregation: file summaries are combined into the final doc

Small packages (under MAX_SOURCE_DIRECT chars) skip phase 1 and go direct.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

from agent_docs.claude_cli import generate
from agent_docs.packages import PackageInfo
from agent_docs.parsing import ClassInfo, FunctionInfo
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
    summary_model: str,
) -> str:
    """Generate a concise structured summary for one file's public items."""
    sources = _format_sources(items)
    prompt = FILE_SUMMARY_PROMPT.format(
        file_path=file_path,
        package_name=package_name,
        sources=sources,
    )
    return generate(prompt, model=summary_model, max_budget_usd=0.25)


def _generate_file_summaries(
    name: str,
    public_items: list[FunctionInfo | ClassInfo],
    summary_model: str,
) -> str:
    """Phase 1: generate per-file summaries and concatenate them."""
    by_file = _group_items_by_file(public_items)
    sorted_files = sorted(by_file.items())
    total_files = len(sorted_files)
    summaries: list[str] = []
    phase_t0 = time.monotonic()

    for i, (file_path, items) in enumerate(sorted_files, 1):
        short_path = Path(file_path).name
        logger.info("  [%d/%d] Summarizing %s (%d items)...", i, total_files, short_path, len(items))
        t0 = time.monotonic()
        summary = _generate_file_summary(name, file_path, items, summary_model)
        elapsed = time.monotonic() - t0
        logger.info("  [%d/%d] %s done (%.1fs, %d chars)", i, total_files, short_path, elapsed, len(summary))
        summaries.append(summary)

    phase_elapsed = time.monotonic() - phase_t0
    logger.info("  Phase 1 complete: %d files in %.1fs (model=%s)", total_files, phase_elapsed, summary_model)

    return "\n\n---\n\n".join(summaries)


def _generate_direct(
    name: str,
    public_items: list[FunctionInfo | ClassInfo],
    callee_context: str,
    model: str,
) -> str:
    """Small-package path: send raw source directly to the package prompt."""
    sources = _format_sources(public_items)
    logger.info("  Direct generation (%d chars source)...", len(sources))
    prompt = PACKAGE_PROMPT.format(
        package_name=name,
        input_description=f"Given the source code below for package `{name}`,",
        callee_context=callee_context,
        input_section_header=f"Source code for package `{name}`",
        sources=sources,
    )
    t0 = time.monotonic()
    result = generate(prompt, model=model)
    logger.info("  Direct generation done (%.1fs, %d chars output)", time.monotonic() - t0, len(result))
    return result


def _aggregate_package_doc(
    name: str,
    file_summaries: str,
    callee_context: str,
    model: str,
) -> str:
    """Phase 2: aggregate file summaries into the final package doc."""
    logger.info("  Phase 2: aggregating %d chars of file summaries...", len(file_summaries))
    prompt = PACKAGE_PROMPT.format(
        package_name=name,
        input_description=f"Given the file-level summaries below for package `{name}`,",
        callee_context=callee_context,
        input_section_header=f"File summaries for package `{name}`",
        sources=file_summaries,
    )
    t0 = time.monotonic()
    result = generate(prompt, model=model)
    logger.info("  Phase 2 done (%.1fs, %d chars output)", time.monotonic() - t0, len(result))
    return result


def _generate_doc(
    name: str,
    pkg: PackageInfo,
    callee_context: str,
    model: str,
    summary_model: str,
) -> str:
    """Generate a package doc via file-level summaries or direct generation."""
    public_items = [f for f in pkg.functions if f.is_public] + [c for c in pkg.classes if c.is_public]
    sources = _format_sources(public_items)

    if len(sources) <= MAX_SOURCE_DIRECT:
        return _generate_direct(name, public_items, callee_context, model)

    logger.info("Package %s is large (%d chars), using file-summary pipeline", name, len(sources))
    file_summaries = _generate_file_summaries(name, public_items, summary_model)
    return _aggregate_package_doc(name, file_summaries, callee_context, model)


def _get_callee_context(pkg: PackageInfo, existing_docs: dict[str, str]) -> str:
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
    target_packages: set[str],
    repo_root: Path,
    *,
    model: str = "sonnet",
    summary_model: str = "haiku",
    dry_run: bool = False,
) -> set[str]:
    """Generate package reference cards for target packages.

    Processes packages in topological order so that callee docs are available
    as context when generating caller docs.

    Returns the set of package names that were generated.
    """
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pkg_order = _topo_sort_packages(packages)

    # Pre-load existing docs for non-target packages (callee context)
    existing_docs: dict[str, str] = {}
    for pkg_name in pkg_order:
        if pkg_name not in target_packages:
            md_path = output_dir / f"{pkg_name}.md"
            if md_path.exists():
                existing_docs[pkg_name] = md_path.read_text()

    targets_with_items = []
    for pkg_name in pkg_order:
        if pkg_name not in target_packages:
            continue
        pkg = packages[pkg_name]
        public_items = [f for f in pkg.functions if f.is_public] + [c for c in pkg.classes if c.is_public]
        if not public_items:
            logger.info("Skipping %s (no public items)", pkg_name)
            continue
        targets_with_items.append((pkg_name, pkg, public_items))

    total = len(targets_with_items)
    if total == 0:
        logger.info("No target packages with public items to generate")
        return set()

    logger.info("Generating docs for %d package(s)", total)

    updated: set[str] = set()
    pipeline_t0 = time.monotonic()

    for idx, (pkg_name, pkg, public_items) in enumerate(targets_with_items, 1):
        if dry_run:
            sources = _format_sources(public_items)
            path = "file-summary" if len(sources) > MAX_SOURCE_DIRECT else "direct"
            logger.info("[dry-run] [%d/%d] Would generate %s (%d items, %s)", idx, total, pkg_name, len(public_items), path)
            continue

        logger.info(
            "[%d/%d] Generating %s (%d public items, %d files)",
            idx,
            total,
            pkg_name,
            len(public_items),
            len(pkg.file_paths),
        )
        pkg_t0 = time.monotonic()

        callee_context = _get_callee_context(pkg, existing_docs)
        response = _generate_doc(pkg_name, pkg, callee_context, model, summary_model)

        md_path = output_dir / f"{pkg_name}.md"
        md_path.write_text(response.rstrip() + "\n")

        existing_docs[pkg_name] = response
        updated.add(pkg_name)

        pkg_elapsed = time.monotonic() - pkg_t0
        logger.info("[%d/%d] %s done (%.1fs, %d chars)", idx, total, pkg_name, pkg_elapsed, len(response))

    total_elapsed = time.monotonic() - pipeline_t0
    logger.info("Pipeline complete: %d/%d packages generated in %.1fs", len(updated), total, total_elapsed)

    return updated
