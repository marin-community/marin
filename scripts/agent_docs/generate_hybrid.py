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
#     "tiktoken>=0.7",
# ]
# ///
"""Hybrid agent-doc generation: agentic NARRATIVE half + mechanical REFERENCE half.

Experiment v3. Each doc is split into two complementary halves within a 1500-token
budget:

  - the NARRATIVE half is written by a tool-using agent (Read/Grep/Glob) that
    explores real source and conveys what a signature list cannot — concepts,
    data flow, the reasoning behind the design;
  - the REFERENCE half is derived MECHANICALLY from the tree-sitter parse (no LLM):
    the important files, the entry-point signatures with defaults, or the internal
    edit seams, depending on the axis.

The two halves play to each track's strength: the prior 2x2 found mechanical docs
win on exact symbols/defaults (what a docs-only coder needs) while agentic docs
read better as narrative. The hybrid keeps both.

Cost control: the narrative half (the only paid part) is cached per (project, axis)
under ``--narrative-cache``, so iterating the free mechanical half costs nothing.

Same output layout as the other tracks (``<project>/{overview,ops,architecture}.md``
+ ``MAP.md``) so the probe eval scores all three identically.
"""

import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import click

# Add scripts/ to path so agent_docs is importable as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.claude_cli import generate, generate_with_tools
from agent_docs.generate_agentic import ALLOWED_TOOLS, DEFAULT_TOTAL_BUDGET_USD, source_roots
from agent_docs.generate_taxonomy import _enforce_budget, discover_subprojects
from agent_docs.packages import PackageInfo, discover_packages
from agent_docs.parsing import ClassInfo, FunctionInfo
from agent_docs.prompts import (
    HYBRID_NARRATIVE_ARCHITECTURE_PROMPT,
    HYBRID_NARRATIVE_OPS_PROMPT,
    HYBRID_NARRATIVE_OVERVIEW_PROMPT,
    TAXONOMY_MAP_PROMPT,
)
from agent_docs.tokens import count_tokens

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent

# Budget split within the 1500-token/doc cap. The narrative half is LLM-shortened
# to its budget; the reference half is then rendered to whatever is left, so the
# whole doc never exceeds HYBRID_DOC_TOKEN_BUDGET without ever LLM-touching (and
# risking corrupting) the mechanically-exact reference half.
HYBRID_DOC_TOKEN_BUDGET = 1500
NARRATIVE_TOKEN_BUDGET = 750
# Per-doc spend cap for the agentic narrative half (API-equivalent).
PER_DOC_BUDGET_USD = 1.5

HYBRID_REFERENCE_HEADER = "## Reference (mechanically derived from source)"

# The overview axis is a 30-second router, not a file census: cap its file list
# regardless of remaining budget. (ops/architecture want as many entries as fit.)
OVERVIEW_FILES_LIMIT = 12

# Per-class method caps in the architecture edit-site half: a god-class
# (Controller, Trainer) has dozens of methods; without a cap one class block
# would consume the whole reference budget, starving every other edit site.
MAX_METHODS_PER_CLASS = 6
MAX_INTERNAL_PER_CLASS = 4

_WS = re.compile(r"\s+")


def _norm_sig(sig: str) -> str:
    """Collapse a possibly multi-line tree-sitter signature onto one line."""
    return _WS.sub(" ", sig).strip()


def _strip_preamble(text: str) -> str:
    """Drop any agent chatter before the doc's first markdown heading.

    A tool-using agent sometimes prefixes its answer with a line like "Now I have
    enough information to write the doc." Every narrative prompt mandates a doc
    that starts with a ``##`` section, so we trim to the first heading; if there
    is none, return the text unchanged.
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            return "\n".join(lines[i:]).strip()
    return text.strip()


NARRATIVE_PROMPTS = {
    "overview": HYBRID_NARRATIVE_OVERVIEW_PROMPT,
    "ops": HYBRID_NARRATIVE_OPS_PROMPT,
    "architecture": HYBRID_NARRATIVE_ARCHITECTURE_PROMPT,
}
AXES = ("overview", "ops", "architecture")


# ---------------------------------------------------------------------------
# Mechanical reference half (deterministic — no LLM, no hallucination)
# ---------------------------------------------------------------------------


def _rel(path: str) -> str:
    """Make an absolute source path repo-relative for compact, clickable anchors."""
    root = str(REPO_ROOT) + "/"
    return path[len(root) :] if path.startswith(root) else path


def _public_funcs(pkg: PackageInfo) -> list[FunctionInfo]:
    return [f for f in pkg.functions if f.is_public]


def _public_classes(pkg: PackageInfo) -> list[ClassInfo]:
    return [c for c in pkg.classes if c.is_public]


def _round_robin(per_group: list[list[str]]) -> list[str]:
    """Interleave per-group line lists so every group is represented before any
    group's long tail — this is how the reference half gets BREADTH across the
    sub-project's many packages instead of exhausting one package first."""
    out: list[str] = []
    for i in range(max((len(g) for g in per_group), default=0)):
        for group in per_group:
            if i < len(group):
                out.append(group[i])
    return out


def _trim_to_budget(header: str, lines: list[str], budget: int) -> str:
    """Keep ``lines`` (in priority order) under ``header`` within ``budget`` tokens.

    Skips an over-large item and keeps scanning rather than stopping at the first
    overflow, so one god-class block does not strand the remaining budget — later,
    smaller edit sites still get listed.
    """
    kept: list[str] = []
    omitted = 0
    for ln in lines:
        if count_tokens("\n".join([header, *kept, ln])) > budget:
            omitted += 1
            continue
        kept.append(ln)
    if omitted > 0:
        kept.append(f"- … ({omitted} more symbols omitted for budget)")
    return "\n".join([header, *kept]) if kept else ""


def render_important_files(packages: dict[str, PackageInfo], names: list[str], budget: int) -> str:
    """REFERENCE half for the overview axis: files ranked by public-symbol count."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # path -> [funcs, classes]
    for name in names:
        pkg = packages[name]
        for f in _public_funcs(pkg):
            counts[f.file_path][0] += 1
        for c in _public_classes(pkg):
            counts[c.file_path][1] += 1
    ranked = sorted(counts.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:OVERVIEW_FILES_LIMIT]
    lines = [f"- `{_rel(p)}` — {nf} functions, {nc} classes" for p, (nf, nc) in ranked]
    return _trim_to_budget("### Files that matter (by public-symbol count)", lines, budget)


def render_entry_index(packages: dict[str, PackageInfo], names: list[str], budget: int) -> str:
    """REFERENCE half for the ops axis: entry-point signatures, breadth-first.

    Class constructors and top-level functions across the WHOLE sub-project, with
    full signatures + default values, interleaved across packages for coverage.
    """
    per_pkg: list[list[str]] = []
    for name in sorted(names):
        pkg = packages[name]
        entries: list[str] = []
        for c in _public_classes(pkg):
            ctor = next((m for m in c.methods if m.name == "__init__"), None)
            sig = _norm_sig(ctor.signature) if ctor else "()"
            entries.append(f"- `{c.qualified_name}{sig}` ({_rel(c.file_path)}:{c.line_number})")
        for f in _public_funcs(pkg):
            entries.append(f"- `{f.qualified_name}{_norm_sig(f.signature)}` ({_rel(f.file_path)}:{f.line_number})")
        if entries:
            per_pkg.append(entries)
    return _trim_to_budget("### Entry-point index (signatures, breadth-first)", _round_robin(per_pkg), budget)


def _edit_site_class(cls: ClassInfo) -> str:
    """One compact edit-site block: ctor + a few public + a few internal methods."""
    ctor = next((m for m in cls.methods if m.name == "__init__"), None)
    sig = _norm_sig(ctor.signature) if ctor else "()"
    head = f"- class `{cls.qualified_name}{sig}` ({_rel(cls.file_path)}:{cls.line_number})"
    pub = [m for m in cls.methods if m.is_public][:MAX_METHODS_PER_CLASS]
    internal = [m for m in cls.methods if not m.is_public and m.name != "__init__"][:MAX_INTERNAL_PER_CLASS]
    lines = [head]
    lines += [f"    - `{m.name}{_norm_sig(m.signature)}`" for m in pub]
    lines += [f"    - internal `{m.name}{_norm_sig(m.signature)}`" for m in internal]
    return "\n".join(lines)


def _internal_method_count(cls: ClassInfo) -> int:
    return sum(1 for m in cls.methods if not m.is_public and m.name != "__init__")


def render_edit_sites(packages: dict[str, PackageInfo], names: list[str], budget: int) -> str:
    """REFERENCE half for the architecture axis: edit seams incl. internals.

    Classes (ctor + a few public + a few internal methods) and internal
    module-level functions — the places behavior is actually implemented and
    changed. Classes that HAVE internal seams come first (those are the real edit
    sites), then by total method count; each class block is method-capped so one
    god-class cannot starve the rest of the budget.
    """
    classes: list[ClassInfo] = []
    internal_funcs: list[FunctionInfo] = []
    for name in names:
        pkg = packages[name]
        classes += _public_classes(pkg)
        internal_funcs += [f for f in pkg.functions if not f.is_public and not f.name.startswith("__")]
    classes.sort(key=lambda c: (-_internal_method_count(c), -len(c.methods)))
    lines = [_edit_site_class(c) for c in classes]
    lines += [
        f"- internal `{f.qualified_name}{_norm_sig(f.signature)}` ({_rel(f.file_path)}:{f.line_number})"
        for f in internal_funcs
    ]
    return _trim_to_budget("### Edit sites (classes, methods, internal seams)", lines, budget)


def render_mechanical(packages: dict[str, PackageInfo], names: list[str], axis: str, budget: int) -> str:
    """Render the axis-appropriate mechanical reference half within ``budget``."""
    if axis == "ops":
        return render_entry_index(packages, names, budget)
    if axis == "architecture":
        return render_edit_sites(packages, names, budget)
    return render_important_files(packages, names, budget)


# ---------------------------------------------------------------------------
# Narrative half (agentic — cached)
# ---------------------------------------------------------------------------


def _narrative(
    project: str, axis: str, roots_block: str, *, model: str, max_budget_usd: float, cache: Path | None
) -> tuple[str, float]:
    """Return the (LLM-shortened) narrative half and its cost.

    Reuses a cached narrative for (project, axis) when ``cache`` is set and the
    file exists — cost 0.0 — so iterating the free mechanical half is free.
    """
    cache_file = cache / f"{project}-{axis}.md" if cache else None
    if cache_file and cache_file.exists():
        logger.info("  narrative %s/%s: cache hit", project, axis)
        return _strip_preamble(cache_file.read_text()), 0.0

    prompt = NARRATIVE_PROMPTS[axis].format(project_name=project, source_roots=roots_block)
    text, cost = generate_with_tools(
        prompt, model=model, allowed_tools=ALLOWED_TOOLS, cwd=str(REPO_ROOT), max_budget_usd=max_budget_usd
    )
    text = _strip_preamble(_enforce_budget(text, model, budget=NARRATIVE_TOKEN_BUDGET))
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text + "\n")
    return text, cost


def generate_taxonomy_hybrid(
    packages: dict[str, PackageInfo],
    subprojects: list[str],
    output_dir: Path,
    *,
    model: str,
    budget_usd: float,
    narrative_cache: Path | None = None,
    dry_run: bool = False,
) -> float:
    """Generate the hybrid taxonomy. Returns total API-equivalent spend.

    Each doc = narrative half (agentic, budget-shortened) + mechanical reference
    half (deterministic), assembled to stay within HYBRID_DOC_TOKEN_BUDGET.
    """
    by_project = discover_subprojects(packages)
    overviews: dict[str, str] = {}
    spent = 0.0

    for project in subprojects:
        names = by_project.get(project)
        if not names:
            logger.warning("Sub-project %r has no packages with public items; skipping", project)
            continue
        roots = source_roots(project)
        roots_block = "\n".join(f"- `{r}`" for r in roots)
        logger.info("%s: %d packages, roots %s", project, len(names), ", ".join(roots))
        if dry_run:
            continue

        project_dir = output_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        for axis in AXES:
            if spent >= budget_usd:
                logger.warning("Budget $%.2f reached; stopping before %s/%s", budget_usd, project, axis)
                return spent
            t0 = time.monotonic()
            narrative, cost = _narrative(
                project,
                axis,
                roots_block,
                model=model,
                max_budget_usd=min(PER_DOC_BUDGET_USD, budget_usd - spent),
                cache=narrative_cache,
            )
            spent += cost
            # Render the mechanical half into whatever budget the narrative left,
            # so the combined doc stays <= 1500 without LLM-touching the reference.
            used = count_tokens(narrative) + count_tokens(HYBRID_REFERENCE_HEADER)
            mech_budget = max(HYBRID_DOC_TOKEN_BUDGET - used, 200)
            mechanical = render_mechanical(packages, names, axis, mech_budget)
            doc = f"{narrative}\n\n{HYBRID_REFERENCE_HEADER}\n\n{mechanical}".rstrip() + "\n"
            (project_dir / f"{axis}.md").write_text(doc)
            logger.info(
                "  %s/%s.md: %d tokens (narr %d + ref %d), $%.2f (%.0fs) [spent $%.2f/$%.2f]",
                project,
                axis,
                count_tokens(doc),
                count_tokens(narrative),
                count_tokens(mechanical),
                cost,
                time.monotonic() - t0,
                spent,
                budget_usd,
            )
            if axis == "overview":
                overviews[project] = doc

    if overviews and not dry_run:
        blocks = [f"## {p}\n\n{text}" for p, text in sorted(overviews.items())]
        map_doc = _enforce_budget(
            generate(TAXONOMY_MAP_PROMPT.format(overviews="\n\n---\n\n".join(blocks)), model=model, max_budget_usd=0.5),
            model,
        )
        (output_dir / "MAP.md").write_text(map_doc.rstrip() + "\n")
        logger.info("Wrote MAP.md (%d tokens)", count_tokens(map_doc))

    logger.info("Hybrid generation spent $%.2f (API-equivalent) of $%.2f", spent, budget_usd)
    return spent


@click.command()
@click.option("--output-dir", default="docs/agent", type=click.Path(), help="Where the taxonomy is written.")
@click.option("--subproject", "subprojects", multiple=True, required=True, help="Sub-projects to generate.")
@click.option("--model", default="sonnet", help="Narrative-half generation model.")
@click.option("--budget-usd", default=DEFAULT_TOTAL_BUDGET_USD, type=float, help="Total spend cap (API-equivalent).")
@click.option("--narrative-cache", type=click.Path(), default=None, help="Reuse/store narrative halves here.")
@click.option("--dry-run", is_flag=True, help="Report packages/roots without LLM calls.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    output_dir: str,
    subprojects: tuple[str, ...],
    model: str,
    budget_usd: float,
    narrative_cache: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Generate the hybrid (narrative + mechanical) agent-doc taxonomy."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    packages = discover_packages(REPO_ROOT)
    spent = generate_taxonomy_hybrid(
        packages,
        list(subprojects),
        Path(output_dir),
        model=model,
        budget_usd=budget_usd,
        narrative_cache=Path(narrative_cache) if narrative_cache else None,
        dry_run=dry_run,
    )
    if not dry_run:
        logger.info("Done. Budget $%.2f/$%.2f. Per-doc budget %d tokens.", spent, budget_usd, HYBRID_DOC_TOKEN_BUDGET)


if __name__ == "__main__":
    main()
