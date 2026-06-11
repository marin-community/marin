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
"""Generate the budgeted two-axis agent-doc taxonomy.

For each sub-project the generator emits, under ``docs/agent/``:
  - ``<project>/overview.md``     — 30-second orientation + intent router
  - ``<project>/ops.md``          — how to USE the project
  - ``<project>/architecture.md`` — how to UNDERSTAND and CHANGE it
  - ``MAP.md``                    — monorepo index across sub-projects

Every file is held to a ~1000-token budget: a draft over budget gets one
shorten pass. Docs are generated from a signatures-only source digest (built
from the tree-sitter parse, no LLM), which keeps the per-round cost bounded and
lets each iteration vary only the prompts.
"""

import logging
import re
import sys
import time
from pathlib import Path

import click

# Add scripts/ to path so agent_docs is importable as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.claude_cli import generate
from agent_docs.packages import PackageInfo, discover_packages
from agent_docs.parsing import ClassInfo
from agent_docs.prompts import ARCHITECTURE_PROMPT, OPS_PROMPT, OVERVIEW_PROMPT, SHORTEN_PROMPT, TAXONOMY_MAP_PROMPT
from agent_docs.tokens import count_tokens

logger = logging.getLogger(__name__)

DOC_TOKEN_BUDGET = 1000
MAX_DIGEST_CHARS = 600_000

# Package-name head -> sub-project. The finelog Rust crate is keyed separately
# from the finelog Python package to avoid a name collision; both are one project.
SUBPROJECT_ALIASES = {"finelog_rust": "finelog"}

# The axes generated per sub-project, in order.
AXES = (
    ("overview", OVERVIEW_PROMPT),
    ("ops", OPS_PROMPT),
    ("architecture", ARCHITECTURE_PROMPT),
)


def subproject_of(package_name: str) -> str:
    """Map a dotted package name to its sub-project."""
    head = package_name.split(".")[0]
    return SUBPROJECT_ALIASES.get(head, head)


def discover_subprojects(packages: dict[str, PackageInfo]) -> dict[str, list[str]]:
    """Group package names by sub-project, keeping only those with public items."""
    by_project: dict[str, list[str]] = {}
    for name, pkg in sorted(packages.items()):
        has_public = any(f.is_public for f in pkg.functions) or any(c.is_public for c in pkg.classes)
        if not has_public:
            continue
        by_project.setdefault(subproject_of(name), []).append(name)
    return by_project


_FIELD_RE = re.compile(r"^\s+[A-Za-z_]\w*\s*:\s*.+$")
MAX_CLASS_FIELDS = 30
# Per-class cap on private methods surfaced in the architecture digest. The
# output budget forces selection anyway; this just bounds digest bloat.
MAX_PRIVATE_METHODS = 25


def _class_fields(cls: ClassInfo) -> list[str]:
    """Extract dataclass-style field declarations from a class's source.

    Dataclasses (pervasive in this codebase for configs) have no explicit
    ``__init__``, so their fields live as class-body annotated assignments
    (``name: type = default``). Without these the digest shows ``Foo()`` and the
    generator invents field names. Collect the annotated assignments that appear
    before the first method, skipping the docstring.
    """
    lines = cls.source.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if line.strip().startswith("class ")), 0)
    fields: list[str] = []
    in_doc = False
    doc_delim = ""
    for line in lines[start:]:
        s = line.strip()
        if in_doc:
            if doc_delim in s:
                in_doc = False
            continue
        if s.startswith(('"""', "'''")):
            doc_delim = s[:3]
            if not (len(s) > 3 and s.endswith(doc_delim)):
                in_doc = True
            continue
        if s.startswith(("def ", "async def ", "@")):
            break
        if not s or s.startswith("#"):
            continue
        if _FIELD_RE.match(line):
            fields.append(s)
        if len(fields) >= MAX_CLASS_FIELDS:
            break
    return fields


def _format_class(cls: ClassInfo, *, include_internals: bool = False) -> str:
    """One compact digest entry for a class: name, ctor/fields, public methods.

    With ``include_internals`` (the architecture axis), also list private methods
    — the internal seams where behavior is actually implemented and edited —
    capped per class and marked ``internal``. This lets the generator cite the
    real edit site (e.g. ``Controller._reconcile_tick``) in "where to change
    things" instead of stopping at the public entry point.
    """
    ctor = next((m for m in cls.methods if m.name == "__init__"), None)
    head = f"- class `{cls.qualified_name}{ctor.signature if ctor else '()'}`  ({cls.file_path}:{cls.line_number})"
    lines = [head]
    if ctor is None:
        lines += [f"    - field `{f}`" for f in _class_fields(cls)]
    lines += [f"    - `{m.name}{m.signature}`" for m in cls.methods if m.is_public]
    if include_internals:
        private = [m for m in cls.methods if not m.is_public and m.name != "__init__"]
        lines += [f"    - internal `{m.name}{m.signature}`" for m in private[:MAX_PRIVATE_METHODS]]
    return "\n".join(lines)


def build_digest(
    packages: dict[str, PackageInfo], package_names: list[str], *, include_internals: bool = False
) -> tuple[str, bool]:
    """Build a signatures-only digest for a sub-project's packages.

    Returns (digest, truncated). The digest lists, per package, its public
    function signatures and classes (constructor + public methods) with
    ``file:line`` anchors — no source bodies. Capped at MAX_DIGEST_CHARS.

    With ``include_internals`` (architecture axis), also surface private
    module-level functions and private methods on public classes — the internal
    seams an architecture doc must name to answer "where would I change this".
    """
    parts: list[str] = []
    for name in package_names:
        pkg = packages[name]
        funcs = [f for f in pkg.functions if f.is_public]
        classes = [c for c in pkg.classes if c.is_public]
        internal_funcs = (
            [f for f in pkg.functions if not f.is_public and not f.name.startswith("__")] if include_internals else []
        )
        if not funcs and not classes and not internal_funcs:
            continue
        section = [f"### package `{name}` ({pkg.language})"]
        for f in funcs:
            section.append(f"- `{f.qualified_name}{f.signature}`  ({f.file_path}:{f.line_number})")
        for f in internal_funcs:
            section.append(f"- internal `{f.qualified_name}{f.signature}`  ({f.file_path}:{f.line_number})")
        for c in classes:
            section.append(_format_class(c, include_internals=include_internals))
        parts.append("\n".join(section))

    digest = "\n\n".join(parts)
    if len(digest) <= MAX_DIGEST_CHARS:
        return digest, False
    return digest[:MAX_DIGEST_CHARS] + "\n\n... (digest truncated at budget)", True


def _enforce_budget(doc: str, model: str, *, budget: int = DOC_TOKEN_BUDGET) -> str:
    """Return ``doc`` shortened to ``budget`` tokens (one LLM pass) if it is over."""
    tokens = count_tokens(doc)
    if tokens <= budget:
        return doc
    logger.info("  over budget (%d > %d tokens); shortening...", tokens, budget)
    shortened = generate(
        SHORTEN_PROMPT.format(tokens=tokens, budget=budget, document=doc),
        model=model,
        max_budget_usd=0.50,
    )
    final = count_tokens(shortened)
    if final > budget:
        logger.warning("  still over budget after shorten (%d tokens); keeping shortened version", final)
    return shortened


def _generate_axis(axis: str, template: str, project: str, digest: str, model: str) -> str:
    """Generate one axis doc for a sub-project and enforce its budget."""
    prompt = template.format(
        project_name=project,
        input_description=f"Given the public-API digest below for sub-project `{project}`,",
        sources=digest,
    )
    doc = generate(prompt, model=model, max_budget_usd=1.0)
    return _enforce_budget(doc, model)


def generate_map(overviews: dict[str, str], output_dir: Path, model: str) -> None:
    """Generate the top-level MAP.md from the per-sub-project overviews."""
    blocks = [f"## {project}\n\n{text}" for project, text in sorted(overviews.items())]
    prompt = TAXONOMY_MAP_PROMPT.format(overviews="\n\n---\n\n".join(blocks))
    doc = _enforce_budget(generate(prompt, model=model, max_budget_usd=0.50), model)
    (output_dir / "MAP.md").write_text(doc.rstrip() + "\n")
    logger.info("Wrote MAP.md (%d tokens)", count_tokens(doc))


def generate_taxonomy(
    packages: dict[str, PackageInfo],
    subprojects: list[str],
    output_dir: Path,
    *,
    model: str,
    dry_run: bool,
) -> None:
    """Generate the full taxonomy for the given sub-projects."""
    by_project = discover_subprojects(packages)
    overviews: dict[str, str] = {}

    for project in subprojects:
        names = by_project.get(project)
        if not names:
            logger.warning("Sub-project %r has no packages with public items; skipping", project)
            continue

        digest, truncated = build_digest(packages, names)
        arch_digest, _ = build_digest(packages, names, include_internals=True)
        logger.info(
            "%s: %d packages, digest %d chars (arch %d chars)%s",
            project,
            len(names),
            len(digest),
            len(arch_digest),
            " (truncated)" if truncated else "",
        )
        if dry_run:
            continue

        project_dir = output_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        for axis, template in AXES:
            t0 = time.monotonic()
            # The architecture axis answers "where would I change this", so it
            # gets the internals-inclusive digest; ops/overview stay public-only.
            axis_digest = arch_digest if axis == "architecture" else digest
            doc = _generate_axis(axis, template, project, axis_digest, model)
            (project_dir / f"{axis}.md").write_text(doc.rstrip() + "\n")
            logger.info("  %s/%s.md: %d tokens (%.1fs)", project, axis, count_tokens(doc), time.monotonic() - t0)
            if axis == "overview":
                overviews[project] = doc

    if dry_run:
        logger.info("[dry-run] would generate %d sub-projects + MAP.md", len(subprojects))
        return

    if overviews:
        generate_map(overviews, output_dir, model)


@click.command()
@click.option("--output-dir", default="docs/agent", type=click.Path(), help="Where the taxonomy is written.")
@click.option("--subproject", "subprojects", multiple=True, help="Sub-projects to generate (repeatable). Default: all.")
@click.option("--model", default="sonnet", help="Generation model.")
@click.option("--dry-run", is_flag=True, help="Report digests without LLM calls.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(output_dir: str, subprojects: tuple[str, ...], model: str, dry_run: bool, verbose: bool) -> None:
    """Generate the budgeted two-axis agent-doc taxonomy."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).parent.parent.parent
    logger.info("Discovering packages...")
    packages = discover_packages(repo_root)
    by_project = discover_subprojects(packages)

    targets = list(subprojects) if subprojects else sorted(by_project)
    unknown = [p for p in targets if p not in by_project]
    if unknown:
        raise click.UsageError(
            f"Unknown sub-project(s): {', '.join(unknown)}. Available: {', '.join(sorted(by_project))}"
        )

    logger.info("Target sub-projects (%d): %s", len(targets), ", ".join(targets))
    generate_taxonomy(packages, targets, Path(output_dir), model=model, dry_run=dry_run)


if __name__ == "__main__":
    main()
