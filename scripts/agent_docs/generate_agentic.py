#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click>=8.0",
#     "tiktoken>=0.7",
#     "tree-sitter==0.24.0",
#     "tree-sitter-python==0.23.6",
#     "tree-sitter-rust==0.23.2",
#     "pyyaml>=6.0",
# ]
# ///
"""Agentic agent-doc generation: a tool-using agent writes each budgeted doc.

The mechanical generator (``generate_taxonomy.py``) compresses a ~100k-token
signatures digest into a 1000-token doc in a single tool-less call, which forces
it to drop symbols the downstream coder then fabricates. This generator instead
gives an agent read-only tools (Read/Grep/Glob) and the repo, and lets it explore
real source to write each budgeted doc — every symbol grounded in code it read.

Same output layout as the mechanical track (``<project>/{overview,ops,
architecture}.md`` + ``MAP.md``) so the probe-eval harness scores both identically.

Spend is capped (CLI API-equivalent cost) so an experiment round stays bounded.
"""

import logging
import sys
import time
from pathlib import Path

import click

# Add scripts/ to path so agent_docs is importable as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.claude_cli import generate, generate_with_tools
from agent_docs.generate_taxonomy import DOC_TOKEN_BUDGET, SUBPROJECT_ALIASES, _enforce_budget
from agent_docs.parsing import PYTHON_LIBS, RUST_CRATES
from agent_docs.prompts import (
    AGENTIC_ARCHITECTURE_PROMPT,
    AGENTIC_OPS_PROMPT,
    AGENTIC_OVERVIEW_PROMPT,
    TAXONOMY_MAP_PROMPT,
)
from agent_docs.tokens import count_tokens

logger = logging.getLogger(__name__)

# Read-only toolset: the agent explores source but cannot modify the repo.
ALLOWED_TOOLS = ["Read", "Grep", "Glob"]
PER_DOC_BUDGET_USD = 1.5
DEFAULT_TOTAL_BUDGET_USD = 10.0

AXES = (
    ("overview", AGENTIC_OVERVIEW_PROMPT),
    ("ops", AGENTIC_OPS_PROMPT),
    ("architecture", AGENTIC_ARCHITECTURE_PROMPT),
)


def source_roots(subproject: str) -> list[str]:
    """Source directories an agent should explore for a sub-project.

    Combines the Python lib root with any Rust crates that alias to the same
    sub-project (e.g. ``finelog`` spans Python + the ``finelog_rust`` crate).
    """
    roots: list[str] = []
    if subproject in PYTHON_LIBS:
        roots.append(PYTHON_LIBS[subproject])
    for crate, path in RUST_CRATES.items():
        if SUBPROJECT_ALIASES.get(crate, crate) == subproject:
            roots.append(path)
    return roots


def generate_taxonomy_agentic(
    subprojects: list[str],
    output_dir: Path,
    *,
    model: str,
    budget_usd: float,
    dry_run: bool,
) -> float:
    """Generate the taxonomy agentically. Returns total API-equivalent spend.

    Stops early (and logs) if ``budget_usd`` is reached, so a round never blows
    the cap. Generated docs land in the same layout the eval harness expects.
    """
    repo_root = str(Path(__file__).parent.parent.parent)
    overviews: dict[str, str] = {}
    spent = 0.0

    for project in subprojects:
        roots = source_roots(project)
        if not roots:
            logger.warning("No source roots for %r; skipping", project)
            continue
        roots_block = "\n".join(f"- `{r}`" for r in roots)
        logger.info("%s: roots %s", project, ", ".join(roots))
        if dry_run:
            continue

        project_dir = output_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        for axis, template in AXES:
            if spent >= budget_usd:
                logger.warning("Budget $%.2f reached; stopping before %s/%s", budget_usd, project, axis)
                return spent
            prompt = template.format(project_name=project, source_roots=roots_block)
            t0 = time.monotonic()
            doc, cost = generate_with_tools(
                prompt,
                model=model,
                allowed_tools=ALLOWED_TOOLS,
                cwd=repo_root,
                max_budget_usd=min(PER_DOC_BUDGET_USD, budget_usd - spent),
            )
            spent += cost
            doc = _enforce_budget(doc, model)
            (project_dir / f"{axis}.md").write_text(doc.rstrip() + "\n")
            logger.info(
                "  %s/%s.md: %d tokens, $%.2f (%.0fs) [spent $%.2f/$%.2f]",
                project,
                axis,
                count_tokens(doc),
                cost,
                time.monotonic() - t0,
                spent,
                budget_usd,
            )
            if axis == "overview":
                overviews[project] = doc

    if overviews and not dry_run:
        blocks = [f"## {project}\n\n{text}" for project, text in sorted(overviews.items())]
        map_doc = _enforce_budget(
            generate(TAXONOMY_MAP_PROMPT.format(overviews="\n\n---\n\n".join(blocks)), model=model, max_budget_usd=0.5),
            model,
        )
        (output_dir / "MAP.md").write_text(map_doc.rstrip() + "\n")
        logger.info("Wrote MAP.md (%d tokens)", count_tokens(map_doc))

    logger.info("Agentic generation spent $%.2f (API-equivalent) of $%.2f", spent, budget_usd)
    return spent


@click.command()
@click.option("--output-dir", default="docs/agent", type=click.Path(), help="Where the taxonomy is written.")
@click.option("--subproject", "subprojects", multiple=True, required=True, help="Sub-projects to generate (repeatable).")
@click.option("--model", default="sonnet", help="Generation model (sonnet or opus).")
@click.option("--budget-usd", default=DEFAULT_TOTAL_BUDGET_USD, type=float, help="Total spend cap (API-equivalent).")
@click.option("--dry-run", is_flag=True, help="Report source roots without LLM calls.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    output_dir: str, subprojects: tuple[str, ...], model: str, budget_usd: float, dry_run: bool, verbose: bool
) -> None:
    """Generate the agent-doc taxonomy agentically (tool-using agent per doc)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    spent = generate_taxonomy_agentic(
        list(subprojects), Path(output_dir), model=model, budget_usd=budget_usd, dry_run=dry_run
    )
    if not dry_run:
        logger.info("Done. Budget used: $%.2f / $%.2f. Doc budget: %d tokens each.", spent, budget_usd, DOC_TOKEN_BUDGET)


if __name__ == "__main__":
    main()
