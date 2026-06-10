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
"""Run one experiment round: generate the taxonomy, eval the probe suite, record.

A "round" is one (doc-generation prompt) x (probe suite) measurement. Between
rounds you edit the prompts/anchors in source; each round is a fresh invocation
with a new --label. Results append to ``<expdir>/results.jsonl`` so the final
report can summarize the whole sweep, and the scoreboard prints with a delta
against the previous round.

Reuse docs from a prior round with --reuse-docs to test eval/judge changes
without paying for regeneration.

    # generate fresh docs + eval (a doc-prompt change):
    ./scripts/agent_docs/run_round.py --label R11
    # reuse R9 docs, just re-eval (a judge/anchor change):
    ./scripts/agent_docs/run_round.py --label R10 --reuse-docs /tmp/autodoc-exp/round3/docs
"""

import json
import logging
import sys
from pathlib import Path

import click

# Add scripts/ to path so agent_docs is importable as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.eval import score_probe, write_probe_artifacts, write_summary
from agent_docs.generate_taxonomy import generate_taxonomy
from agent_docs.packages import discover_packages
from agent_docs.probes import PROBES

logger = logging.getLogger(__name__)

DEFAULT_EXPDIR = Path("/tmp/autodoc-exp")
# The sub-projects the 10 probes target; default generation scope.
PROBE_SUBPROJECTS = ("iris", "marin", "levanter", "finelog")
PROBES_BY_ID = {p.id: p for p in PROBES}


def _append_ledger(expdir: Path, row: dict) -> list[dict]:
    """Append a round result row to the JSONL ledger; return all rows."""
    ledger = expdir / "results.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")
    rows.append(row)
    return rows


def _print_round(row: dict, prev: dict | None) -> None:
    """Print the round scoreboard with a per-probe table and a delta vs prev."""
    sep = "=" * 84
    print(f"\n{sep}\n  ROUND {row['label']}  (gen={row['gen_model']} judge={row['review_model']})\n{sep}")
    print(f"  {'ID':<4}{'SUBPROJ':<9}{'INTENT':<11}{'RUBRIC':<8}{'QUAL':<5}{'PASS':<5}HALLUC")
    print("-" * 84)
    for p in row["probes"]:
        rub = f"{p['rubric_correct']}/{p['rubric_total']}"
        hall = ", ".join(p["hallucinated"])[:38] or "-"
        passed = "Y" if p["passed"] else "."
        cols = f"{p['id']:<4}{p['subproject']:<9}{p['intent']:<11}{rub:<8}{p['quality']:<5}{passed:<5}"
        print(f"  {cols}{hall}")
    print("-" * 84)
    line = (
        f"  mean rubric {row['mean_rubric_acc']:.2f}   mean quality {row['mean_quality']:.2f}   "
        f"passed {row['num_passed']}/{row['num_probes']}   cost(API-equiv) ${row['eval_cost_usd']:.2f}"
    )
    if prev:
        d_r = row["mean_rubric_acc"] - prev["mean_rubric_acc"]
        d_q = row["mean_quality"] - prev["mean_quality"]
        d_p = row["num_passed"] - prev["num_passed"]
        line += f"\n  delta vs {prev['label']}: rubric {d_r:+.2f}  quality {d_q:+.2f}  passed {d_p:+d}"
    print(line + f"\n{sep}\n")


@click.command()
@click.option("--label", required=True, help="Round label, e.g. R10 (used for the result row and output dir).")
@click.option(
    "--reuse-docs", type=click.Path(exists=True), default=None, help="Reuse an existing docs dir; skip generation."
)
@click.option("--subproject", "subprojects", multiple=True, help="Sub-projects to generate (default: probe-targeted).")
@click.option("--probe", "probe_ids", multiple=True, help="Probe ids to run (default: all).")
@click.option("--gen-model", default="sonnet", help="Doc generation model.")
@click.option("--review-model", default="sonnet", help="Judge model.")
@click.option("--coder-model", default="haiku", help="Coder model (writes scripts, no tools).")
@click.option("--expdir", type=click.Path(), default=str(DEFAULT_EXPDIR), help="Experiment root dir.")
@click.option("-v", "--verbose", is_flag=True)
def main(
    label: str,
    reuse_docs: str | None,
    subprojects: tuple[str, ...],
    probe_ids: tuple[str, ...],
    gen_model: str,
    review_model: str,
    coder_model: str,
    expdir: str,
    verbose: bool,
) -> None:
    """Run one round (generate or reuse docs -> eval -> record)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    exp = Path(expdir)
    round_dir = exp / label
    round_dir.mkdir(parents=True, exist_ok=True)

    if reuse_docs:
        docs_dir = Path(reuse_docs)
        logger.info("[%s] reusing docs from %s (skip generation)", label, docs_dir)
    else:
        docs_dir = round_dir / "docs"
        targets = list(subprojects) if subprojects else list(PROBE_SUBPROJECTS)
        logger.info("[%s] generating docs for %s -> %s", label, ", ".join(targets), docs_dir)
        packages = discover_packages(Path(__file__).parent.parent.parent)
        generate_taxonomy(packages, targets, docs_dir, model=gen_model, dry_run=False)

    selected = [PROBES_BY_ID[p] for p in probe_ids] if probe_ids else list(PROBES)
    eval_dir = round_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for probe in selected:
        result = score_probe(probe, docs_dir, coder_model, review_model)
        write_probe_artifacts(result, eval_dir)
        results.append(result)
    summary = write_summary(results, eval_dir, coder_model, review_model)

    row = {
        "label": label,
        "gen_model": "(reused)" if reuse_docs else gen_model,
        "review_model": review_model,
        "coder_model": coder_model,
        "docs_dir": str(docs_dir),
        "mean_rubric_acc": summary["mean_rubric_acc"],
        "mean_quality": summary["mean_quality"],
        "num_passed": summary["num_passed"],
        "num_probes": summary["num_probes"],
        "eval_cost_usd": summary["total_cost_usd"],
        "probes": summary["probes"],
    }
    rows = _append_ledger(exp, row)
    prev = rows[-2] if len(rows) > 1 else None
    _print_round(row, prev)


if __name__ == "__main__":
    main()
