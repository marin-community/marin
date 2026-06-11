#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["click"]
# ///
"""Score a generated agent-doc set against the 10-probe suite.

For each probe: assemble the doc bundle (the doc(s) a well-organized agent should
consult, plus MAP.md), have a coder model write a script, then have a judge model
review the script against the probe's source-verified anchors plus a holistic
quality rubric. The coder is a realistic agent: it leads with the docs but may
open a FEW files (read-only Read/Grep/Glob, NO execution) under a tight spend
cap. So the score measures how well the docs ROUTE a capable, budget-limited
agent to the right source — good docs point it at the right files fast; bad docs
make it waste its budget or read the wrong things.

Generalizes the former single-probe validation.py; P8 carries the fuzzy-dedup
regression anchors verbatim.

Usage:
    ./scripts/agent_docs/eval.py --list
    ./scripts/agent_docs/eval.py --docs-dir docs/agent --output-dir /tmp/autodoc-eval
    ./scripts/agent_docs/eval.py --probe P8 --output-dir /tmp/autodoc-eval
"""

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import click

# Add scripts/ to path so agent_docs package is importable as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_docs.claude_cli import generate_json, run_agent_json, strip_markdown_fences
from agent_docs.probes import PROBES, Probe
from agent_docs.prompts import CODER_PROMPT, JUDGE_PROMPT

logger = logging.getLogger(__name__)

MAP_FILENAME = "MAP.md"
# A probe passes on documentation grounds: the key APIs are used correctly and
# nothing is hallucinated. Quality (1-5) is reported as a secondary signal, not a
# gate — it was flat (~1.3) across rounds and did not discriminate.
RUBRIC_PASS_THRESHOLD = 0.8

# The coder is a realistic, budget-limited agent: read-only tools (no execution)
# and a tight spend cap that, with no --max-turns flag available, bounds it to a
# few file reads before it must write the script. Run from the repo root so its
# reads resolve.
CODER_ALLOWED_TOOLS = ["Read", "Grep", "Glob"]
CODER_MAX_BUDGET_USD = 0.30
REPO_ROOT = Path(__file__).parent.parent.parent

CODER_SYSTEM_PROMPT = (
    "You are a Python engineer on the Marin monorepo with read-only tools and a tight budget. "
    "Lead with the provided docs; open at most a few files only to confirm details. "
    "Your final message must be only the script."
)
JUDGE_SYSTEM_PROMPT = "You are a senior engineer reviewing code. Output only the requested JSON."

PROBES_BY_ID = {p.id: p for p in PROBES}


@dataclass(frozen=True)
class CoderResult:
    """The coder's emitted script plus cost/usage metadata."""

    script: str
    cost_usd: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class JudgeResult:
    """The judge's scoring of one script against a probe's ground truth."""

    anchor_scores: dict[str, int]  # symbol -> 0/1
    hallucinated: list[str]
    quality: int  # 1-5
    notes: str
    cost_usd: float
    raw: str


@dataclass(frozen=True)
class ProbeResult:
    """Combined per-probe outcome."""

    probe_id: str
    subproject: str
    intent: str
    bundle_chars: int
    script: str
    coder: CoderResult
    judge: JudgeResult
    rubric_correct: int
    rubric_total: int
    rubric_acc: float
    quality: int
    passed: bool
    total_cost_usd: float


def build_bundle(probe: Probe, docs_dir: Path) -> str:
    """Concatenate MAP.md (if present) and the probe's doc bundle with headers.

    A missing bundle file is expected here — the doc set under test may be
    incomplete — and is recorded inline rather than raised, so the score
    reflects the gap instead of crashing.
    """
    parts: list[str] = []

    map_path = docs_dir / MAP_FILENAME
    if map_path.exists():
        parts.append(f"# {MAP_FILENAME}\n\n{map_path.read_text()}")
    else:
        parts.append(f"# {MAP_FILENAME}\n\n(doc not found: {map_path})")

    for rel in probe.doc_bundle:
        path = docs_dir / rel
        if path.exists():
            parts.append(f"# {rel}\n\n{path.read_text()}")
        else:
            logger.warning("Probe %s: doc not found: %s", probe.id, path)
            parts.append(f"# {rel}\n\n(doc not found: {path})")

    return "\n\n---\n\n".join(parts)


def run_coder(bundle: str, probe: Probe, model: str) -> CoderResult:
    """Ask the budget-limited coder to write the probe's script.

    The coder leads with the doc bundle but may open a few files (read-only)
    under CODER_MAX_BUDGET_USD; it cannot execute anything. This measures how
    well the docs route a capable agent to the right source within a tight budget.
    """
    prompt = CODER_PROMPT.format(bundle=bundle, task=probe.prompt)
    data = run_agent_json(
        prompt,
        model=model,
        allowed_tools=CODER_ALLOWED_TOOLS,
        cwd=str(REPO_ROOT),
        system_prompt=CODER_SYSTEM_PROMPT,
        max_budget_usd=CODER_MAX_BUDGET_USD,
    )
    script = strip_markdown_fences(data.get("result", ""))
    usage = data.get("usage", {})
    return CoderResult(
        script=script,
        cost_usd=data.get("total_cost_usd", 0.0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


def _format_anchors(probe: Probe) -> str:
    return "\n".join(f"- `{a.symbol}` @ `{a.path}` — {a.note}" for a in probe.anchors)


def _format_forbidden(probe: Probe) -> str:
    if not probe.forbidden:
        return ""
    terms = ", ".join(f"`{t}`" for t in probe.forbidden)
    return (
        "## Forbidden terms (hallucination markers)\n\n"
        f"If the script uses any of these, list it in `hallucinated`: {terms}\n\n"
    )


def run_judge(script: str, probe: Probe, model: str) -> JudgeResult:
    """Review the script against the probe's anchors and quality rubric."""
    prompt = JUDGE_PROMPT.format(
        task=probe.prompt,
        intent=str(probe.intent).upper(),
        anchors=_format_anchors(probe),
        forbidden_block=_format_forbidden(probe),
        script=script,
    )
    data = generate_json(prompt, model=model, system_prompt=JUDGE_SYSTEM_PROMPT)
    raw = data.get("result", "")
    cost = data.get("total_cost_usd", 0.0)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Probe {probe.id}: judge returned no JSON object:\n{raw[:500]}")
    parsed = json.loads(match.group())

    raw_anchor_scores = parsed.get("anchors", {})
    anchor_scores = {a.symbol: int(raw_anchor_scores.get(a.symbol, 0)) for a in probe.anchors}
    return JudgeResult(
        anchor_scores=anchor_scores,
        hallucinated=list(parsed.get("hallucinated", [])),
        quality=int(parsed.get("quality", 0)),
        notes=str(parsed.get("notes", "")),
        cost_usd=cost,
        raw=raw,
    )


def score_probe(probe: Probe, docs_dir: Path, gen_model: str, review_model: str) -> ProbeResult:
    """Run one probe end to end: bundle -> coder -> judge -> scores."""
    bundle = build_bundle(probe, docs_dir)
    logger.info("Probe %s: bundle %d chars; asking %s for a script...", probe.id, len(bundle), gen_model)
    coder = run_coder(bundle, probe, gen_model)

    logger.info("Probe %s: reviewing with %s...", probe.id, review_model)
    judge = run_judge(coder.script, probe, review_model)

    rubric_correct = sum(judge.anchor_scores.values())
    rubric_total = len(probe.anchors)
    rubric_acc = rubric_correct / rubric_total if rubric_total else 0.0
    passed = rubric_acc >= RUBRIC_PASS_THRESHOLD and not judge.hallucinated

    return ProbeResult(
        probe_id=probe.id,
        subproject=str(probe.subproject),
        intent=str(probe.intent),
        bundle_chars=len(bundle),
        script=coder.script,
        coder=coder,
        judge=judge,
        rubric_correct=rubric_correct,
        rubric_total=rubric_total,
        rubric_acc=rubric_acc,
        quality=judge.quality,
        passed=passed,
        total_cost_usd=coder.cost_usd + judge.cost_usd,
    )


def _probe_result_to_dict(result: ProbeResult) -> dict:
    data = asdict(result)
    # asdict already recurses into nested dataclasses; keep the script out of the
    # nested coder dict to avoid duplicating it (top-level "script" carries it).
    data["coder"].pop("script", None)
    return data


def write_probe_artifacts(result: ProbeResult, output_dir: Path) -> None:
    """Write `<probe_id>/script.py` and `<probe_id>/result.json`."""
    probe_dir = output_dir / result.probe_id
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "script.py").write_text(result.script + "\n")
    (probe_dir / "result.json").write_text(json.dumps(_probe_result_to_dict(result), indent=2) + "\n")


def write_summary(results: list[ProbeResult], output_dir: Path, gen_model: str, review_model: str) -> dict:
    """Write `summary.json` with per-probe and suite-level aggregates."""
    per_probe = [
        {
            "id": r.probe_id,
            "subproject": r.subproject,
            "intent": r.intent,
            "rubric_correct": r.rubric_correct,
            "rubric_total": r.rubric_total,
            "rubric_acc": r.rubric_acc,
            "quality": r.quality,
            "hallucinated": r.judge.hallucinated,
            "passed": r.passed,
            # coder_cost is the routing-efficiency signal: with a capable coder
            # that may read files, good docs route it to the answer for less spend.
            "coder_cost_usd": r.coder.cost_usd,
            "cost_usd": r.total_cost_usd,
        }
        for r in results
    ]
    n = len(results)
    summary = {
        "gen_model": gen_model,
        "review_model": review_model,
        "mean_coder_cost_usd": sum(r.coder.cost_usd for r in results) / n if n else 0.0,
        "num_probes": n,
        "mean_rubric_acc": sum(r.rubric_acc for r in results) / n if n else 0.0,
        "mean_quality": sum(r.quality for r in results) / n if n else 0.0,
        "num_passed": sum(1 for r in results if r.passed),
        "total_cost_usd": sum(r.total_cost_usd for r in results),
        "probes": per_probe,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def print_summary(results: list[ProbeResult], summary: dict) -> None:
    """Print a per-probe table and the suite-level aggregate."""
    separator = "=" * 78
    print(f"\n{separator}")
    print("  Agent-Docs Eval: per-probe scores")
    print(separator)
    header = f"  {'ID':<4} {'SUBPROJECT':<10} {'INTENT':<11} {'RUBRIC':<8} {'QUAL':<5} {'RESULT':<6}"
    print(header)
    print("-" * 78)
    for r in results:
        rubric = f"{r.rubric_correct}/{r.rubric_total}"
        verdict = "PASS" if r.passed else "FAIL"
        flag = "  (hallucination)" if r.judge.hallucinated else ""
        print(f"  {r.probe_id:<4} {r.subproject:<10} {r.intent:<11} {rubric:<8} {r.quality:<5} {verdict:<6}{flag}")
    print("-" * 78)
    print(
        f"  mean rubric acc: {summary['mean_rubric_acc']:.2f}   "
        f"mean quality: {summary['mean_quality']:.2f}   "
        f"passed: {summary['num_passed']}/{summary['num_probes']}"
    )
    print(f"  total cost: ${summary['total_cost_usd']:.4f}")
    print(f"{separator}\n")


def print_probe_list() -> None:
    """Print the 10 probes (id, subproject, intent, one-line) with no API calls."""
    print(f"{'ID':<4} {'SUBPROJECT':<10} {'INTENT':<11} TASK")
    print("-" * 100)
    for p in PROBES:
        first_line = p.prompt.strip().splitlines()[0]
        print(f"{p.id:<4} {p.subproject!s:<10} {p.intent!s:<11} {first_line}")


@click.command()
@click.option("--docs-dir", default="docs/agent", type=click.Path(), help="Root of the doc taxonomy to evaluate.")
@click.option("--probe", "probe_ids", multiple=True, help="Run only these probe ids (repeatable). Default: all.")
@click.option("--list", "list_probes", is_flag=True, help="List the 10 probes and exit (no API calls).")
@click.option("--gen-model", default="sonnet", help="Coder model (writes scripts; reads a few files, no execution).")
@click.option("--review-model", default="sonnet", help="Judge model (reviews scripts).")
@click.option("--output-dir", type=click.Path(), default=None, help="Where artifacts go (required unless --list).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    docs_dir: str,
    probe_ids: tuple[str, ...],
    list_probes: bool,
    gen_model: str,
    review_model: str,
    output_dir: str | None,
    verbose: bool,
) -> None:
    """Score a doc set against the probe suite."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if list_probes:
        print_probe_list()
        return

    if output_dir is None:
        raise click.UsageError("--output-dir is required unless --list is given.")

    if probe_ids:
        unknown = [pid for pid in probe_ids if pid not in PROBES_BY_ID]
        if unknown:
            raise click.UsageError(f"Unknown probe id(s): {', '.join(unknown)}")
        selected = [PROBES_BY_ID[pid] for pid in probe_ids]
    else:
        selected = list(PROBES)

    docs_path = Path(docs_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []
    for probe in selected:
        result = score_probe(probe, docs_path, gen_model, review_model)
        write_probe_artifacts(result, out)
        results.append(result)

    summary = write_summary(results, out, gen_model, review_model)
    print_summary(results, summary)


if __name__ == "__main__":
    main()
