# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Context-efficiency analysis pipeline — a single ``ArtifactStep`` DAG that measures
how much agent token budget a memory/wiki/index/tooling change would save, from your
own ``~/.claude`` session transcripts.

Run it end to end (outputs land under ``$MARIN_PREFIX/context_efficiency/*``, or
``/tmp/marin`` if unset)::

    MARIN_PREFIX=~/scratch/ce python -m experiments.context_efficiency.pipeline \\
        --agent-command 'claude -p --model haiku' \\
        --val-agent-command 'claude -p --model sonnet'

The stages, each reading the previous stage's structured output:

    parse ─┬─ accounting ─┐
           ├─ budget ─────┤
           └─ sample ─┬─ label ─┬─ cluster ─┐
                      │         └───────────┼─ analysis  (final: semantic_analysis.json)
                      └─ label_val ─────────┘

``label``/``label_val`` shell out to the configured headless agent (``claude -p`` by
default; ``codex exec`` or any other works). ``label_val`` re-labels a fraction with a
stronger model to calibrate the bulk labeler's optimism. Everything else is local
compute. Step identity is content-addressed: re-running skips finished stages, and
labeling is batch-idempotent, so an interrupted run resumes cheaply.
"""

import argparse
import logging
import os
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner

from experiments.context_efficiency.accounting import AccountingConfig, BudgetConfig, run_accounting, run_budget
from experiments.context_efficiency.analysis import AnalysisConfig, run_analysis
from experiments.context_efficiency.episodes import EpisodesConfig, run_sampling
from experiments.context_efficiency.labeling import ClusterConfig, LabelConfig, run_clustering, run_labeling
from experiments.context_efficiency.transcripts import ParseConfig, run_parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Everything that parameterizes a run. Identity-bearing values (paths, sample
    size, agent commands) fork a fresh output; execution knobs (concurrency, timeout)
    are passed as runtime args so tuning them never re-runs a finished stage."""

    projects_dir: str
    session_glob: str
    limit: int
    agents_md_path: str
    memory_md_path: str
    n_episodes: int
    batch_size: int
    seed: int
    agent_command: str
    val_agent_command: str
    val_fraction: float
    label_concurrency: int
    agent_timeout: int
    label_retries: int
    min_coverage: float
    version: str


def build(cfg: PipelineConfig) -> ArtifactStep[Artifact]:
    """Wire the full DAG and return the terminal analysis step."""
    v = cfg.version

    parse = ArtifactStep(
        name="context_efficiency/parse",
        version=v,
        artifact_type=Artifact,
        run=run_parse,
        build_config=lambda ctx: ParseConfig(
            projects_dir=cfg.projects_dir,
            session_glob=cfg.session_glob,
            limit=cfg.limit,
            output_path=ctx.output_path,
        ),
    )

    accounting = ArtifactStep(
        name="context_efficiency/accounting",
        version=v,
        artifact_type=Artifact,
        run=run_accounting,
        build_config=lambda ctx: AccountingConfig(sessions_path=ctx.artifact_path(parse), output_path=ctx.output_path),
        deps=(parse,),
    )

    budget = ArtifactStep(
        name="context_efficiency/budget",
        version=v,
        artifact_type=Artifact,
        run=run_budget,
        build_config=lambda ctx: BudgetConfig(
            sessions_path=ctx.artifact_path(parse),
            agents_md_path=cfg.agents_md_path,
            memory_md_path=cfg.memory_md_path,
            output_path=ctx.output_path,
        ),
        deps=(parse,),
    )

    sample = ArtifactStep(
        name="context_efficiency/sample",
        version=v,
        artifact_type=Artifact,
        run=run_sampling,
        build_config=lambda ctx: EpisodesConfig(
            sessions_path=ctx.artifact_path(parse),
            accounting_path=ctx.artifact_path(accounting),
            projects_dir=cfg.projects_dir,
            session_glob=cfg.session_glob,
            n=cfg.n_episodes,
            batch=cfg.batch_size,
            seed=cfg.seed,
            output_path=ctx.output_path,
        ),
        deps=(parse, accounting),
    )

    label_runtime = {
        "concurrency": cfg.label_concurrency,
        "timeout": cfg.agent_timeout,
        "retries": cfg.label_retries,
        "min_coverage": cfg.min_coverage,
    }

    def _label_config(ctx: StepContext, agent_command: str, fraction: float) -> LabelConfig:
        return LabelConfig(
            episodes_path=ctx.artifact_path(sample),
            output_path=ctx.output_path,
            agent_command=agent_command,
            fraction=fraction,
            concurrency=ctx.runtime_arg("concurrency"),
            timeout=ctx.runtime_arg("timeout"),
            retries=ctx.runtime_arg("retries"),
            min_coverage=ctx.runtime_arg("min_coverage"),
        )

    label = ArtifactStep(
        name="context_efficiency/label",
        version=v,
        artifact_type=Artifact,
        run=run_labeling,
        build_config=lambda ctx: _label_config(ctx, cfg.agent_command, 1.0),
        deps=(sample,),
        runtime_args=label_runtime,
    )

    label_val = ArtifactStep(
        name="context_efficiency/label_val",
        version=v,
        artifact_type=Artifact,
        run=run_labeling,
        build_config=lambda ctx: _label_config(ctx, cfg.val_agent_command, cfg.val_fraction),
        deps=(sample,),
        runtime_args=label_runtime,
    )

    cluster = ArtifactStep(
        name="context_efficiency/cluster",
        version=v,
        artifact_type=Artifact,
        run=run_clustering,
        build_config=lambda ctx: ClusterConfig(
            labels_path=ctx.artifact_path(label),
            output_path=ctx.output_path,
            agent_command=cfg.agent_command,
            timeout=ctx.runtime_arg("timeout"),
        ),
        deps=(label,),
        runtime_args={"timeout": cfg.agent_timeout},
    )

    analysis = ArtifactStep(
        name="context_efficiency/analysis",
        version=v,
        artifact_type=Artifact,
        run=run_analysis,
        build_config=lambda ctx: AnalysisConfig(
            episodes_path=ctx.artifact_path(sample),
            labels_path=ctx.artifact_path(label),
            labels_val_path=ctx.artifact_path(label_val),
            clusters_path=ctx.artifact_path(cluster),
            budget_path=ctx.artifact_path(budget),
            output_path=ctx.output_path,
        ),
        deps=(sample, label, label_val, cluster, budget),
    )
    return analysis


def _parse_args() -> tuple[PipelineConfig, int]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"), help="~/.claude projects dir")
    ap.add_argument("--glob", default="*", help="project-dir glob, e.g. '-home-you-code-repo*'")
    ap.add_argument("--limit", type=int, default=0, help="cap session files (debug)")
    ap.add_argument("--agents-md", default="AGENTS.md", help="repo AGENTS.md, for prelude decomposition")
    ap.add_argument("--memory-md", default="", help="MEMORY.md index, for prelude decomposition")
    ap.add_argument("--n", type=int, default=2000, help="target labeled episodes")
    ap.add_argument("--batch", type=int, default=15, help="episodes per labeling shard")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--agent-command", default="claude -p", help="headless bulk labeler CLI")
    ap.add_argument("--val-agent-command", default="claude -p", help="stronger-model validator CLI")
    ap.add_argument(
        "--val-fraction", type=float, default=0.12, help="fraction of batches to re-label for calibration (0 disables)"
    )
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent agent invocations")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per agent invocation")
    ap.add_argument("--retries", type=int, default=1, help="retries per batch on a malformed reply")
    ap.add_argument("--min-coverage", type=float, default=0.9, help="fail labeling below this labeled-batch fraction")
    ap.add_argument("--version", default="dev", help="artifact version tag")
    ap.add_argument("--max-concurrent", type=int, default=4, help="concurrent pipeline steps")
    args = ap.parse_args()
    cfg = PipelineConfig(
        projects_dir=args.projects,
        session_glob=args.glob,
        limit=args.limit,
        agents_md_path=args.agents_md,
        memory_md_path=args.memory_md,
        n_episodes=args.n,
        batch_size=args.batch,
        seed=args.seed,
        agent_command=args.agent_command,
        val_agent_command=args.val_agent_command,
        val_fraction=args.val_fraction,
        label_concurrency=args.concurrency,
        agent_timeout=args.timeout,
        label_retries=args.retries,
        min_coverage=args.min_coverage,
        version=args.version,
    )
    return cfg, args.max_concurrent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, max_concurrent = _parse_args()
    StepRunner().run([build(cfg).lower()], max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
