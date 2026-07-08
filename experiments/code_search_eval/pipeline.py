# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code-search evaluation pipeline — a single ``ArtifactStep`` DAG that benchmarks
local semantic code-search engines against a ripgrep baseline on queries mined from
your own agent sessions.

Run it end to end (outputs land under ``$MARIN_PREFIX/code_search_eval/*``)::

    MARIN_PREFIX=~/scratch/cse python -m experiments.code_search_eval.pipeline \\
        --repo /home/you/code/marin \\
        --glob '-home-you-code-marin*' \\
        --agent-command 'claude -p'

The stages::

    benchmark ─┬─────────────────────────────────────────────┐
               │  per engine:  build ─→ query ─→ judge ───────┤
               └─────────────────────────────────────────────┴─ score (results.json/.md)

``benchmark`` mines navigation moments from transcripts and uses the agent to turn
each into a clean query with a gold file. Each engine ``build`` indexes the repo
(cached — independent of the benchmark), ``query`` returns top-K hits, and ``judge``
asks the agent whether the hits answer the need. ``score`` reports recall@k,
judge-hit@k, and answer tokens@k across engines.

Engines are ``uv run`` adapter scripts under ``engines/`` with their own isolated
deps; the pipeline shells out to them so heavy deps (embeddings, bm25) never touch the
marin environment.
"""

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from rigging.filesystem import prefix_join

from experiments.code_search_eval.benchmark import BenchmarkConfig, run_benchmark
from experiments.code_search_eval.common import DEFAULT_K
from experiments.code_search_eval.judge import JudgeConfig, run_judge
from experiments.code_search_eval.scoring import ScoringConfig, run_scoring

logger = logging.getLogger(__name__)

# repo root that contains `experiments/` — put on PYTHONPATH so adapter scripts (run in
# isolated uv envs) can import experiments.code_search_eval.common.
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines")
ENGINE_SCRIPTS = {
    "ripgrep": "ripgrep_engine.py",
    "bm25": "bm25_engine.py",
    "dense": "dense_engine.py",
    "seagoat": "seagoat_engine.py",
    "vectorcode": "vectorcode_engine.py",
}


@dataclass(frozen=True)
class EngineBuildConfig:
    engine: str
    repo_root: str
    index_dir: str
    embed_model: str


@dataclass(frozen=True)
class EngineQueryConfig:
    engine: str
    repo_root: str
    index_dir: str
    benchmark_path: str
    output_path: str
    k: int
    embed_model: str


def _engine_env(engine: str, embed_model: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": PKG_PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if engine == "dense":
        env["CSE_EMBED_MODEL"] = embed_model
    return env


def _run_adapter(engine: str, args: list[str], env: dict[str, str]) -> None:
    script = os.path.join(ENGINES_DIR, ENGINE_SCRIPTS[engine])
    cmd = ["uv", "run", "--no-project", script, *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{engine} adapter failed ({' '.join(args[:1])}): {(proc.stderr or '')[-800:]}")


def run_engine_build(cfg: EngineBuildConfig) -> None:
    os.makedirs(cfg.index_dir, exist_ok=True)
    _run_adapter(
        cfg.engine,
        ["build", "--repo", cfg.repo_root, "--index", cfg.index_dir],
        _engine_env(cfg.engine, cfg.embed_model),
    )
    logger.info("%s: index built at %s", cfg.engine, cfg.index_dir)


def run_engine_query(cfg: EngineQueryConfig) -> None:
    os.makedirs(cfg.output_path, exist_ok=True)
    out = prefix_join(cfg.output_path, f"{cfg.engine}_hits.jsonl")
    _run_adapter(
        cfg.engine,
        ["query", "--repo", cfg.repo_root, "--index", cfg.index_dir,
         "--queries", prefix_join(cfg.benchmark_path, "benchmark.jsonl"), "--out", out, "--k", str(cfg.k)],
        _engine_env(cfg.engine, cfg.embed_model),
    )  # fmt: skip
    logger.info("%s: wrote hits for k=%d", cfg.engine, cfg.k)


@dataclass(frozen=True)
class PipelineConfig:
    projects_dir: str
    session_glob: str
    repo_root: str
    limit: int
    max_candidates: int
    max_queries: int
    seed: int
    agent_command: str
    judge_agent_command: str
    bench_batch: int
    judge_batch: int
    agent_concurrency: int
    agent_timeout: int
    engines: tuple[str, ...]
    embed_model: str
    k: int
    k_values: tuple[int, ...]
    version: str


def build(cfg: PipelineConfig) -> ArtifactStep[Artifact]:
    """Wire the full DAG and return the terminal scoring step."""
    v = cfg.version
    agent_runtime = {"concurrency": cfg.agent_concurrency, "timeout": cfg.agent_timeout}

    benchmark = ArtifactStep(
        name="code_search_eval/benchmark",
        version=v,
        artifact_type=Artifact,
        run=run_benchmark,
        build_config=lambda ctx: BenchmarkConfig(
            projects_dir=cfg.projects_dir,
            session_glob=cfg.session_glob,
            repo_root=cfg.repo_root,
            limit=cfg.limit,
            max_candidates=cfg.max_candidates,
            max_queries=cfg.max_queries,
            seed=cfg.seed,
            agent_command=cfg.agent_command,
            concurrency=ctx.runtime_arg("concurrency"),
            timeout=ctx.runtime_arg("timeout"),
            batch=cfg.bench_batch,
            output_path=ctx.output_path,
        ),
        runtime_args=agent_runtime,
    )

    query_steps: list[tuple[str, ArtifactStep]] = []
    judge_steps: list[tuple[str, ArtifactStep]] = []
    build_steps: list[tuple[str, ArtifactStep]] = []
    for engine in cfg.engines:
        build_step = ArtifactStep(
            name=f"code_search_eval/index_{engine}",
            version=v,
            artifact_type=Artifact,
            run=run_engine_build,
            build_config=lambda ctx, e=engine: EngineBuildConfig(
                engine=e, repo_root=cfg.repo_root, index_dir=ctx.output_path, embed_model=cfg.embed_model
            ),
        )
        query_step = ArtifactStep(
            name=f"code_search_eval/query_{engine}",
            version=v,
            artifact_type=Artifact,
            run=run_engine_query,
            build_config=lambda ctx, e=engine, b=build_step: EngineQueryConfig(
                engine=e,
                repo_root=cfg.repo_root,
                index_dir=ctx.artifact_path(b),
                benchmark_path=ctx.artifact_path(benchmark),
                output_path=ctx.output_path,
                k=cfg.k,
                embed_model=cfg.embed_model,
            ),
            deps=(build_step, benchmark),
        )
        judge_step = ArtifactStep(
            name=f"code_search_eval/judge_{engine}",
            version=v,
            artifact_type=Artifact,
            run=run_judge,
            build_config=lambda ctx, e=engine, q=query_step: JudgeConfig(
                engine=e,
                benchmark_path=ctx.artifact_path(benchmark),
                hits_path=ctx.artifact_path(q),
                repo_root=cfg.repo_root,
                output_path=ctx.output_path,
                agent_command=cfg.judge_agent_command,
                concurrency=ctx.runtime_arg("concurrency"),
                timeout=ctx.runtime_arg("timeout"),
                batch=cfg.judge_batch,
            ),
            deps=(query_step, benchmark),
            runtime_args=agent_runtime,
        )
        build_steps.append((engine, build_step))
        query_steps.append((engine, query_step))
        judge_steps.append((engine, judge_step))

    score = ArtifactStep(
        name="code_search_eval/score",
        version=v,
        artifact_type=Artifact,
        run=run_scoring,
        build_config=lambda ctx: ScoringConfig(
            benchmark_path=ctx.artifact_path(benchmark),
            repo_root=cfg.repo_root,
            k_values=cfg.k_values,
            engine_hits=tuple((e, ctx.artifact_path(s)) for e, s in query_steps),
            engine_judge=tuple((e, ctx.artifact_path(s)) for e, s in judge_steps),
            engine_index=tuple((e, ctx.artifact_path(s)) for e, s in build_steps),
            output_path=ctx.output_path,
        ),
        deps=(benchmark, *[s for _, s in query_steps], *[s for _, s in judge_steps], *[s for _, s in build_steps]),
    )
    return score


def _parse_args() -> tuple[PipelineConfig, int]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"), help="~/.claude projects dir")
    ap.add_argument("--glob", default="-home-*-code-marin*", help="project-dir glob for the sessions to mine")
    ap.add_argument("--repo", default="/home/power/code/marin", help="repo root to index and resolve gold against")
    ap.add_argument("--limit", type=int, default=0, help="cap session files scanned (debug)")
    ap.add_argument("--max-candidates", type=int, default=600, help="raw navigation candidates fed to the cleaner")
    ap.add_argument("--max-queries", type=int, default=200, help="final benchmark size")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--agent-command", default="claude -p", help="headless agent for query cleaning")
    ap.add_argument("--judge-agent-command", default="claude -p", help="headless agent for judging results")
    ap.add_argument("--bench-batch", type=int, default=12, help="candidates per cleaning call")
    ap.add_argument("--judge-batch", type=int, default=5, help="queries per judging call")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent agent invocations")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per agent invocation")
    ap.add_argument("--engines", default="ripgrep,bm25,dense", help="comma-separated engines to compare")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="fastembed model for the dense engine")
    ap.add_argument("--version", default="dev", help="artifact version tag")
    ap.add_argument("--max-concurrent", type=int, default=3, help="concurrent pipeline steps")
    args = ap.parse_args()
    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    unknown = [e for e in engines if e not in ENGINE_SCRIPTS]
    if unknown:
        ap.error(f"unknown engines {unknown}; known: {sorted(ENGINE_SCRIPTS)}")
    cfg = PipelineConfig(
        projects_dir=args.projects,
        session_glob=args.glob,
        repo_root=args.repo,
        limit=args.limit,
        max_candidates=args.max_candidates,
        max_queries=args.max_queries,
        seed=args.seed,
        agent_command=args.agent_command,
        judge_agent_command=args.judge_agent_command,
        bench_batch=args.bench_batch,
        judge_batch=args.judge_batch,
        agent_concurrency=args.concurrency,
        agent_timeout=args.timeout,
        engines=engines,
        embed_model=args.embed_model,
        k=DEFAULT_K,
        k_values=(1, 3, 5, 10),
        version=args.version,
    )
    return cfg, args.max_concurrent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, max_concurrent = _parse_args()
    StepRunner().run([build(cfg).lower()], max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
