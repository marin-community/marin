# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One entry point for the grug-moe tokenizer FLOP-equivalent bake-off.

The bake-off asks how much real, compute-fair uplift grug-moe gets from an alternative tokenizer
family, measured in FLOP-equivalent bits-per-byte (feBPB) so a denser (cheaper-to-serve) tokenizer
is credited for reinvesting its serving saving into training. The full pipeline is a DAG of
:class:`~marin.execution.lazy.ArtifactStep` stages, one module each:

    corpus            build the raw multi-domain tokenizer-training corpus   (corpus.py)
      -> train_tok    train + push each SuperBPE/BPE tokenizer               (train_tokenizer.py)
      -> fertility    measure tokens/byte per arm on the soak corpus         (fertility.py)
    hf_download       mirror each soak data source to S3                     (train_model.py)
      -> tokenize     tokenize each source with each arm's tokenizer         (train_model.py)
      -> train_model  one 24h grug-moe soak per arm, logging BPB to W&B      (train_model.py)
    collect_results   pull the BPB ladder + loss curves from W&B            (collect_results.py)
      + fertility  -> analysis   score feBPB + raw BPB                       (analysis.py)

This module holds the arm roster and drives the pieces that run from a laptop/CPU box:

* ``prep``  — build the corpus, train the fixed-soak tokenizers, and measure fertility (StepRunner).
* ``soak``  — print the per-arm ``iris job run`` commands that launch the 24h GPU soaks (each is its
  own long job, launched and monitored individually — this only prints the plan).
* ``score`` — pull results from W&B and compute the feBPB analysis (StepRunner), then print it.

Findings and the corrected result: ``.agents/projects/20260703_tokenizer_bakeoff_findings.md``;
per-experiment reproduce commands: ``EXPERIMENT_LOG.md``.
"""

from __future__ import annotations

import click
from marin.execution.lazy import lower, resolve
from marin.execution.step_runner import StepRunner

from experiments.tokenization.analysis import bakeoff_report_step, print_report
from experiments.tokenization.collect_results import GROUP, soak_results_step
from experiments.tokenization.corpus import tokenizer_training_corpus_raw
from experiments.tokenization.fertility import fertility_over_corpus_step
from experiments.tokenization.train_model import SoakParams
from experiments.tokenization.train_tokenizer import FIXED_SOAK_SPECS, trained_tokenizer

# The eight arms the corrected soak scored. The two off-the-shelf arms (the Llama-3 baseline and the
# UW SuperBPE control) need no training; the six ``-fixed`` arms are trained by ``prep`` from
# FIXED_SOAK_SPECS. Order is baseline-first for the fertility/analysis tables.
SOAK_ARM_NAMES: tuple[str, ...] = (
    "marin-128k",  # Llama-3 vocab incumbent (off-the-shelf)
    "soak-superbpe-64k-fixed",
    "soak-superbpe-128k-fixed",
    "soak-superbpe-64k-digits-fixed",
    "soak-superbpe-128k-digits-fixed",
    "soak-superbpe-64k-llama-fixed",
    "soak-superbpe-128k-llama-fixed",
    "superbpe-128k",  # off-the-shelf UW SuperBPE control
)

# The default soak shape (see train_model.SoakParams / EXPERIMENT_LOG EXP-011).
SOAK_PARAMS = SoakParams()
CLUSTER = "cw-rno2a"


def soak_launch_command(arm_name: str, params: SoakParams = SOAK_PARAMS, *, run_id: str | None = None) -> str:
    """The ``iris job run`` command that launches one arm's 24h soak (the in-cluster entrypoint)."""
    run_id = run_id or arm_name
    env = {
        "BAKEOFF_ARM": arm_name,
        "SCALE_GPU_REPLICAS": params.replicas,
        "SCALE_EXPERT_AXIS": params.expert_axis,
        "SCALE_HIDDEN_DIM": 2560,
        "SCALE_NUM_LAYERS": 8,
        "SCALE_NUM_EXPERTS": 128,
        "SCALE_TOP_K": 4,
        "SCALE_BATCH": params.batch_size,
        "SCALE_SEQ_LEN": 4096,
        "SCALE_STEPS": params.steps,
        "SCALE_TRACKER": "wandb",
        "SCALE_RAM": params.ram,
        "RUN_ID": run_id,
    }
    env_flags = " ".join(f"-e {k} {v}" for k, v in env.items())
    return (
        f"uv run iris --cluster={CLUSTER} job run --no-wait --cpu 2 --memory 3GB --extra cpu "
        f"--job-name {run_id} {env_flags} -- python -m experiments.tokenization.train_model"
    )


@click.group()
def cli() -> None:
    """Drive the tokenizer bake-off pipeline."""


@cli.command()
@click.option("--run", "do_run", is_flag=True, help="Build the steps (default: print the plan).")
@click.option("--local", is_flag=True, help="Train tokenizers inline instead of dispatching per-arm cluster jobs.")
@click.option("--version", default="dev", help="Artifact version for the prep steps.")
def prep(do_run: bool, local: bool, version: str) -> None:
    """Build the corpus, train the fixed-soak tokenizers, and measure fertility (CPU steps)."""
    corpus = tokenizer_training_corpus_raw()
    # --local trains inline; otherwise each tokenizer uses trained_tokenizer's default cluster resources.
    tok_kwargs = {"resources": None} if local else {}
    tokenizers = [trained_tokenizer(spec, corpus, version=version, **tok_kwargs) for spec in FIXED_SOAK_SPECS]
    fertility = fertility_over_corpus_step(SOAK_ARM_NAMES, corpus, version=version)
    if not do_run:
        click.echo("prep DAG:")
        click.echo(f"  corpus:      {lower(corpus).name}")
        for spec, step in zip(FIXED_SOAK_SPECS, tokenizers, strict=True):
            click.echo(f"  tokenizer:   {spec.name} -> {lower(step).name}")
        click.echo(f"  fertility:   {lower(fertility).name}  over {len(SOAK_ARM_NAMES)} arms")
        click.echo("\nre-run with --run to build.")
        return
    # Tokenizers must be pushed before fertility loads them by their trained/<name> ref.
    StepRunner().run([corpus.lower(), *(t.lower() for t in tokenizers)])
    StepRunner().run([fertility.lower()])


@cli.command()
def soak() -> None:
    """Print the per-arm ``iris job run`` commands for the 24h GPU soaks (launch/monitor individually)."""
    for name in SOAK_ARM_NAMES:
        note = "  # off-the-shelf tokenizer (no prep training)" if name in ("marin-128k", "superbpe-128k") else ""
        click.echo(soak_launch_command(name) + note)


@cli.command()
@click.option("--run", "do_run", is_flag=True, help="Run the scoring steps (default: print the plan).")
@click.option("--group", default=GROUP, help="W&B group to pull the soak runs from.")
@click.option("--serving-ratio", default=1.0, help="Lifetime serving/training weight for feBPB.")
@click.option("--ref-budget", default=6e19, help="Reference training-FLOP budget C_ref for feBPB.")
@click.option("--version", default="dev", help="Artifact version for the scoring steps.")
def score(do_run: bool, group: str, serving_ratio: float, ref_budget: float, version: str) -> None:
    """Pull the W&B ladder + fertility and compute the feBPB analysis."""
    ladder = soak_results_step(group=group, version=version)
    fertility = fertility_over_corpus_step(SOAK_ARM_NAMES, tokenizer_training_corpus_raw(), version=version)
    report = bakeoff_report_step(ladder, fertility, serving_ratio=serving_ratio, ref_budget=ref_budget, version=version)
    if not do_run:
        click.echo("score DAG:")
        click.echo(f"  ladder:    {lower(ladder).name}  (W&B group {group})")
        click.echo(f"  fertility: {lower(fertility).name}")
        click.echo(f"  report:    {lower(report).name}  (serving_ratio={serving_ratio}, ref_budget={ref_budget:.0e})")
        click.echo("\nre-run with --run to score.")
        return
    report_artifact = resolve(report)
    print("\n### feBPB (serving saving reinvested into training) ###")
    print_report(report_artifact.febpb)
    print("\n### raw BPB (serving_ratio=0) ###")
    print_report(report_artifact.raw_bpb)


if __name__ == "__main__":
    cli()
