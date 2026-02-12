# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for pretraining datasets."""

import sys

import click
from marin.execution.executor import ExecutorStep, executor_main

from experiments.pretraining_datasets import DATASETS


def get_steps(dataset_names: list[str], *, download: bool = False, tokenize: bool = True) -> list[ExecutorStep]:
    """
    Get steps for the specified dataset names.

    Args:
        dataset_names: List of dataset names. Can be:
            - Simple dataset names: "dclm_baseline", "slimpajama_6b" (implicit :all)
            - Multi-subset with subset name: "dolmino:dclm", "nemotron_cc:hq_actual"
            - Multi-subset all: "dolmino:all", "nemotron_cc:all"
        download: Whether to get download steps
        tokenize: Whether to get tokenization steps

    Returns:
        List of ExecutorSteps
    """
    steps = []

    for name in dataset_names:
        if ":" in name:
            dataset_family, subset = name.split(":", 1)
        else:
            dataset_family = name
            subset = "all"

        if dataset_family not in DATASETS:
            click.echo(f"Error: Unknown dataset '{dataset_family}'", err=True)
            click.echo(f"Available datasets: {', '.join(sorted(DATASETS.keys()))}", err=True)
            click.echo("Use 'list' command to see all available datasets and subsets", err=True)
            sys.exit(1)

        dataset_info = DATASETS[dataset_family]
        available_subsets = dataset_info["subsets"]

        if subset == "all":
            subsets_to_process = available_subsets
        elif subset in available_subsets:
            subsets_to_process = [subset]
        else:
            click.echo(f"Error: Unknown {dataset_family} subset '{subset}'", err=True)
            click.echo(f"Available subsets: {', '.join(available_subsets)}", err=True)
            sys.exit(1)

        if download:
            steps.append(dataset_info["download"])

        if tokenize:
            tokenize_result = dataset_info["tokenize_fn"]()
            for sub in subsets_to_process:
                key = f"{dataset_family}/{sub}"
                if key in tokenize_result:
                    steps.append(tokenize_result[key])
                else:
                    click.echo(f"Error: Tokenization step not found for {key}", err=True)
                    sys.exit(1)

    return steps


@click.group(invoke_without_command=True)
@click.option("--dry_run", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.option("--executor_info_base_path", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.option("--prefix", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.pass_context
def cli(ctx, dry_run, executor_info_base_path, prefix):
    """Manage pretraining datasets: download, tokenize, and list available datasets."""
    # These arguments are passed by the test framework but used by executor_main, not this CLI
    if ctx.invoked_subcommand is None:
        ctx.invoke(_list)


@cli.command(name="list")
def _list():
    """List all available datasets and their subsets."""
    click.echo("=" * 70)
    click.echo("SIMPLE DATASETS (single subset)")
    click.echo("=" * 70)

    simple_datasets = {k: v for k, v in DATASETS.items() if v["subsets"] == ["all"]}
    for name in sorted(simple_datasets.keys()):
        click.echo(f"  {name}")
        click.echo(f"    Usage: {name} or {name}:all")

    multi_datasets = {k: v for k, v in DATASETS.items() if v["subsets"] != ["all"]}
    for family in sorted(multi_datasets.keys()):
        info = multi_datasets[family]
        click.echo(f"\n{'=' * 70}")
        click.echo(f"{family.upper()} SUBSETS")
        click.echo("=" * 70)
        click.echo(f"  Usage: {family}:SUBSET or {family}:all")
        click.echo()
        for subset in sorted(info["subsets"]):
            click.echo(f"    {subset}")


@cli.command()
@click.argument("datasets", nargs=-1, required=True)
def download(datasets):
    """
    Download raw datasets.

    DATASETS: One or more dataset names to download.

    Examples:
      # Download raw datasets
      uv run experiments/pretraining_datasets/main.py download dolmino nemotron_cc

      # Download a specific subset
      uv run experiments/pretraining_datasets/main.py download dolmino:dclm
    """
    steps = get_steps(list(datasets), download=True, tokenize=False)

    if not steps:
        click.echo("Error: No steps found", err=True)
        sys.exit(1)

    click.echo(f"Running {len(steps)} step(s) to download: {', '.join(datasets)}")
    executor_main(steps=steps, description=f"Download: {', '.join(datasets)}")


@cli.command()
@click.argument("datasets", nargs=-1, required=True)
def tokenize(datasets):
    """
    Tokenize datasets.

    DATASETS: One or more dataset names to tokenize.

    Examples:
      # Tokenize simple datasets (implicit :all)
      uv run experiments/pretraining_datasets/main.py tokenize proofpile_2 slimpajama_6b

      # Tokenize specific subsets
      uv run experiments/pretraining_datasets/main.py tokenize dolmino:dclm nemotron_cc:hq_actual

      # Tokenize all subsets of a multi-subset dataset
      uv run experiments/pretraining_datasets/main.py tokenize dolmino:all
    """
    steps = get_steps(list(datasets), download=False, tokenize=True)

    if not steps:
        click.echo("Error: No steps found", err=True)
        sys.exit(1)

    click.echo(f"Running {len(steps)} step(s) to tokenize: {', '.join(datasets)}")
    executor_main(steps=steps, description=f"Tokenize: {', '.join(datasets)}")


if __name__ == "__main__":
    cli()
