# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        # Parse dataset name and subset
        if ":" in name:
            dataset_family, subset = name.split(":", 1)
        else:
            # Implicit :all for simple datasets
            dataset_family = name
            subset = "all"

        # Look up in registry
        if dataset_family not in DATASETS:
            click.echo(f"Error: Unknown dataset '{dataset_family}'", err=True)
            click.echo(f"Available datasets: {', '.join(sorted(DATASETS.keys()))}", err=True)
            click.echo("Use 'list' command to see all available datasets and subsets", err=True)
            sys.exit(1)

        dataset_info = DATASETS[dataset_family]
        available_subsets = dataset_info["subsets"]

        # Determine which subsets to process
        if subset == "all":
            # Process all available subsets
            subsets_to_process = available_subsets
        elif subset in available_subsets:
            # Process specific subset
            subsets_to_process = [subset]
        else:
            click.echo(f"Error: Unknown {dataset_family} subset '{subset}'", err=True)
            click.echo(f"Available subsets: {', '.join(available_subsets)}", err=True)
            sys.exit(1)

        # Add download step (only once per dataset)
        if download:
            steps.append(dataset_info["download"])

        # Add tokenization steps
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


@click.group()
def cli():
    """Manage pretraining datasets: download, tokenize, and list available datasets."""
    pass


@cli.command()
def list():
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
@click.option("--download", is_flag=True, help="Download raw datasets")
@click.option("--tokenize/--no-tokenize", default=True, help="Tokenize datasets (default: enabled)")
def run(datasets, download, tokenize):
    """
    Process datasets: download and/or tokenize.

    DATASETS: One or more dataset names to process.

    \b
    Examples:
      # Tokenize simple datasets (implicit :all)
      python -m experiments.pretraining_datasets run proofpile_2 slimpajama_6b

      # Download raw datasets
      python -m experiments.pretraining_datasets run --download dolmino nemotron_cc

      # Tokenize specific subsets
      python -m experiments.pretraining_datasets run dolmino:dclm nemotron_cc:hq_actual

      # Tokenize all subsets of a multi-subset dataset
      python -m experiments.pretraining_datasets run dolmino:all

      # Download and tokenize
      python -m experiments.pretraining_datasets run --download dolmino:dclm
    """
    if not download and not tokenize:
        click.echo("Error: Must specify at least one of --download or --tokenize", err=True)
        sys.exit(1)

    steps = get_steps(list(datasets), download=download, tokenize=tokenize)

    if not steps:
        click.echo("Error: No steps found", err=True)
        sys.exit(1)

    action_str = []
    if download:
        action_str.append("download")
    if tokenize:
        action_str.append("tokenize")

    click.echo(f"Running {len(steps)} step(s) to {' and '.join(action_str)}: {', '.join(datasets)}")
    executor_main(steps=steps, description=f"{'/'.join(action_str).title()}: {', '.join(datasets)}")


if __name__ == "__main__":
    cli()
