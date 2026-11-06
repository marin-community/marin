# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data Belay command line: `uv run python -m experiments.data_belay <command>`."""

import sys

import click
from levanter.tokenizers import load_tokenizer
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.data_belay.intake import MAX_EPOCHS, DatasetRequest, LicenseVerdict, run_intake
from experiments.data_belay.rungs import ladder_tokens
from experiments.grug.fast_track.launch import V16384_TOKENIZER


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--dataset", "repo", required=True, help="Hugging Face dataset id, for example HuggingFaceFW/fineweb.")
@click.option("--revision", default=None, help="Commit, tag or branch; the check pins it to a commit.")
@click.option("--config", "config_name", default=None, help="Dataset config (subset).")
@click.option("--split", default="train", show_default=True)
@click.option("--text-field", default="text", show_default=True)
@click.option("--share", type=float, default=0.2, show_default=True, help="Share of the rung's tokens.")
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
def check(
    repo: str,
    revision: str | None,
    config_name: str | None,
    split: str,
    text_field: str,
    share: float,
    as_json: bool,
) -> None:
    """Pin the revision, check the license and estimate the dataset's size at each rung."""
    configure_coreweave_s3()
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    request = DatasetRequest(repo, revision, config_name, split, text_field)
    report = run_intake(request, share, ladder_tokens(), tokenizer)
    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(f"dataset: {report.repo}@{report.revision} config={report.config} split={report.split}")
        click.echo(f"license: {', '.join(report.licenses) or 'none on the card'} -> {report.license_verdict}")
        if report.gated:
            click.echo("gated: yes (the terms of access must allow training)")
        click.echo(
            f"size: {report.num_rows:,} rows, about {report.estimated_tokens:,} tokens "
            f"({report.sample.tokens_per_row:,.0f} tokens per row over {report.sample.rows:,} sampled rows)"
        )
        for need in report.needs:
            click.echo(
                f"{need.rung}: {need.new_tokens:,} new tokens of {need.rung_tokens:,} at share {report.share:g} "
                f"= {need.epochs:.3f} epochs"
            )
    if report.license_verdict is not LicenseVerdict.ALLOWED:
        click.echo(f"refusing: license {report.license_verdict}; a person must approve it", err=True)
        sys.exit(2)
    if not report.passed:
        click.echo(f"refusing: more than {MAX_EPOCHS:g} epochs at some rung; lower --share", err=True)
        sys.exit(2)
    click.echo("passed")


if __name__ == "__main__":
    cli()
