# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5094: opt-in public diagnostic-log sourcing for training."""

import json

import click
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer
from marin.datakit.download.diagnostic_logs import (
    DEFAULT_GHALOGS_MAX_MEMBERS,
    DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    DEFAULT_LOGHUB_MAX_FILES,
    blocked_sources,
    extract_diagnostic_logs_steps,
    extract_ghalogs_step,
    source_inventory,
)


def _inventory_payload() -> list[dict[str, object]]:
    return [source.to_dict() for source in source_inventory()]


def _ghalogs_extract_step(
    source_path: str,
    max_ghalogs_members: int,
) -> ExecutorStep:
    return extract_ghalogs_step(
        source_path=source_path,
        max_members=max_ghalogs_members,
    ).as_executor_step()


def _extract_steps(
    source_path: str,
    max_ghalogs_members: int,
    max_logchunks_examples: int,
    max_loghub_files: int,
) -> tuple[ExecutorStep, ExecutorStep, ExecutorStep]:
    return tuple(
        step.as_executor_step()
        for step in extract_diagnostic_logs_steps(
            source_path=source_path,
            max_ghalogs_members=max_ghalogs_members,
            max_logchunks_examples=max_logchunks_examples,
            max_loghub_files=max_loghub_files,
        )
    )


def _tokenize_step(ghalogs_step: ExecutorStep) -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/diagnostic_logs/ghalogs_sample",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[ghalogs_step.as_input_name() / "train/*.jsonl"],
            validation_paths=[ghalogs_step.as_input_name() / "dev/*.jsonl"],
            cache_path=this_output_path(),
            tokenizer=versioned(llama3_tokenizer),
            format=TextLmDatasetFormat(text_key="text"),
        ),
    )


@click.group(invoke_without_command=True)
@click.option("--dry_run", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.option("--executor_info_base_path", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.option("--prefix", type=str, hidden=True, help="Passed by test framework; ignored by this CLI.")
@click.pass_context
def cli(
    ctx: click.Context,
    dry_run: str | None,
    executor_info_base_path: str | None,
    prefix: str | None,
) -> None:
    """Public diagnostic-log sourcing workflow."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(inventory_cmd)


@cli.command("inventory")
def inventory_cmd() -> None:
    """Print source inventory and gating status as JSON."""
    click.echo(json.dumps(_inventory_payload(), indent=2, sort_keys=True))


@cli.command("extract")
@click.option("--source_path", required=True, help="Path to staged diagnostic-log source files.")
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
@click.option("--max_logchunks_examples", default=DEFAULT_LOGCHUNKS_MAX_EXAMPLES, show_default=True, type=int)
@click.option("--max_loghub_files", default=DEFAULT_LOGHUB_MAX_FILES, show_default=True, type=int)
def extract_cmd(
    source_path: str,
    max_ghalogs_members: int,
    max_logchunks_examples: int,
    max_loghub_files: int,
) -> None:
    """Extract capped samples of public diagnostic logs."""
    executor_main(
        steps=list(_extract_steps(source_path, max_ghalogs_members, max_logchunks_examples, max_loghub_files)),
        description="Issue #5094 extract public diagnostic log sample",
    )


@cli.command("tokenize")
@click.option("--source_path", required=True, help="Path to staged diagnostic-log source files.")
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
def tokenize_cmd(source_path: str, max_ghalogs_members: int) -> None:
    """Tokenize the same capped GHALogs sample (train/dev only)."""
    ghalogs_step = _ghalogs_extract_step(source_path, max_ghalogs_members)
    tokenize_step = _tokenize_step(ghalogs_step)
    executor_main(steps=[ghalogs_step, tokenize_step], description="Issue #5094 tokenize GHALogs sample")


@cli.command("all")
@click.option("--source_path", required=True, help="Path to staged diagnostic-log source files.")
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
@click.option("--max_logchunks_examples", default=DEFAULT_LOGCHUNKS_MAX_EXAMPLES, show_default=True, type=int)
@click.option("--max_loghub_files", default=DEFAULT_LOGHUB_MAX_FILES, show_default=True, type=int)
def all_cmd(
    source_path: str,
    max_ghalogs_members: int,
    max_logchunks_examples: int,
    max_loghub_files: int,
) -> None:
    """Run sample extraction and tokenization for public diagnostic logs."""
    ghalogs_step, logchunks_step, loghub_step = _extract_steps(
        source_path,
        max_ghalogs_members,
        max_logchunks_examples,
        max_loghub_files,
    )
    tokenize_step = _tokenize_step(ghalogs_step)
    executor_main(
        steps=[ghalogs_step, logchunks_step, loghub_step, tokenize_step],
        description="Issue #5094 public diagnostic logs sample",
    )


if __name__ == "__main__":
    blocked = [entry.source_label for entry in blocked_sources()]
    if blocked:
        click.echo(
            "Blocked external sources (license/provenance review required before training ingest): " + ", ".join(blocked)
        )
    cli()
