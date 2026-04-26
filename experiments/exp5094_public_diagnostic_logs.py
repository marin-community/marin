# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5094: opt-in public diagnostic-log sourcing for training."""

import json

import click
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer
from marin.datakit.download.diagnostic_logs import (
    DEFAULT_SAMPLE_MAX_PARQUET_FILES,
    DEFAULT_SAMPLE_MAX_ROWS,
    DiagnosticSourceStatus,
    extract_starcoder_fixture_logs_step,
    source_inventory,
)


def _inventory_payload() -> list[dict[str, object]]:
    payload = []
    for source in source_inventory():
        payload.append(
            {
                "name": source.name,
                "status": source.status.value,
                "source_url": source.source_url,
                "license": source.source_license,
                "format": source.source_format,
                "compressed_size_bytes": source.compressed_size_bytes,
                "rough_tokens_b": source.rough_tokens_b,
                "risk": source.contamination_risk,
                "provenance_notes": source.provenance_notes,
            }
        )
    return payload


def _extract_step(source_path: str, max_parquet_files: int, max_rows: int) -> ExecutorStep:
    return extract_starcoder_fixture_logs_step(
        source_path=source_path,
        max_parquet_files=max_parquet_files,
        max_rows=max_rows,
    ).as_executor_step()


def _tokenize_step(extracted_step: ExecutorStep) -> ExecutorStep:
    return ExecutorStep(
        name="tokenized/diagnostic_logs/github_fixtures_sample",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[extracted_step.as_input_name() / "train/*.jsonl"],
            validation_paths=[extracted_step.as_input_name() / "dev/*.jsonl"],
            cache_path=this_output_path(),
            tokenizer=versioned(llama3_tokenizer),
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
@click.option("--source_path", required=True, help="Path to already-staged StarCoder parquet shards.")
@click.option("--max_parquet_files", default=DEFAULT_SAMPLE_MAX_PARQUET_FILES, show_default=True, type=int)
@click.option("--max_rows", default=DEFAULT_SAMPLE_MAX_ROWS, show_default=True, type=int)
def extract_cmd(source_path: str, max_parquet_files: int, max_rows: int) -> None:
    """Extract a capped sample of partitioned diagnostic logs from staged source corpora."""
    step = _extract_step(source_path, max_parquet_files, max_rows)
    executor_main(steps=[step], description="Issue #5094 extract public diagnostic logs sample")


@cli.command("tokenize")
@click.option("--source_path", required=True, help="Path to already-staged StarCoder parquet shards.")
@click.option("--max_parquet_files", default=DEFAULT_SAMPLE_MAX_PARQUET_FILES, show_default=True, type=int)
@click.option("--max_rows", default=DEFAULT_SAMPLE_MAX_ROWS, show_default=True, type=int)
def tokenize_cmd(source_path: str, max_parquet_files: int, max_rows: int) -> None:
    """Tokenize the same capped sample (train/dev only)."""
    extract_step = _extract_step(source_path, max_parquet_files, max_rows)
    tokenize_step = _tokenize_step(extract_step)
    executor_main(steps=[extract_step, tokenize_step], description="Issue #5094 tokenize diagnostic logs sample")


@cli.command("all")
@click.option("--source_path", required=True, help="Path to already-staged StarCoder parquet shards.")
@click.option("--max_parquet_files", default=DEFAULT_SAMPLE_MAX_PARQUET_FILES, show_default=True, type=int)
@click.option("--max_rows", default=DEFAULT_SAMPLE_MAX_ROWS, show_default=True, type=int)
def all_cmd(source_path: str, max_parquet_files: int, max_rows: int) -> None:
    """Run sample extraction and tokenization for approved training-ready diagnostic logs."""
    extract_step = _extract_step(source_path, max_parquet_files, max_rows)
    tokenize_step = _tokenize_step(extract_step)
    executor_main(
        steps=[extract_step, tokenize_step],
        description="Issue #5094 public diagnostic logs sample",
    )


if __name__ == "__main__":
    blocked = [entry.name for entry in source_inventory() if entry.status == DiagnosticSourceStatus.BLOCKED_LICENSE]
    if blocked:
        click.echo(
            "Blocked external sources (license/provenance review required before training ingest): " + ", ".join(blocked)
        )
    cli()
