# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5094: opt-in public diagnostic-log sourcing for training."""

import json
from dataclasses import dataclass

import click
from levanter.data.text import TextLmDatasetFormat
from marin.datakit.download.diagnostic_logs import (
    DEFAULT_GHALOGS_MATERIALIZE_SHARDS,
    DEFAULT_GHALOGS_MAX_MEMBERS,
    DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    DEFAULT_LOGHUB_MAX_FILES,
    GHALOGS_STAGED_PREFIX,
    blocked_sources,
    extract_ghalogs,
    extract_logchunks_step,
    extract_loghub_step,
    materialize_ghalogs_step,
    source_inventory,
    source_manifest,
)
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    VersionedValue,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer

_GHALOGS_STAGED_SOURCE = InputName.hardcoded(GHALOGS_STAGED_PREFIX)


@dataclass(frozen=True)
class _GhalogsExtractConfig:
    source_path: str | InputName
    output_path: str
    max_members: int | VersionedValue[int]
    source_content_fingerprint: str | VersionedValue[str]
    split_policy: str | VersionedValue[str]
    sanitization_rules: str | VersionedValue[str]


def _inventory_payload() -> list[dict[str, object]]:
    return [source.to_dict() for source in source_inventory()]


def _run_ghalogs_extract(cfg: _GhalogsExtractConfig) -> object:
    assert isinstance(
        cfg.source_path, str
    ), f"Executor should resolve source_path before runtime, got {type(cfg.source_path)}"
    assert isinstance(
        cfg.max_members, int
    ), f"Executor should resolve max_members before runtime, got {type(cfg.max_members)}"
    return extract_ghalogs(cfg.source_path, cfg.output_path, max_members=cfg.max_members)


def _ghalogs_extract_step(
    source_path: str | InputName | None,
    max_ghalogs_members: int,
) -> ExecutorStep:
    source = source_manifest("ghalogs")
    resolved_source_path = _GHALOGS_STAGED_SOURCE if source_path is None else source_path
    return ExecutorStep(
        name="processed/diagnostic_logs/ghalogs_public_sample",
        fn=_run_ghalogs_extract,
        config=_GhalogsExtractConfig(
            source_path=resolved_source_path,
            output_path=this_output_path(),
            max_members=versioned(max_ghalogs_members),
            source_content_fingerprint=versioned(source.fingerprint()),
            split_policy=versioned("97% train / 1% dev / 1% test / 1% issue_5093_holdout"),
            sanitization_rules=versioned("gh token/aws key/secret kv/email/user path/internal gs path"),
        ),
    )


def _extract_steps(
    source_path: str | None,
    max_ghalogs_members: int,
    max_logchunks_examples: int,
    max_loghub_files: int,
) -> tuple[ExecutorStep, ExecutorStep, ExecutorStep]:
    return (
        _ghalogs_extract_step(source_path, max_ghalogs_members),
        extract_logchunks_step(max_examples=max_logchunks_examples).as_executor_step(),
        extract_loghub_step(max_files=max_loghub_files).as_executor_step(),
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


def _run_executor(steps: list[ExecutorStep], description: str) -> None:
    executor_main.__wrapped__(ExecutorMainConfig(), steps=steps, description=description)


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
@click.option(
    "--source_path",
    default=None,
    help="Path to the staged GHALogs source prefix. Defaults to the standard hardcoded staged artifact.",
)
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
@click.option("--max_logchunks_examples", default=DEFAULT_LOGCHUNKS_MAX_EXAMPLES, show_default=True, type=int)
@click.option("--max_loghub_files", default=DEFAULT_LOGHUB_MAX_FILES, show_default=True, type=int)
def extract_cmd(
    source_path: str | None,
    max_ghalogs_members: int,
    max_logchunks_examples: int,
    max_loghub_files: int,
) -> None:
    """Extract capped samples of public diagnostic logs."""
    _run_executor(
        steps=list(_extract_steps(source_path, max_ghalogs_members, max_logchunks_examples, max_loghub_files)),
        description="Issue #5094 extract public diagnostic log sample",
    )


@cli.command("tokenize")
@click.option(
    "--source_path",
    default=None,
    help="Path to the staged GHALogs source prefix. Defaults to the standard hardcoded staged artifact.",
)
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
def tokenize_cmd(source_path: str | None, max_ghalogs_members: int) -> None:
    """Tokenize the same capped GHALogs sample (train/dev only)."""
    ghalogs_step = _ghalogs_extract_step(source_path, max_ghalogs_members)
    tokenize_step = _tokenize_step(ghalogs_step)
    _run_executor(steps=[ghalogs_step, tokenize_step], description="Issue #5094 tokenize GHALogs sample")


@cli.command("all")
@click.option(
    "--source_path",
    default=None,
    help="Path to the staged GHALogs source prefix. Defaults to the standard hardcoded staged artifact.",
)
@click.option("--max_ghalogs_members", default=DEFAULT_GHALOGS_MAX_MEMBERS, show_default=True, type=int)
@click.option("--max_logchunks_examples", default=DEFAULT_LOGCHUNKS_MAX_EXAMPLES, show_default=True, type=int)
@click.option("--max_loghub_files", default=DEFAULT_LOGHUB_MAX_FILES, show_default=True, type=int)
def all_cmd(
    source_path: str | None,
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
    _run_executor(
        steps=[ghalogs_step, logchunks_step, loghub_step, tokenize_step],
        description="Issue #5094 public diagnostic logs sample",
    )


@cli.command("materialize-ghalogs")
@click.option(
    "--source_path",
    default=GHALOGS_STAGED_PREFIX,
    show_default=True,
    help="Path to the staged GHALogs source prefix.",
)
@click.option(
    "--max_members",
    default=None,
    type=int,
    help="Optional cap on GHALogs members during materialization. Omit for the full archive.",
)
@click.option("--num_shards", default=DEFAULT_GHALOGS_MATERIALIZE_SHARDS, show_default=True, type=int)
def materialize_ghalogs_cmd(
    source_path: str,
    max_members: int | None,
    num_shards: int,
) -> None:
    """Materialize sanitized GHALogs parquet shards for datakit/pretraining use."""
    _run_executor(
        steps=[
            materialize_ghalogs_step(
                source_path=source_path,
                max_members=max_members,
                num_shards=num_shards,
            ).as_executor_step()
        ],
        description="Issue #5094 materialize GHALogs parquet",
    )


if __name__ == "__main__":
    blocked = [entry.source_label for entry in blocked_sources()]
    if blocked:
        click.echo(
            "Blocked external sources (license/provenance review required before training ingest): " + ", ".join(blocked)
        )
    cli()
