# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5094: opt-in public diagnostic-log sourcing for training."""

import json

import click

from marin.datakit.download.diagnostic_logs import DiagnosticSourceStatus, source_inventory


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


if __name__ == "__main__":
    blocked = [entry.name for entry in source_inventory() if entry.status == DiagnosticSourceStatus.BLOCKED_LICENSE]
    if blocked:
        click.echo(
            "Blocked external sources (license/provenance review required before training ingest): " + ", ".join(blocked)
        )
    cli()
