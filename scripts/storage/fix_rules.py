#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull the latest rules JSON from GCS and fix patterns corrupted by bad consolidation.

Bad patterns end with ``-/%``: the consolidation logic kept a literal ``-/``
before the wildcard, so ``checkpoints/adamh-lr2.83e-/%`` never matches
``checkpoints/adamh-lr2.83e-04/…``.  The fix strips the trailing ``-/%`` and
replaces it with ``%``.

Usage:
    uv run scripts/storage/fix_rules.py [--gcs-prefix GCS_PREFIX] [--dry-run]
"""

from __future__ import annotations

import json
import logging
import subprocess

import click

from scripts.storage.db import DEFAULT_CATALOG, StorageCatalog

log = logging.getLogger(__name__)


def _latest_gcs_file(gcs_prefix: str, stem: str) -> str | None:
    """Return the URI of the most recent timestamped archive file for *stem*.

    Lists ``{gcs_prefix}/archive/{stem}_*.json`` and returns the lexicographically
    last entry (timestamps are ISO-formatted so lexicographic order == time order).
    """
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{gcs_prefix}/archive/{stem}_*.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    uris = sorted(result.stdout.strip().splitlines())
    return uris[-1]


def _fix_pattern(pattern: str) -> str:
    """Replace a broken ``-/%``-terminated pattern with a correct ``%`` wildcard.

    ``checkpoints/adamh-lr2.83e-/%``  →  ``checkpoints/adamh-lr2.83e-%``
    """
    if pattern.endswith("-/%"):
        return pattern[: -len("-/%")] + "%"
    return pattern


def fix_rules(
    gcs_prefix: str,
    catalog: StorageCatalog = DEFAULT_CATALOG,
    dry_run: bool = False,
) -> None:
    """Pull the latest rules from GCS, fix broken patterns, and write back."""
    for stem, local_path in [
        ("protect_rules", catalog.protect_rules_json),
        ("delete_rules", catalog.delete_rules_json),
    ]:
        latest_uri = _latest_gcs_file(gcs_prefix, stem)
        if latest_uri:
            click.echo(f"Pulling {stem} from {latest_uri}")
            subprocess.run(
                ["gcloud", "storage", "cp", latest_uri, str(local_path)],
                check=True,
            )
        elif local_path.exists():
            click.echo(f"No GCS archive found for {stem}; using local {local_path}")
        else:
            click.echo(f"No rules found for {stem}, skipping")
            continue

        rules: list[dict] = json.loads(local_path.read_text())
        fixed = 0
        for rule in rules:
            old = rule.get("pattern", "")
            new = _fix_pattern(old)
            if new != old:
                click.echo(f"  fix: {old!r}  →  {new!r}")
                rule["pattern"] = new
                fixed += 1

        click.echo(f"  {fixed} patterns fixed in {stem}")
        if not dry_run:
            local_path.write_text(json.dumps(rules, indent=2) + "\n")
            click.echo(f"  written to {local_path}")
        else:
            click.echo("  (dry-run: not written)")


@click.command()
@click.option(
    "--gcs-prefix",
    envvar="GCS_DATA_PREFIX",
    required=True,
    help="GCS prefix where rules are stored (e.g. gs://marin-tmp-us-central2/ttl=30d/delete-o-tron).",
)
@click.option("--dry-run", is_flag=True, help="Print fixes without writing files.")
def main(gcs_prefix: str, dry_run: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    fix_rules(gcs_prefix, dry_run=dry_run)


if __name__ == "__main__":
    main()
