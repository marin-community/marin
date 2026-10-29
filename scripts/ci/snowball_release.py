# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish changed, reviewed Snowball tree source through Marina's release CLI."""

import json
import subprocess
from pathlib import Path
from typing import Any

import click

APP_DIR = "infra/marina/apps/snowball_family_tree"


def marina(*arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "marina", *arguments, "--json"], check=True, stdout=subprocess.PIPE, text=True
    )
    return json.loads(result.stdout)


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record) + "\n")


@click.command()
@click.option(
    "--record", type=click.Path(path_type=Path, dir_okay=False), required=True, help="JSON release record path."
)
@click.option("--dry-run", is_flag=True, help="Inspect live status and validate a changed package without publishing.")
@click.option(
    "--bootstrap-unrecorded",
    is_flag=True,
    help="Explicitly replace a legacy live package without recorded Git provenance with reviewed main source.",
)
def main(record: Path, dry_run: bool, bootstrap_unrecorded: bool) -> None:
    """Skip unchanged source; otherwise release once using the observed live version.

    Run from the repository root after syncing the Marina environment. Production
    publication delegates clean-current-main checks and version conflicts to Marina.
    """
    before = marina("release-status", APP_DIR)
    # Establish a writable record before any publication can occur.
    write_record(record, {"action": "inspected", "before": before})
    state = before["state"]
    if state == "current":
        result = {"action": "unchanged", "status": before}
        write_record(record, result)
        click.echo(json.dumps(result))
        return
    if state == "unrecorded" and not bootstrap_unrecorded:
        raise click.ClickException(
            "Live source is unrecorded; inspect it, then dispatch with bootstrap_unrecorded=true."
        )
    if state not in {"different", "unrecorded"}:
        raise click.ClickException(f"Refusing to release source in state {state!r}")

    subprocess.run(["node", f"{APP_DIR}/verify_layout.js"], check=True, stdout=subprocess.DEVNULL)
    if dry_run:
        validation = marina("release", APP_DIR, "--dry-run")
        result = {"action": "would_release", "before": before, "validation": validation}
        write_record(record, result)
        click.echo(json.dumps(result))
        return

    published = marina("release", APP_DIR, "--base-version", str(before["version"]))
    result = {"action": "released", "before": before, "release": published, "verification": "pending"}
    # Save the successful mutation before making another fallible request.
    write_record(record, result)
    try:
        after = marina("release-status", APP_DIR)
    except subprocess.CalledProcessError:
        result["verification"] = "failed"
        write_record(record, result)
        raise
    result["after"] = after
    if after["state"] != "current" or after["version"] != published["version"]:
        result["verification"] = "failed"
        write_record(record, result)
        raise click.ClickException(f"Published revision is no longer current: {json.dumps(published)}")
    result["verification"] = "passed"
    write_record(record, result)
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
