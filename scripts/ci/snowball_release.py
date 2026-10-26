# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish changed, reviewed Snowball tree source through Marina's release CLI."""

import json
import subprocess
from typing import Any

import click

APP_DIR = "infra/marina/applets/snowball_family_tree"


def marina(*arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "marina", *arguments, "--json"], check=True, stdout=subprocess.PIPE, text=True
    )
    return json.loads(result.stdout)


@click.command()
@click.option("--dry-run", is_flag=True, help="Inspect live status and validate a changed package without publishing.")
@click.option(
    "--bootstrap-unrecorded",
    is_flag=True,
    help="Explicitly replace a legacy live package without recorded Git provenance with reviewed main source.",
)
def main(dry_run: bool, bootstrap_unrecorded: bool) -> None:
    """Skip unchanged source; otherwise release once using the observed live version.

    Run from the repository root after syncing the Marina environment. Production
    publication delegates clean-current-main checks and version conflicts to Marina.
    """
    before = marina("release-status", APP_DIR)
    state = before["state"]
    if state == "current":
        click.echo(json.dumps({"action": "unchanged", "status": before}))
        return
    if state == "unrecorded" and not bootstrap_unrecorded:
        raise click.ClickException("Live source is unrecorded; inspect it, then dispatch with bootstrap_unrecorded=true.")
    if state not in {"different", "unrecorded"}:
        raise click.ClickException(f"Refusing to release source in state {state!r}")

    subprocess.run(["node", f"{APP_DIR}/verify_layout.js"], check=True, stdout=subprocess.DEVNULL)
    if dry_run:
        validation = marina("release", APP_DIR, "--dry-run")
        click.echo(json.dumps({"action": "would_release", "before": before, "validation": validation}))
        return

    published = marina("release", APP_DIR, "--base-version", str(before["version"]))
    after = marina("release-status", APP_DIR)
    if after["state"] != "current" or after["version"] != published["version"]:
        raise click.ClickException(f"Published revision is no longer current: {json.dumps(published)}")
    click.echo(json.dumps({"action": "released", "before": before, "release": published, "after": after}))


if __name__ == "__main__":
    main()
