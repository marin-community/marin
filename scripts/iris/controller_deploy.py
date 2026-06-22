#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The mechanical spine of the deploy-controller-fix runbook.

Deploying a merged controller/iris fix is three steps: rebuild the image, restart
the controller, confirm the controller is actually running the new code. Two of
those are purely mechanical and easy to get subtly wrong by hand; this tool owns
them:

    build    Trigger the "Ops - Docker Images" workflow on a ref and stream the
             run to completion. Skipping this is the classic trap: a restart
             alone re-pulls whatever ``:latest`` already is and ships nothing.

    verify   Read the controller's running git hash and check it matches a ref.
             The hash baked into the image is a *tree* hash (``HEAD^{tree}``),
             not a commit sha, so a plain commit sha never matches — this command
             resolves the expected tree hash and compares correctly.

The restart in between is deliberately NOT automated here. It is human-gated
(controller-only vs. full restart, ``--skip-checkpoint`` recovery) and the
runbook owns when and how. See .agents/runbooks/deploy-controller-fix.md.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

import click

WORKFLOW = "Ops - Docker Images"
DEFAULT_REPO = "marin-community/marin"
DEFAULT_REF = "origin/main"
DEFAULT_CLUSTER = "marin"

# `iris cluster status` prints one line per controller field; the git hash row is
# `  Git Hash: <hash>` (lib/iris/src/iris/cli/cluster.py).
_GIT_HASH_LINE = re.compile(r"^\s*Git Hash:\s*(\S+)\s*$", re.MULTILINE)

# The controller reports "unknown" when the image was built without IRIS_GIT_HASH
# (lib/iris/src/iris/cluster/process_status.py); we cannot verify against that.
UNKNOWN_HASH = "unknown"


def _run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command and streaming its output. Raises on failure."""
    click.echo(f"$ {' '.join(cmd)}", err=True)
    subprocess.run(cmd, check=True)


def tree_hash(ref: str) -> str:
    """Short hash of ``ref^{tree}``, the git hash baked into the controller image."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", f"{ref}^{{tree}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Cannot resolve tree hash for ref {ref!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def deployed_hash(cluster: str) -> str:
    """The git hash the controller reports it is running, via ``iris cluster status``."""
    result = subprocess.run(
        ["iris", "--cluster", cluster, "cluster", "status"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"`iris cluster status` failed for cluster {cluster!r}: {result.stderr.strip()}")
    match = _GIT_HASH_LINE.search(result.stdout)
    if not match:
        raise click.ClickException(
            "Could not find a 'Git Hash:' line in `iris cluster status` output. "
            "Is the controller reachable? Output was:\n" + result.stdout
        )
    return match.group(1)


def _existing_run_ids(repo: str) -> set[int]:
    # Quiet (no command echo) — this is polled every few seconds during a build.
    result = subprocess.run(
        ["gh", "run", "list", "--workflow", WORKFLOW, "-R", repo, "-L", "30", "--json", "databaseId"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {row["databaseId"] for row in json.loads(result.stdout)}


@click.group()
def cli() -> None:
    """Build and verify the Iris controller image (deploy-controller-fix spine)."""


@cli.command("build")
@click.option("--ref", default="main", show_default=True, help="Git ref to build the image from.")
@click.option("--repo", default=DEFAULT_REPO, show_default=True)
@click.option("--poll-timeout", default=120, show_default=True, help="Seconds to wait for the run to appear.")
@click.option("--dry-run", is_flag=True, help="Print the gh commands without running them.")
def build_cmd(ref: str, repo: str, poll_timeout: int, dry_run: bool) -> None:
    """Trigger the image rebuild and stream it to completion.

    A restart only re-pulls whatever ``:latest`` currently is; without this step
    a merged fix never reaches the cluster.
    """
    if dry_run:
        click.echo(f"$ gh workflow run {WORKFLOW!r} -R {repo} --ref {ref}")
        click.echo(f"$ gh run watch <new-run-id> -R {repo} --exit-status")
        click.echo(f"would build tree hash {tree_hash(ref)} (verify against this after restart)")
        return

    before = _existing_run_ids(repo)
    _run(["gh", "workflow", "run", WORKFLOW, "-R", repo, "--ref", ref])

    # workflow_dispatch runs appear within a few seconds; poll until a new id shows up.
    deadline = time.monotonic() + poll_timeout
    run_id: int | None = None
    while time.monotonic() < deadline:
        time.sleep(3)
        new = _existing_run_ids(repo) - before
        if new:
            run_id = max(new)
            break
    if run_id is None:
        raise click.ClickException(f"No new {WORKFLOW!r} run appeared within {poll_timeout}s; check `gh run list`.")

    click.echo(f"Watching run {run_id} ...")
    # --exit-status makes a failed run a non-zero exit here.
    _run(["gh", "run", "watch", str(run_id), "-R", repo, "--exit-status"])
    click.echo(f"Build succeeded. Image carries tree hash {tree_hash(ref)}.")
    click.echo("Next: restart the controller (human-gated, see the runbook), then `verify`.")


@cli.command("verify")
@click.option(
    "--cluster", default=DEFAULT_CLUSTER, show_default=True, help="Cluster shorthand passed to `iris --cluster`."
)
@click.option("--ref", default=DEFAULT_REF, show_default=True, help="Ref whose code the controller should be running.")
@click.option("--expected", default=None, help="Expected git (tree) hash; overrides --ref.")
def verify_cmd(cluster: str, ref: str, expected: str | None) -> None:
    """Check the controller is running the expected code, not merely back up.

    Exits 0 on match, 1 on mismatch, 2 if the controller reports an unknown hash.
    """
    expected_hash = expected or tree_hash(ref)
    running = deployed_hash(cluster)

    click.echo(f"expected: {expected_hash}" + ("" if expected else f"  ({ref} tree)"))
    click.echo(f"deployed: {running}")

    if running == UNKNOWN_HASH:
        click.echo("MISMATCH: controller reports an unknown git hash (image built without IRIS_GIT_HASH).")
        sys.exit(2)
    if running == expected_hash:
        click.echo("MATCH: controller is running the expected code.")
        return
    click.echo("MISMATCH: controller is running different code than expected.")
    click.echo("The rebuild didn't run, the restart shipped a stale :latest, or the AR pull was cached.")
    sys.exit(1)


if __name__ == "__main__":
    cli()
