#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a reproducible OpenReward task manifest outside RL training."""

import logging

import click

from rigging.log_setup import configure_logging
from marin.rl.openreward import prepare_openreward_task_manifest, save_openreward_task_manifest

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--environment", "environment_name", required=True, help="OpenReward environment name, e.g. user/env")
@click.option("--split", required=True, help="Environment split to snapshot")
@click.option("--output", required=True, help="Output JSON path for the task manifest")
@click.option("--variant", default=None, help="Optional environment variant")
@click.option("--base-url", default=None, help="Override the OpenReward API base URL")
@click.option(
    "--api-key",
    envvar="OPENREWARD_API_KEY",
    default=None,
    help="OpenReward API key. Defaults to OPENREWARD_API_KEY when set.",
)
@click.option("--start", type=int, default=None, help="Inclusive task index start using Python slice semantics")
@click.option("--stop", type=int, default=None, help="Exclusive task index stop using Python slice semantics")
def main(
    environment_name: str,
    split: str,
    output: str,
    variant: str | None,
    base_url: str | None,
    api_key: str | None,
    start: int | None,
    stop: int | None,
) -> None:
    """Snapshot prompts and tools for one OpenReward split."""

    manifest = prepare_openreward_task_manifest(
        environment_name,
        split,
        base_url=base_url,
        api_key=api_key,
        variant=variant,
        start=start,
        stop=stop,
    )
    save_openreward_task_manifest(manifest, output)
    logger.info(
        "Wrote %d OpenReward tasks for %s split %s to %s",
        manifest.task_count,
        manifest.deployment_name,
        split,
        output,
    )


if __name__ == "__main__":
    main()
