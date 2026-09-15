# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit hero checkpoint samples, publish reports, or show sampling status."""

import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import click
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep.hero_recipe import HERO_PROCESSES_PER_TASK
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleStore
from experiments.grug.moe_hero_ep.ops.vibe_check.config import (
    CHECKPOINT_RUNS,
    STORE_ROOT,
    TARGET_CLUSTER,
    discover_requests,
    sampling_resources,
    sampling_spec,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import JOB_USER, IrisSamplingJobs, submit_pending
from experiments.grug.moe_hero_ep.ops.vibe_check.publishing import publish_reports, update_issue_comment
from experiments.grug.moe_hero_ep.ops.vibe_check.status import render_sampling_summary

logger = logging.getLogger(__name__)


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("action", type=click.Choice(["reconcile", "report", "inventory", "status"]))
@click.option("--store-root", default=STORE_ROOT, show_default=True)
def main(action: str, store_root: str) -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    store = SampleStore(store_root)
    now = datetime.now(UTC)
    if action == "report":
        # The report day starts at 08:00 UTC.
        report_day = (now - timedelta(hours=8)).date()
        url = publish_reports(store, report_day, partial(update_issue_comment, token=os.environ["GH_TOKEN"]))
        logger.info("Current report: %s", url)
        return
    if action == "status":
        with connect_controller(cluster_name="marin") as endpoint:
            with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
                jobs = client.list_jobs(prefix=f"/{JOB_USER}/")
        summary = render_sampling_summary(store.requests(), store.completed_ids(), jobs, endpoint.url)
        click.echo(summary)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as handle:
                handle.write(summary)
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    requests = discover_requests(CHECKPOINT_RUNS, sampling_spec(), revision, target_cluster=TARGET_CLUSTER)
    if action == "inventory":
        for request in sorted(requests, key=lambda value: value.checkpoint.step):
            logger.info("%s step=%d %s", request.sample_id, request.checkpoint.step, request.checkpoint.uri)
        logger.info("%d permanent checkpoints", len(requests))
        return
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True, stdout=subprocess.DEVNULL)
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            jobs = IrisSamplingJobs(
                client,
                endpoint,
                Path.cwd(),
                store_root,
                sampling_resources(),
                HERO_PROCESSES_PER_TASK,
                sampler_module="experiments.grug.moe_hero_ep.ops.vibe_check.sample",
            )
            submit_pending(store, jobs, requests)


if __name__ == "__main__":
    main()
