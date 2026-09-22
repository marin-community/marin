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
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep.checkpoints import hero_checkpoint_paths
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleStore
from experiments.grug.moe_hero_ep.ops.vibe_check.config import (
    SAMPLING_GPUS_PER_NODE,
    STORE_ROOT,
    TARGET_CLUSTER,
    discover_requests,
    sampling_resources,
    sampling_spec,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import JOB_USER, IrisSamplingJobs, SubmissionMode, submit_pending
from experiments.grug.moe_hero_ep.ops.vibe_check.publishing import publish_reports, update_issue_comment
from experiments.grug.moe_hero_ep.ops.vibe_check.status import render_sampling_summary

logger = logging.getLogger(__name__)

CONTROLLER_CLUSTER = "marin"


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("action", type=click.Choice(["reconcile", "report", "inventory", "status"]))
@click.option("--store-root", default=STORE_ROOT, show_default=True)
@click.option(
    "--submission",
    type=click.Choice([mode.value for mode in SubmissionMode]),
    default=SubmissionMode.NEXT.value,
    show_default=True,
    help="Submit the next request or all unfinished checkpoints in the current discovery set.",
)
@click.option(
    "--priority",
    type=click.Choice([name for name in PRIORITY_BAND_NAMES if name != "system"], case_sensitive=False),
    help="Retain this priority for all discovered checkpoints. Omit to preserve saved priorities (otherwise batch).",
)
def main(action: str, store_root: str, priority: str | None, submission: str) -> None:
    if priority is not None and action != "reconcile":
        raise click.UsageError("--priority applies only to reconcile")
    submission_mode = SubmissionMode(submission)
    if submission_mode != SubmissionMode.NEXT and action != "reconcile":
        raise click.UsageError("--submission applies only to reconcile")
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    store = SampleStore(store_root)
    spec = sampling_spec()
    now = datetime.now(UTC)
    if action == "report":
        # The report day starts at 08:00 UTC.
        report_day = (now - timedelta(hours=8)).date()
        url = publish_reports(store, report_day, partial(update_issue_comment, token=os.environ["GH_TOKEN"]), spec=spec)
        logger.info("Current report: %s", url)
        return
    if action == "status":
        with connect_controller(cluster_name=CONTROLLER_CLUSTER) as endpoint:
            with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
                jobs = client.list_jobs(prefix=f"/{JOB_USER}/")
        summary = render_sampling_summary(store.requests(spec), store.completed_ids(), jobs, endpoint.url)
        click.echo(summary)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as handle:
                handle.write(summary)
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    requests = discover_requests(hero_checkpoint_paths(), spec, revision, target_cluster=TARGET_CLUSTER)
    if action == "inventory":
        for request in sorted(requests, key=lambda value: value.checkpoint.step):
            logger.info("%s step=%d %s", request.sample_id, request.checkpoint.step, request.checkpoint.uri)
        logger.info("%d permanent checkpoints", len(requests))
        return
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True, stdout=subprocess.DEVNULL)
    with connect_controller(cluster_name=CONTROLLER_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            jobs = IrisSamplingJobs(
                client,
                endpoint,
                Path.cwd(),
                store_root,
                sampling_resources(),
                SAMPLING_GPUS_PER_NODE,
                sampler_module="experiments.grug.moe_hero_ep.ops.vibe_check.sample",
            )
            submit_pending(
                store,
                jobs,
                requests,
                spec=spec,
                priority_band=priority_band_value(priority) if priority is not None else None,
                submission=submission_mode,
            )


if __name__ == "__main__":
    main()
