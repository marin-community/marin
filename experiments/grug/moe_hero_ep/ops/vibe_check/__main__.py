# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile permanent hero checkpoints or publish their daily completion history."""

import argparse
import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.moe_hero_ep.hero_recipe import HERO_PROCESSES_PER_TASK
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleStore, reconcile
from experiments.grug.moe_hero_ep.ops.vibe_check.config import (
    STORE_ROOT,
    discover_requests,
    production_run,
    sampling_resources,
    sampling_spec,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import IrisSamplingJobs
from experiments.grug.moe_hero_ep.ops.vibe_check.publishing import publish_daily, update_issue_comment

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reconcile", "report", "inventory"])
    parser.add_argument("--store-root", default=STORE_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    store = SampleStore(args.store_root)
    now = datetime.now(UTC)
    if args.action == "report":
        # The report day starts at 08:00 UTC.
        report_day = (now - timedelta(hours=8)).date()
        url = publish_daily(store, report_day, partial(update_issue_comment, token=os.environ["GH_TOKEN"]))
        logger.info("Daily report: %s", url)
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    requests = discover_requests(production_run(), sampling_spec(), revision)
    if args.action == "inventory":
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
                args.store_root,
                sampling_resources(),
                HERO_PROCESSES_PER_TASK,
                sampler_module="experiments.grug.moe_hero_ep.ops.vibe_check.sample",
            )
            queue = reconcile(store, jobs, requests, now)
    logger.info("Checkpoint history contains %d sample sets", len(queue.entries))


if __name__ == "__main__":
    main()
