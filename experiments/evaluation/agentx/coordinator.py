# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a fixed grid inside one Iris CPU job; coordinator loss needs manual recovery."""

import dataclasses
import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.types import JobName
from iris.resources.state import is_job_finished
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration, ExponentialBackoff

from experiments.evaluation.agentx.artifacts import read_record, write_record
from experiments.evaluation.agentx.collect import collect_terminal
from experiments.evaluation.agentx.recipe import series_path
from experiments.evaluation.agentx.runtime import runtime_metadata
from experiments.evaluation.agentx.submit import point_manifest, submit_point
from experiments.evaluation.agentx.workload import prepare_workload

logger = logging.getLogger(__name__)
COORDINATOR_TIMEOUT = 7 * 86400


def resolve_plan(plan: dict, workload: dict) -> dict:
    """Resolve the profile's pinned release once for the prepared grid."""
    release = json.loads(json.dumps(runtime_metadata(plan["profile"])))
    return plan | {
        "points": [
            {
                "name": f"c{value}",
                "manifest": point_manifest(plan["profile"], value, plan["source"], workload, release, plan["placement"]),
            }
            for value in plan["concurrencies"]
        ]
    }


def collect_point(client: IrisClient, job_id: JobName, point: dict, plan: dict, work: Path) -> dict:
    """Wait for successful output capture or retain a compact final failure."""
    logger.info("Collecting %s; Iris output finalization may take ten minutes", job_id)
    receipt = None

    def collected() -> bool:
        nonlocal receipt
        status = client.job_status(job_id)
        if not is_job_finished(status.state):
            raise ValueError("Collection requires a terminal child; it never advances execution")
        tasks = tuple(client.task_status(task.task_id) for task in client.list_tasks(job_id))
        receipt = collect_terminal(
            dataclasses.replace(status, tasks=tasks),
            point["manifest"],
            plan["retention_uri"],
            work,
            time.time(),
            client=client,
        )
        return receipt is not None

    ExponentialBackoff(initial=5, maximum=30).wait_until_or_raise(
        collected, Duration.from_seconds(660), error_message=f"Iris outputs for {job_id} did not finalize"
    )
    assert receipt is not None
    return receipt


def publish_results(plan: dict, points: list[dict]) -> str:
    """Publish completed results for the existing GitHub artifact/import workflow."""
    result = {"source": plan["source"], "points": points, "coordinator_job_id": plan["job_id"]}
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    source = plan["source"]
    uri = prefix_join(
        plan["root"], f"completed/{source['run_id']}-{source['run_attempt']}-{plan['profile']}-{digest}.json"
    )
    write_record(uri, result)
    return uri


def run_points(client: IrisClient, plan: dict, work: Path) -> None:
    """Submit children serially. Iris alone retries whole measurement attempts."""
    # Commit exact preparation lineage before any child can exist. Recovery reads
    # this resolved plan even if the CPU coordinator dies during GPU queueing.
    write_record(prefix_join(plan["retention_uri"], "plan.json"), plan)
    completed = []
    for point in plan["points"]:
        logger.info("Submitting %s concurrency=%s; queue ETA unknown", plan["profile"], point["manifest"]["concurrency"])
        job = submit_point(client, point["name"], point["manifest"], plan["priority"])
        job.wait(timeout=COORDINATOR_TIMEOUT, raise_on_failure=False, stream_logs=True)
        receipt = collect_point(client, job.job_id, point, plan, work)
        completed.append(point | {"job_id": str(job.job_id), "receipt": receipt})
        # Retain each point before waiting for the next GPU allocation.
        logger.info("Retained %s: %s", job.job_id, publish_results(plan, completed))
    if any(point["receipt"]["status"] != "valid" for point in completed):
        raise RuntimeError("One or more points failed; completed siblings remain importable")


def recover_results(client: IrisClient, plan: dict, work: Path) -> str:
    """Salvage terminal children after coordinator loss; never submit or retry work."""
    if not is_job_finished(client.job_status(JobName.from_string(plan["job_id"])).state):
        raise ValueError("Coordinator is still live; reattach or explicitly cancel it before recovery")
    try:
        plan = read_record(prefix_join(plan["retention_uri"], "plan.json"))
    except FileNotFoundError:
        # Preparation never reached the durable-before-submit boundary.
        return publish_results(plan, [])
    completed = []
    for point in plan["points"]:
        job_id = JobName.from_string(plan["job_id"]).child(point["name"])
        try:
            receipt = collect_point(client, job_id, point, plan, work)
        except ConnectError as error:
            if error.code != Code.NOT_FOUND:
                raise
            continue  # Coordinator died before submitting this point.
        completed.append(point | {"job_id": str(job_id), "receipt": receipt})
    return publish_results(plan, completed)


def run_coordinator(plan: dict) -> None:
    """Prepare on CPU, then measure the requested grid on the same Iris peer."""
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="agentx-coordinator-") as directory:
        work = Path(directory)
        client = iris_ctx().client
        assert client is not None
        workload = prepare_workload(work / "workload", series_path=series_path(plan["profile"]))
        run_points(client, resolve_plan(plan, workload), work)
