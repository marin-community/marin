# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare on CPU, run one Iris GPU trial and preserve its recovery identity."""

import hashlib
import json
import logging
import tempfile
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, JobName, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from iris.rpc.proto_display import priority_band_value
from marin.testing.inference.goldens import LookupStatus, lookup_reference, read_fixture
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.evaluation.knotmeter.artifacts import write_record
from experiments.evaluation.knotmeter.collect import collect_trial, record_preparation_failure
from experiments.evaluation.knotmeter.config import Profile
from experiments.evaluation.knotmeter.recipe import PointManifest, TrialPlan
from experiments.evaluation.knotmeter.report import report_campaign
from experiments.evaluation.knotmeter.run import run
from experiments.evaluation.knotmeter.workload import prepare_workload

logger = logging.getLogger(__name__)


def submit_point(client: IrisClient, plan: TrialPlan):
    profile = Profile.model_validate(plan["manifest"]["series"]["profile"])
    series = plan["manifest"]["series"]
    return client.submit(
        entrypoint=Entrypoint.from_callable(run, plan),
        name="serve",
        resources=ResourceSpec(
            cpu=profile.cpu,
            memory=profile.memory,
            disk=profile.disk,
            device=gpu_device(profile.gpus.type, profile.gpus.per_host),
        ),
        environment=EnvironmentSpec(
            sync_packages=["marin-core"], extras=["cpu"], env_vars={"MARIN_PREFIX": plan["marin_prefix"]}
        ),
        constraints=[],
        replicas=profile.gpus.count // profile.gpus.per_host,
        coscheduling=(
            CoschedulingConfig(group_by="nvlink.domain") if profile.gpus.count > profile.gpus.per_host else None
        ),
        priority_band=priority_band_value(profile.priority),
        scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
        timeout=Duration.from_seconds(series["task_timeout"]),
        max_retries_failure=series["max_retries_failure"],
        max_task_failures=series["max_retries_failure"],
        max_retries_preemption=series["max_retries_preemption"],
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=series["task_image"],
    )


def prepare_plan(plan: TrialPlan, work: Path) -> TrialPlan:
    """Freeze the native workload before any GPU child can be submitted."""
    series = plan["series"]
    for check in Profile.model_validate(series["profile"]).checks:
        if check.fixture:
            reference = lookup_reference(read_fixture(check.fixture))
            if reference.status != LookupStatus.FOUND and not series.get("measure_failed_checks"):
                raise ValueError(f"Required reference is {reference.status}: {reference.reason}")
    path = work / "series.json"
    path.write_text(json.dumps(series))
    workload = prepare_workload(work / "preparation", series_path=path)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"series": series, "workload_key": workload["key"], "runtime": plan["vllm_release"]}, sort_keys=True
        ).encode()
    ).hexdigest()
    manifest: PointManifest = {
        "profile": "h100" if series["profile"]["gpus"]["type"] == "H100" else "gb200",
        "concurrency": series["concurrency"],
        "series": series,
        "recipe_fingerprint": fingerprint,
        "vllm_release": plan["vllm_release"],
        "workload": workload,
        "source": plan["source"],
        "placement": plan["placement"],
        "diagnostic": bool(series.get("diagnostic")),
    }
    return {**plan, "manifest": manifest}


def run_coordinator(plan: TrialPlan) -> None:
    logging.basicConfig(level=logging.INFO)
    client = iris_ctx().client
    assert client is not None
    with tempfile.TemporaryDirectory(prefix="knotmeter-trial-") as directory:
        work = Path(directory)
        job = None
        try:
            plan = prepare_plan(plan, work)
            write_record(prefix_join(plan["trial_uri"], "plan.json"), plan)
            job = submit_point(client, plan)
            logger.info("Serving job %s; recovery plan %s/plan.json", job.job_id, plan["trial_uri"])
            job.wait(
                timeout=plan["series"]["scheduling_timeout"] + plan["series"]["task_timeout"] * 3,
                raise_on_failure=False,
                stream_logs=True,
            )
            collect_trial(client, plan, work / "collection")
            report_campaign(plan["campaign_uri"])
            logger.info("Campaign report: %s/summary.md", plan["campaign_uri"])
        except BaseException as error:
            write_record(
                prefix_join(plan["trial_uri"], "coordinator-error.json"),
                {
                    "error": f"{type(error).__name__}: {error}",
                    "job_id": plan["job_id"],
                    "gpu_job_id": plan["gpu_job_id"],
                    "plan_uri": prefix_join(plan["trial_uri"], "plan.json"),
                },
            )
            if job is None:
                absent = "manifest" not in plan
                if not absent:
                    try:
                        client.job_status(JobName.from_string(plan["gpu_job_id"]))
                    except ConnectError as status_error:
                        absent = status_error.code == Code.NOT_FOUND
                        if not absent:
                            logger.warning("GPU submission status remains unknown: %s", status_error)
                record_preparation_failure(plan, f"{type(error).__name__}: {error}", gpu_job_absent=absent)
            raise
