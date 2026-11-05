# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off CPU preparation acceptance; this entrypoint never submits GPU work."""

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import Constraint, ConstraintOp, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.evaluation.agentx.artifacts import read_record, write_record
from experiments.evaluation.agentx.control import source_run
from experiments.evaluation.agentx.recipe import load_series, series_path
from experiments.evaluation.agentx.runtime import resolve_release
from experiments.evaluation.agentx.workload import prepare_workload


def prepare(profile: str, source: dict, release: dict, root: str) -> None:
    logging.basicConfig(level=logging.INFO)
    info = get_job_info()
    assert info is not None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="agentx-preparation-acceptance-") as directory:
        work = Path(directory)
        first = prepare_workload(work / "first", series_path=series_path(profile))
        record = {
            "kind": "agentx-preparation-only",
            "profile": profile,
            "source": source,
            "selected_release": release,
            "job_id": str(info.job_id),
            "task_id": os.environ["IRIS_TASK_ID"],
            "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
            "workload": first,
            "gpu_jobs_submitted": 0,
        }
        output = str(Path(os.environ["IRIS_OUTPUT_DIR"]) / "preparation.json")
        write_record(output, record)
        restored = prepare_workload(work / "restored", series_path=series_path(profile))
        if restored["cache_status"] != "hit" or restored["archive_sha256"] != first["archive_sha256"]:
            raise ValueError("Native regional cache did not restore the published workload")
        record.update({"restored": restored, "elapsed_seconds": time.monotonic() - started, "status": "succeeded"})
        write_record(output, record)
        write_record(prefix_join(root, f"preparation/{profile}.json"), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["both", "h100", "gb200"], default="both")
    parser.add_argument("--vllm-release", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    profiles = ["h100", "gb200"] if args.profile == "both" else [args.profile]
    source = source_run()
    release = resolve_release(args.vllm_release)
    jobs = []
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            try:
                for profile in profiles:
                    series = load_series(series_path(profile))
                    cluster = series["profiles"][profile]["cluster"]
                    region = {"h100": "s3://marin-rno2a", "gb200": "s3://marin-us-east-08a"}[profile]
                    job = client.submit(
                        name=f"agentx-preparation-{source['run_id']}-{source['run_attempt']}-{profile}",
                        entrypoint=Entrypoint.from_callable(prepare, profile, source, release, args.root),
                        resources=ResourceSpec(cpu=2, memory="32GB", disk="64GB"),
                        environment=EnvironmentSpec(
                            sync_packages=["marin-core"], extras=["cpu"], env_vars={"MARIN_TEMP_PREFIX": region}
                        ),
                        constraints=[
                            Constraint.create(key="cluster", op=ConstraintOp.EQ, value=cluster),
                            preemptible_constraint(False),
                        ],
                        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                        scheduling_timeout=Duration.from_seconds(1800),
                        timeout=Duration.from_seconds(1800),
                        max_retries_failure=0,
                        max_retries_preemption=0,
                        task_image=series["task_image"],
                    )
                    jobs.append((profile, job))
                    write_record(str(args.output / "jobs.json"), {"jobs": [str(handle.job_id) for _, handle in jobs]})
                    print(f"CPU preparation: {job.job_id}; {job.job_id.dashboard_url(endpoint.url)}", flush=True)
                for profile, job in jobs:
                    job.wait(timeout=1200, stream_logs=True)
                    status = client.job_status(job.job_id)
                    if status.resources.device is not None:
                        raise ValueError("Preparation acceptance must request CPU resources only")
                    record = read_record(prefix_join(args.root, f"preparation/{profile}.json"))
                    record["native_state"] = status.state.value
                    record["native_resources"] = {
                        "cpu_millicores": status.resources.cpu_millicores,
                        "memory_bytes": status.resources.memory_bytes,
                        "disk_bytes": status.resources.disk_bytes,
                        "device": None,
                    }
                    write_record(str(args.output / f"{profile}.json"), record)
            except BaseException:
                for _, job in jobs:
                    if not is_job_finished(client.job_status(job.job_id).state):
                        job.cancel()
                raise


if __name__ == "__main__":
    main()
