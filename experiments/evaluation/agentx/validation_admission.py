# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary CPU-only check of GitHub admission to the Hero peer."""

import argparse
import json
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client import resolve_job_user
from iris.cluster.constraints import Constraint, ConstraintOp, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from rigging.timing import Duration

from experiments.evaluation.agentx.artifacts import write_record
from experiments.evaluation.agentx.control import source_run
from experiments.evaluation.agentx.recipe import load_series

COMMAND = """
import json, os, platform, sys
from pathlib import Path
record = {
    "kind": "cpu-admission-validation",
    "source": json.loads(sys.argv[1]),
    "requested_gpus": 0,
    "task_id": os.environ["IRIS_TASK_ID"],
    "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
    "node": os.environ["IRIS_NODE_NAME"],
    "machine": platform.machine(),
}
(Path(os.environ["IRIS_OUTPUT_DIR"]) / "admission.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    source = source_run()
    name = f"agentx-{source['run_id']}-{source['run_attempt']}-8a-cpu-admission"
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            if args.cancel:
                job = client.job(JobName.root(resolve_job_user(None), name))
                if not is_job_finished(job.state):
                    job.cancel()
                return
            args.work.mkdir(parents=True, exist_ok=True)
            write_record(str(args.work / "source.json"), source)
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", "-c", COMMAND, json.dumps(source)),
                name=name,
                resources=ResourceSpec(cpu=1, memory="2GB", disk="8GB"),
                environment=EnvironmentSpec(),
                constraints=[
                    Constraint.create(key="cluster", op=ConstraintOp.EQ, value="cw-us-east-08a"),
                    preemptible_constraint(False),
                ],
                priority_band=job_pb2.PRIORITY_BAND_BATCH,
                scheduling_timeout=Duration.from_seconds(300),
                timeout=Duration.from_seconds(120),
                max_retries_failure=0,
                max_retries_preemption=0,
                task_image=load_series()["task_image"],
            )
            write_record(str(args.work / "submission.json"), {"job_id": str(job.job_id), "source": source})
            print(f"CPU admission check: {job.job_id}", flush=True)
            try:
                status = job.wait(timeout=300, stream_logs=True)
                write_record(str(args.work / "status.json"), {"job_id": str(job.job_id), "state": status.state.name})
            finally:
                if not is_job_finished(job.state):
                    job.cancel()


if __name__ == "__main__":
    main()
