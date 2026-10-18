# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve Snowball v0, the relay student, as a root job, so a lane can use GPUs in another cluster.

The job holds v0's 8xH100 vLLM job (the policy's serving config, as ``harbor_glm.student_inference_config``)
under a fixed endpoint name. A capability URL minted for that name keeps working when Iris moves the vLLM job
after a preemption, and the federation parent relays it to the lane's cluster. Submit it, wait for the endpoint,
then hand the lane ``<capability URL>/v1`` as ``STUDENT_BASE_URL`` (environment only; the URL is a credential):

    uv run iris --cluster=marin job run --target-cluster cw-rno2a --cpu 2 --memory 8GB --job-name v0-student-<lane> \
      --no-wait -- python -m experiments.post_training.sft_shootout.serve_student \
      --cluster cw-rno2a --endpoint /serve/v0-student-<lane>
    uv run iris --cluster=cw-rno2a endpoints mint /serve/v0-student-<lane> --ttl-hours 168
"""

import argparse
import logging

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest
from marin.inference.iris import IrisServiceConfig, run_iris_service

from experiments.post_training.sft_shootout.harbor_glm import student_inference_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster", required=True, help="This job's cluster.")
    parser.add_argument("--endpoint", required=True, help="Fixed endpoint name, e.g. /serve/v0-student-tmax-lane1.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = student_inference_config(args.cluster)
    iris = config.iris
    service = IrisServiceConfig(
        model=config.model,
        engine=config.engine,
        iris=iris,
        endpoint_name=args.endpoint,
        instances=config.instances,
        broker=config.broker,
        timeout_hours=24 * 365,
        port_name=None,
    )
    job = current_client().submit(
        JobRequest(
            name="v0",
            entrypoint=Entrypoint.from_callable(run_iris_service, args=(service,)),
            resources=iris.worker_resources,
            environment=iris.worker_environment,
            max_retries_failure=iris.max_retries_failure,
            max_retries_preemption=iris.max_retries_preemption,
            max_task_failures=iris.max_retries_failure,
            priority=iris.priority,
        )
    )
    logger.info("v0 job %s serves %s", job.job_id, args.endpoint)
    job.wait()


if __name__ == "__main__":
    main()
