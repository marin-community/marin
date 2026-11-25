# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit independent single-node replicas through the maintained Iris serving backend."""

import argparse
import json

from fray.iris_backend import convert_constraints, convert_environment, convert_resources
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint
from iris.rpc.proto_display import priority_band_value
from marin.inference.iris import IrisServiceConfig, run_iris_service
from rigging.config_discovery import find_project_root
from rigging.timing import Duration

from experiments.glm_synthesis_serving import pool

REPLICA_RETRIES = 10


def replica_service(endpoint: str) -> IrisServiceConfig:
    """Serve one engine directly, with no broker or CPU coordinator in its data path."""
    return IrisServiceConfig(
        model=pool.model(),
        engine=pool.engine(),
        iris=pool.iris_config(),
        endpoint_name=endpoint,
        timeout_hours=pool.SERVICE_TIMEOUT_HOURS,
        controller_proxy_timeout_seconds=pool.REQUEST_TIMEOUT_SECONDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--priority", choices=("interactive",), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.start_index < 0 or args.start_index + args.count > pool.POOL_INSTANCES:
        raise ValueError(f"Replica indices must fit within the {pool.POOL_INSTANCES}-engine pool")
    workspace = find_project_root()
    if workspace is None:
        raise RuntimeError("Run from a Marin checkout")
    replicas = [
        (f"{pool.JOB_PREFIX}-{args.tag}-{index:02d}", f"{pool.ENDPOINT_NAME}-{args.tag}-{index:02d}")
        for index in range(args.start_index, args.start_index + args.count)
    ]
    resources = pool.worker_resources()
    if args.dry_run:
        print(json.dumps({"replicas": replicas, "gpus": args.count * pool.GPUS_PER_ENGINE, "resources": str(resources)}))
        return
    with connect_controller(cluster_name=pool.CLUSTER) as connection:
        with IrisClient.remote(connection.url, workspace=workspace, credentials=connection.credentials) as client:
            for name, endpoint in replicas:
                service = replica_service(endpoint)
                job = client.submit(
                    name=name,
                    entrypoint=Entrypoint.from_callable(run_iris_service, service),
                    resources=convert_resources(resources),
                    constraints=convert_constraints(resources),
                    environment=convert_environment(service.iris.worker_environment),
                    ports=["http"],
                    priority_band=priority_band_value(args.priority),
                    max_retries_failure=REPLICA_RETRIES,
                    max_task_failures=REPLICA_RETRIES,
                    max_retries_preemption=pool.MAX_RETRIES_PREEMPTION,
                    timeout=Duration.from_hours(pool.SERVICE_TIMEOUT_HOURS),
                )
                print(json.dumps({"job_id": str(job.job_id), "endpoint": endpoint}), flush=True)


if __name__ == "__main__":
    main()
