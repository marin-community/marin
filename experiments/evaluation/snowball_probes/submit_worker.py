# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register an owner-controlled lab run and submit its driver to an Iris CPU worker."""

import argparse
import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import dotenv_values
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import job_pb2
from marina.client import marina_request

from experiments.evaluation.snowball_probes.cluster_worker import run
from experiments.evaluation.snowball_probes.config import CONCURRENT_CHATS, TOKEN_CAP
from experiments.evaluation.snowball_probes.store import Store
from experiments.evaluation.snowball_probes.worker import (
    APPLET_ORIGIN,
    INFERENCE_TIMEOUT,
    SHELLSIM_CPU_LIMIT,
    Endpoints,
    Worker,
    inspiration_seeds,
)

logger = logging.getLogger(__name__)


async def configuration(args: argparse.Namespace, client: IrisClient, origin: str, seeds: list[dict], key: str) -> dict:
    """Resolve the pinned servers before registering an immutable run configuration."""
    store = Store(args.state_dir / "state.sqlite", args.token_cap)
    worker = Worker(args, store, Endpoints(client, origin, args.coordinator, args.glm_gateway), seeds, key)
    try:
        await worker.endpoints.refresh(worker.http)
        if not worker.endpoints.metadata:
            raise RuntimeError("Wait for the pinned candidate metadata before submitting the driver")
        return worker.configuration()
    finally:
        await worker.http.aclose()
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("applet", "coordinator", "glm-gateway", "run-id", "job-name", "user", "worker-identity"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--glm-env", type=Path, required=True)
    parser.add_argument("--previous-run")
    parser.add_argument("--inspiration-cache", type=Path)
    parser.add_argument("--marina-url", default=APPLET_ORIGIN)
    parser.add_argument("--concurrency", type=int, default=CONCURRENT_CHATS)
    parser.add_argument("--token-cap", type=int, default=TOKEN_CAP)
    parser.add_argument("--shellsim-cpu-limit", type=int, default=SHELLSIM_CPU_LIMIT)
    parser.add_argument("--inference-timeout", type=float, default=INFERENCE_TIMEOUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    key = dotenv_values(args.glm_env).get("GLM_API_TOKEN")
    if not key:
        raise ValueError("GLM_API_TOKEN is missing from the supplied dotenv file")
    args.worker_job = str(JobName.root(args.user, args.job_name))
    prefix = f"/a/{args.applet}/api"
    generated = 0
    if args.previous_run:
        previous = marina_request(args.marina_url, "GET", f"{prefix}/runs/{args.previous_run}")
        assert isinstance(previous, dict)
        if previous["state"] != "stopped" or previous["reserved_tokens"]:
            raise ValueError("Stop the previous worker and flush its reservations before moving the lab")
        generated = previous["generated_tokens"]
    with (
        TemporaryDirectory(prefix="snowball-launch-") as temporary,
        connect_controller(cluster_name="marin") as endpoint,
    ):
        args.state_dir = Path(temporary)
        seeds = inspiration_seeds(args.inspiration_cache or args.state_dir / "wildchat.json")
        client = IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials)
        config = asyncio.run(configuration(args, client, endpoint.url, seeds, key))
        marina_request(
            args.marina_url,
            "POST",
            f"{prefix}/runs",
            json_body={
                "run_id": args.run_id,
                "config": config,
                "token_cap": args.token_cap,
                "concurrency": args.concurrency,
                "generated_tokens": generated,
                "reserved_tokens": 0,
                "active_chats": 0,
                "state": "waiting_for_worker",
            },
        )
        marina_request(
            args.marina_url, "POST", f"{prefix}/runs/{args.run_id}/worker", json_body={"identity": args.worker_identity}
        )
        arguments = []
        for name in (
            "applet",
            "coordinator",
            "glm_gateway",
            "run_id",
            "marina_url",
            "concurrency",
            "token_cap",
            "shellsim_cpu_limit",
            "inference_timeout",
        ):
            arguments.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
        job = client.submit(
            name=args.job_name,
            user=args.user,
            entrypoint=Entrypoint.from_callable(run, arguments, seeds),
            resources=ResourceSpec(cpu=4, memory="4GB", disk="8GB"),
            environment=EnvironmentSpec(sync_packages=["marin-marina"], env_vars={"GLM_API_TOKEN": key}),
            constraints=[preemptible_constraint(False)],
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            max_retries_failure=10,
            max_task_failures=10,
        )
        logger.info("Submitted cluster conversation worker %s for run %s", job.job_id, args.run_id)


if __name__ == "__main__":
    main()
