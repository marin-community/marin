# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic infra canary.

Health checks cover the internal Marin controller, its public IAP/federation
path to every peer, and Finelog. Cluster/provisioning gauges run alongside them,
each as a collector on its own cadence. Samples are logged to stdout (picked up
by Cloud Logging on COS) and fanned to the sinks.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import click
from cluster import collect_jobs, collect_workers
from finelog.client.log_client import FlushResult, LogClient
from finelog.rpc import logging_pb2
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, zone_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from provisioning import collect_provisioning
from rigging.filesystem import load_cluster_config
from rigging.log_setup import configure_logging
from rigging.timing import Duration
from runner import METRIC_UP, Collector, CollectorRunner, MetricSink, health_collector
from sample import Sample
from sinks import FinelogTableSink, JsonlGcsSink

logger = logging.getLogger(__name__)

_MARIN_CONFIG = load_cluster_config("marin")

# Iris advertises the finelog log-server under this logical name in its endpoint
# registry; resolve it to a concrete address via list_endpoints (same name the
# iris worker uses).
LOG_SERVER_ENDPOINT_NAME = "/system/log-server"

# Default zones to canary when --zone is not given: the busiest europe-west4 and
# us-west4 zones in the fleet.
DEFAULT_ZONES = ("europe-west4-b", "us-west4-a")
DEFAULT_IRIS_CLUSTER = "marin"

# Scheduling probes request only enough resources for the pod to exercise Iris,
# federation, Kueue, and Kubernetes placement. They use the task image as-is:
# setup_scripts=[] avoids a workspace upload and uv-sync build phase.
CANARY_CPU = 0.1
CANARY_MEMORY = "128m"
CANARY_SCHEDULING_TIMEOUT = 60.0
CANARY_RUNTIME_TIMEOUT = 60.0
CANARY_WAIT_TIMEOUT = 100.0
CANARY_PEER_TIMEOUT = 120.0
CANARY_COLLECTOR_TIMEOUT = 240.0
CANARY_CADENCE = 300.0

# Provisioning gauge: a trailing window over the controller's iris.provisioning
# namespace, re-emitted each cadence. A 3h window smooths the bursty per-minute
# noise (stockouts persist for hours); 15min cadence is ample resolution and the
# finelog query is sub-second. The timeout covers the query plus aggregation.
PROVISION_WINDOW_HOURS = 3.0
PROVISION_CADENCE = 900.0
PROVISION_TIMEOUT = 60.0

# Cluster-state gauges. Workers is a single ListWorkers RPC paged client-side;
# jobs is one raw-SQL GROUP BY. Both are sub-second, so the cadences are about
# freshness (workers churn faster than the 24h job window) and the timeouts only
# cover a slow/hung controller.
WORKERS_CADENCE = 60.0
WORKERS_TIMEOUT = 30.0
JOBS_CADENCE = 120.0
JOBS_TIMEOUT = 30.0

# finelog-write probe: the key/source the canary writes under. Reads match on
# the KEY column (FetchLogsRequest.source + MatchScope are key matchers despite
# the field name), so the readback queries FINELOG_PROBE_KEY, not the source.
FINELOG_PROBE_KEY = "infra.canary.finelog_probe"
FINELOG_PROBE_SOURCE = "/canary/finelog-write-probe"
# Cap the flush wait: the StatsService write can be slow or hang, and an
# unbounded flush would block the probe to its timeout and leak the worker
# thread. Flush + readback stay under the finelog-write probe timeout.
FINELOG_FLUSH_TIMEOUT = 8.0
FINELOG_READBACK_TIMEOUT = 5.0
FINELOG_READBACK_POLL_INTERVAL = 0.25

# Where each Sample is persisted (beyond the stdout log line). The local dir is
# the VM's /var/lib/probes host mount; finished daily files roll up to GCS in the
# same region as the VM (no cross-region egress).
PROBE_RESULTS_DIR = Path("/var/lib/probes")
PROBE_RESULTS_GCS_PREFIX = f"gs://{_MARIN_CONFIG.region_buckets['us-central1'].name}/infra/probes"
PROBE_RESULTS_NAMESPACE = "infra.canary.metrics"


# ---- health checks --------------------------------------------------------


def probe_controller_ping(iris: RemoteClusterClient) -> bool:
    iris.list_workers()
    return True


def iris_job_succeeds(
    iris: RemoteClusterClient,
    target: str,
    constraints: list[job_pb2.Constraint],
) -> bool:
    job_id = JobName.root(
        "infra-probes",
        f"canary-{target}-{int(time.time())}-{uuid.uuid4().hex[:8]}",
    )
    submitted = iris.submit_job(
        job_id=job_id,
        entrypoint=Entrypoint.from_command("python", "-c", "import time; time.sleep(1)"),
        resources=ResourceSpec(cpu=CANARY_CPU, memory=CANARY_MEMORY).to_proto(),
        environment=EnvironmentSpec(setup_scripts=[]).to_proto(),
        constraints=constraints,
        max_retries_failure=0,
        max_retries_preemption=0,
        scheduling_timeout=Duration.from_seconds(CANARY_SCHEDULING_TIMEOUT),
        timeout=Duration.from_seconds(CANARY_RUNTIME_TIMEOUT),
    )
    status = iris.wait_for_job(submitted, timeout=CANARY_WAIT_TIMEOUT)
    return status.state == job_pb2.JOB_STATE_SUCCEEDED


def _federated_scheduling_health_sample(peer_id: str, succeeded: bool) -> Sample:
    probe_name = f"iris-job-submit/cluster/{peer_id}"
    return Sample.of(
        METRIC_UP,
        1.0 if succeeded else 0.0,
        probe=probe_name,
        cluster=peer_id,
        route="federation",
    )


def _federated_scheduling_sample(iris: RemoteClusterClient, peer_id: str) -> Sample:
    constraint = Constraint.create(
        key=CLUSTER_CONSTRAINT_KEY,
        op=ConstraintOp.EQ,
        value=peer_id,
    )
    try:
        succeeded = iris_job_succeeds(
            iris,
            f"cluster-{peer_id}",
            [constraint.to_proto()],
        )
    except Exception:
        logger.exception("federated scheduling probe failed for %s", peer_id)
        succeeded = False
    return _federated_scheduling_health_sample(peer_id, succeeded)


def collect_federated_scheduling(
    iris: RemoteClusterClient,
    peer_client: ControllerServiceClientSync,
) -> list[Sample]:
    """Return one scheduling-health sample for every Marin federation peer."""
    # Do not filter on the latest heartbeat's backend kind: an unreachable peer
    # has no live backend summary and is exactly the target that must report down.
    response = peer_client.list_peers(controller_pb2.Controller.ListPeersRequest())
    peer_ids = sorted(peer.peer_id for peer in response.peers)
    if not peer_ids:
        raise RuntimeError("the Marin controller reports no federation peers")

    executor = ThreadPoolExecutor(
        max_workers=len(peer_ids),
        thread_name_prefix="federated-scheduling-probe",
    )
    try:
        futures = {executor.submit(_federated_scheduling_sample, iris, peer_id): peer_id for peer_id in peer_ids}
        completed, pending = wait(futures, timeout=CANARY_PEER_TIMEOUT)
        samples = [
            future.result() if future in completed else _federated_scheduling_health_sample(peer_id, False)
            for future, peer_id in futures.items()
        ]
        for future in pending:
            peer_id = futures[future]
            logger.error("federated scheduling probe timed out for %s", peer_id)
        return samples
    finally:
        # Waiting here would discard completed samples when one peer call is stuck.
        executor.shutdown(wait=False, cancel_futures=True)


def probe_finelog_write(finelog: LogClient) -> bool:
    nonce = uuid.uuid4().hex
    ts_ms = int(time.time() * 1000)
    finelog.write_batch(
        key=FINELOG_PROBE_KEY,
        messages=[
            logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=ts_ms),
                source=FINELOG_PROBE_SOURCE,
                data=nonce,
                level=logging_pb2.LOG_LEVEL_INFO,
            )
        ],
    )
    if finelog.flush(timeout=FINELOG_FLUSH_TIMEOUT) != FlushResult.SUCCEEDED:
        return False
    # Re-read our own write until the nonce shows up or the readback budget is
    # spent: the write is durable now but the index lags it, so a single fetch races.
    # source here matches the KEY column (EXACT), not the entry's source field.
    deadline = time.monotonic() + FINELOG_READBACK_TIMEOUT
    while True:
        response = finelog.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=FINELOG_PROBE_KEY,
                match_scope=logging_pb2.MATCH_SCOPE_EXACT,
                since_ms=ts_ms - 1000,
                max_lines=64,
            )
        )
        if any(e.data == nonce for e in response.entries):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(FINELOG_READBACK_POLL_INTERVAL)


# ---- entrypoint -----------------------------------------------------------


def resolve_finelog_address(iris: RemoteClusterClient, name: str) -> str:
    """Resolve the finelog log-server address from iris's endpoint registry."""
    endpoints = iris.list_endpoints(name, exact=True)
    if not endpoints:
        raise ConnectionError(f"no {name!r} endpoint registered on the iris controller")
    return endpoints[0].address


def build_sinks(finelog: LogClient) -> list[MetricSink]:
    """Construct the sample sinks, skipping any that fail to initialize so the
    canary still runs (and reports samples) on a sink-side fault."""
    sinks: list[MetricSink] = []
    try:
        sinks.append(JsonlGcsSink(PROBE_RESULTS_DIR, PROBE_RESULTS_GCS_PREFIX))
    except Exception:
        logger.exception("failed to init JSONL/GCS sink; continuing without it")
    try:
        sinks.append(FinelogTableSink(finelog, PROBE_RESULTS_NAMESPACE))
    except Exception:
        logger.exception("failed to init finelog sink; continuing without it")
    return sinks


def build_collectors(
    iris: RemoteClusterClient,
    federated_iris: RemoteClusterClient,
    finelog: LogClient,
    query_client: ControllerServiceClientSync,
    zones: tuple[str, ...],
) -> list[Collector]:
    """Health checks plus the provisioning/workers/jobs gauges, each on its own cadence."""
    collectors = [
        health_collector("controller-ping", lambda: probe_controller_ping(iris), timeout=5.0, cadence=60.0),
        health_collector("finelog-write", lambda: probe_finelog_write(finelog), timeout=15.0, cadence=60.0),
        Collector(
            name="provisioning",
            collect=lambda: collect_provisioning(finelog, window_hours=PROVISION_WINDOW_HOURS),
            timeout=PROVISION_TIMEOUT,
            cadence=PROVISION_CADENCE,
        ),
        Collector(
            name="workers",
            collect=lambda: collect_workers(iris),
            timeout=WORKERS_TIMEOUT,
            cadence=WORKERS_CADENCE,
        ),
        Collector(
            name="jobs",
            collect=lambda: collect_jobs(query_client),
            timeout=JOBS_TIMEOUT,
            cadence=JOBS_CADENCE,
        ),
    ]
    for zone in zones:
        collectors.append(
            health_collector(
                f"iris-job-submit/{zone}",
                lambda z=zone: iris_job_succeeds(
                    iris,
                    z,
                    [zone_constraint(z).to_proto()],
                ),
                timeout=CANARY_COLLECTOR_TIMEOUT,
                cadence=CANARY_CADENCE,
            )
        )
    collectors.append(
        Collector(
            name="iris-job-submit/federation",
            collect=lambda: collect_federated_scheduling(federated_iris, query_client),
            timeout=CANARY_COLLECTOR_TIMEOUT,
            cadence=CANARY_CADENCE,
        )
    )
    return collectors


@click.command()
@click.option("--iris-endpoint", required=True, help="controller RPC, e.g. http://10.128.0.3:10000")
@click.option(
    "--iris-cluster",
    default=DEFAULT_IRIS_CLUSTER,
    show_default=True,
    help="named public Iris cluster used for authenticated federation probes",
)
@click.option(
    "--zone",
    "zones",
    multiple=True,
    help=f"GCP zone for iris-job-submit; repeat for multiple (default: {', '.join(DEFAULT_ZONES)})",
)
def main(iris_endpoint: str, iris_cluster: str, zones: tuple[str, ...]) -> None:
    zones = zones or DEFAULT_ZONES

    iris = RemoteClusterClient(controller_address=iris_endpoint)
    finelog = LogClient.connect(
        LOG_SERVER_ENDPOINT_NAME,
        resolver=lambda name: resolve_finelog_address(iris, name),
    )
    # Dedicated connect client for the jobs gauge's raw-SQL RPC; null-auth cluster,
    # so no credentials. RemoteClusterClient doesn't surface ExecuteRawQuery.
    query_client = rpc_client(iris_endpoint)

    # Federation probes deliberately traverse the public IAP edge. Ambient
    # service-account credentials become the IAP token; the Marin controller
    # then authenticates and signs each downstream peer handoff.
    with connect_controller(cluster_name=iris_cluster) as public_endpoint:
        federated_iris = RemoteClusterClient(
            controller_address=public_endpoint.url,
            interceptors=public_endpoint.credentials.interceptors(),
        )
        runner = CollectorRunner(sinks=build_sinks(finelog))
        for collector in build_collectors(iris, federated_iris, finelog, query_client, zones):
            runner.add(collector)
        runner.run()


if __name__ == "__main__":
    configure_logging()
    main()
