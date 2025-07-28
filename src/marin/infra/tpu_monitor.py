import logging
import threading
import time
from collections.abc import Iterable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger("ray")

try:
    import requests
    from google.cloud import compute_v1, tpu_v2alpha1

    BAD_STATES = {
        tpu_v2alpha1.Node.State.PREEMPTED,
        tpu_v2alpha1.Node.State.TERMINATED,
    }
    GOOD_STATE = tpu_v2alpha1.Node.State.READY

    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/"
    METADATA_HEADERS = {"Metadata-Flavor": "Google"}

    @ray.remote
    class TpuMonitor:
        """
        Monitor TPUs in a Ray cluster and clean up stale ones.
        Automatically deletes TPUs that are in a bad state or have an incorrect number of workers
        registered in Ray (after a specified wait time).
        """

        def __init__(
            self,
            project: str | None = None,
            zone: str | None = None,
            cluster_name: str | None = None,
            *,
            wait_seconds: int = 600,
            dry_run: bool = False,
        ) -> None:
            if project is None:
                project = self._get_gcp_metadata("project/project-id")

            if zone is None:
                zone_str = self._get_gcp_metadata("instance/zone")
                if zone_str:
                    zone = zone_str.split("/")[-1]

            if cluster_name is None:
                instance_name = self._get_gcp_metadata("instance/name")
                if project and zone and instance_name:
                    try:
                        client = compute_v1.InstancesClient()
                        instance_obj = client.get(project=project, zone=zone, instance=instance_name)
                        cluster_name = instance_obj.labels.get("ray-cluster-name")
                    except Exception as e:
                        logger.warning(f"Could not get cluster name from instance labels: {e}")

            self.project = project
            self.zone = zone
            self.cluster_name = cluster_name

            if self.project is None or self.zone is None or self.cluster_name is None:
                msg = (
                    "Could not determine project, zone, or cluster_name from metadata service or arguments. "
                    f"project={self.project}, zone={self.zone}, cluster_name={self.cluster_name}"
                )
                logger.error(msg)
                raise ValueError(msg)

            self.wait_seconds = wait_seconds
            self.tpu_client = tpu_v2alpha1.TpuClient()
            self.dry_run = dry_run
            self.incomplete_since: dict[str, float] = {}

            # start a background thread to run the monitor
            self._stop_event = threading.Event()
            self._monitor_thread = threading.Thread(
                target=self._run,
                kwargs={"interval_s": 60},
                daemon=True,
            )
            self._monitor_thread.start()
            logger.info(
                f"TpuMonitor started for project={self.project}, zone={self.zone}, cluster_name={self.cluster_name}"
            )

        def _get_gcp_metadata(self, path: str) -> str | None:
            try:
                response = requests.get(METADATA_URL + path, headers=METADATA_HEADERS, timeout=5)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning("Failed to get GCP metadata for path %s: %s", path, e)
                return None

        def _cluster_resources(self) -> dict[str, float]:
            """Return cluster resource counts."""
            try:
                return ray.cluster_resources()
            except Exception as e:
                logger.error("Failed to list cluster resources: %s", e)
                return {}

        def _delete_node(self, name: str, reason: str) -> None:
            if self.dry_run:
                logger.warning(
                    "Would delete TPU %s due to %s (dry run mode)",
                )
                return
            try:
                logger.warning("Deleting TPU %s due to %s", name, reason)
                self.tpu_client.delete_node(name=name)
            except Exception as e:
                logger.error("Failed to delete TPU %s: %s", name, e)

        def _check_once(self) -> None:
            resources = self._cluster_resources()
            parent = f"projects/{self.project}/locations/{self.zone}"
            nodes: Iterable[tpu_v2alpha1.Node] = self.tpu_client.list_nodes(parent=parent)
            now = time.time()

            for node in nodes:
                name = node.name.split("/")[-1]
                cluster_label = getattr(node, "labels", {}).get("ray-cluster-name")
                if cluster_label != self.cluster_name:
                    # Node is not part of this Ray cluster
                    continue
                if node.state in BAD_STATES:
                    self._delete_node(node.name, node.state.name)
                    continue
                if node.state != GOOD_STATE:
                    continue

                expected = len(getattr(node, "network_endpoints", []))
                actual = int(resources.get(name, 0))
                if actual != expected:
                    if actual > expected:
                        logger.warning(
                            "TPU %s has more workers (%d) than expected (%d). Deleting.", name, actual, expected
                        )
                        self._delete_node(node.name, "too many workers")
                        continue
                    else:
                        first = self.incomplete_since.setdefault(name, now)
                        if now - first > self.wait_seconds:
                            self._delete_node(node.name, "wrong worker count")
                            self.incomplete_since.pop(name, None)
                else:
                    self.incomplete_since.pop(name, None)

        def _run(self, interval_s: int = 60) -> None:
            """Run the monitor loop."""
            while not self._stop_event.is_set():
                self._check_once()
                time.sleep(interval_s)

    def start_tpu_monitor_on_head(
        project: str | None = None,
        zone: str | None = None,
        cluster_name: str | None = None,
        *,
        wait_seconds: int = 600,
        dry_run: bool = False,
    ):
        """Launch TpuMonitor on the Ray head node."""
        logger.info("Ensuring TpuMonitor is running on the Ray head node...")

        head_ip = ray.util.get_node_ip_address()
        node_id = next(
            (n["NodeID"] for n in ray.nodes() if n.get("NodeManagerAddress") == head_ip),
            None,
        )
        if node_id is None:
            node_id = ray.get_runtime_context().get_node_id()

        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

        return TpuMonitor.options(
            num_cpus=0,
            scheduling_strategy=strategy,
            lifetime="detached",
            name="tpu_monitor-test",
            get_if_exists=True,
        ).remote(
            project=project,
            zone=zone,
            cluster_name=cluster_name,
            wait_seconds=wait_seconds,
            dry_run=dry_run,
        )

except ImportError as e:

    class TpuMonitor:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Please install marin[gcp] to use TpuMonitor.")

    _ERROR = e

    def start_tpu_monitor_on_head(*args, **kwargs):  # type: ignore
        logger.warning(
            "TpuMonitor is not available because GCP dependencies are not installed. "
            "Install with `pip install 'marin[gcp]'`",
            exc_info=_ERROR,
        )
        return None


if __name__ == "__main__":
    import argparse

    ray.init("auto", namespace="marin")

    args = argparse.ArgumentParser(description="Start a TPU monitor on the Ray head node.")
    args.add_argument("--project", type=str, help="GCP project ID", default=None)
    args.add_argument("--zone", type=str, help="GCP zone", default=None)
    args.add_argument("--cluster_name", type=str, help="Ray cluster name", default=None)
    args.add_argument("--wait_seconds", type=int, help="Wait time before deleting a TPU", default=600)
    args.add_argument("--dry_run", action="store_true", help="Run in dry run mode (do not delete TPUs)")

    args = args.parse_args()

    monitor = start_tpu_monitor_on_head(
        project=args.project,
        zone=args.zone,
        cluster_name=args.cluster_name,
        wait_seconds=args.wait_seconds,
        dry_run=args.dry_run,
    )
    print("TPU Monitor started:", monitor)
