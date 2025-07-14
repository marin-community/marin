import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import ray
import yaml
from google.cloud import tpu_v2alpha1
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger("ray")


BAD_STATES = {
    tpu_v2alpha1.Node.State.PREEMPTED,
    tpu_v2alpha1.Node.State.TERMINATED,
}
GOOD_STATE = tpu_v2alpha1.Node.State.READY


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
        config_path: str | Path = "~/ray_bootstrap_config.yaml",
        dry_run: bool = False,
    ) -> None:
        path = Path(config_path).expanduser()
        cfg = {}
        if project is None or zone is None or cluster_name is None:
            try:
                with path.open() as f:
                    cfg = yaml.safe_load(f)
            except Exception as e:  # pragma: no cover - best effort
                logger.error("Failed to load config %s: %s", path, e)

        provider = cfg.get("provider", {}) if isinstance(cfg, dict) else {}
        self.project = project or provider.get("project_id")
        self.zone = zone or provider.get("availability_zone") or provider.get("zone")
        self.cluster_name = cluster_name or cfg.get("cluster_name")
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
        logger.info(f"TpuMonitor started for project={self.project}, zone={self.zone}, cluster_name={self.cluster_name}")

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
                    logger.warning("TPU %s has more workers (%d) than expected (%d). Deleting.", name, actual, expected)
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
    config_path: str | Path = "~/ray_bootstrap_config.yaml",
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
        name="tpu_monitor",
        get_if_exists=True,
    ).remote(
        project=project,
        zone=zone,
        cluster_name=cluster_name,
        wait_seconds=wait_seconds,
        config_path=config_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    import argparse

    ray.init("auto", namespace="marin")

    args = argparse.ArgumentParser(description="Start a TPU monitor on the Ray head node.")
    args.add_argument("--project", type=str, help="GCP project ID", default=None)
    args.add_argument("--zone", type=str, help="GCP zone", default=None)
    args.add_argument("--cluster_name", type=str, help="Ray cluster name", default=None)
    args.add_argument("--wait_seconds", type=int, help="Wait time before deleting a TPU", default=600)
    args.add_argument(
        "--config_path", type=str, help="Path to Ray bootstrap config", default="~/ray_bootstrap_config.yaml"
    )
    args.add_argument("--dry_run", action="store_true", help="Run in dry run mode (do not delete TPUs)")

    args = args.parse_args()

    monitor = start_tpu_monitor_on_head(
        project=args.project,
        zone=args.zone,
        cluster_name=args.cluster_name,
        wait_seconds=args.wait_seconds,
        config_path=args.config_path,
        dry_run=args.dry_run,
    )
    print("TPU Monitor started:", monitor)
