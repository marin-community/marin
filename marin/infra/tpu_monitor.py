import logging
import time
from collections.abc import Iterable
from pathlib import Path

import ray
import yaml
from google.cloud import tpu_v2alpha1
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)

BAD_STATES = {
    tpu_v2alpha1.Node.State.PREEMPTED,
    tpu_v2alpha1.Node.State.TERMINATED,
}
GOOD_STATE = tpu_v2alpha1.Node.State.READY


@ray.remote
class TPUMonitor:
    """Monitor TPUs in a Ray cluster and clean up stale ones."""

    def __init__(
        self,
        project: str | None = None,
        zone: str | None = None,
        cluster_name: str | None = None,
        *,
        wait_seconds: int = 600,
        config_path: str | Path = "~/ray_bootstrap_config.yaml",
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
        self.missing_since: dict[str, float] = {}
        self.incomplete_since: dict[str, float] = {}

    def _list_ray_hostnames(self) -> set[str]:
        return {node["NodeManagerHostname"] for node in ray.nodes() if node.get("Alive", False)}

    def _cluster_resources(self) -> dict[str, float]:
        """Return cluster resource counts."""
        try:
            return ray.cluster_resources()
        except Exception as e:
            logger.error("Failed to list cluster resources: %s", e)
            return {}

    def _delete_node(self, name: str, reason: str) -> None:
        logger.warning("Deleting TPU %s due to %s", name, reason)
        try:
            self.tpu_client.delete_node(name=name)
        except Exception as e:
            logger.error("Failed to delete TPU %s: %s", name, e)

    def check_once(self) -> None:
        hostnames = self._list_ray_hostnames()
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

            if name not in hostnames:
                first = self.missing_since.setdefault(name, now)
                if now - first > self.wait_seconds:
                    self._delete_node(node.name, "missing from ray")
                    self.missing_since.pop(name, None)
            else:
                self.missing_since.pop(name, None)

            expected = len(getattr(node, "network_endpoints", []))
            actual = int(resources.get(name, 0))
            if actual != expected:
                first = self.incomplete_since.setdefault(name, now)
                if now - first > self.wait_seconds:
                    self._delete_node(node.name, "wrong worker count")
                    self.incomplete_since.pop(name, None)
            else:
                self.incomplete_since.pop(name, None)

    def run(self, interval_s: int = 60) -> None:
        """Run the monitor loop."""
        while True:
            self.check_once()
            time.sleep(interval_s)


def start_tpu_monitor_on_head(*args, **kwargs):
    """Launch :class:`TPUMonitor` on the Ray head node."""

    head_ip = ray.util.get_node_ip_address()
    node_id = next(
        (n["NodeID"] for n in ray.nodes() if n.get("NodeManagerAddress") == head_ip),
        None,
    )
    if node_id is None:
        node_id = ray.get_runtime_context().get_node_id()

    strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    return TPUMonitor.options(
        num_cpus=0,
        scheduling_strategy=strategy,
        lifetime="detached",
        name="tpu_monitor",
        get_if_exists=True,
    ).remote(*args, **kwargs)
