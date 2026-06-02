#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up a (kind or CoreWeave) cluster, run one coscheduled GPU gang, tear it down.

This is an end-to-end smoke test for Iris K8s-direct-provider *gang admission*
via Kueue (see .agents/projects/20260529_iris_k8s_gang_admission.md). It drives
``K8sTaskProvider`` directly against a real Kubernetes API server — no Iris
controller, no worker daemons — which is the smallest harness that exercises
the new code on a live cluster:

  1. setup   — provision a cluster (kind: ``kind create`` + install/configure
               Kueue + queues; coreweave: ``ensure_nodepools`` so CKS spins up
               the H100 pool, Kueue assumed admin-provisioned).
  2. run     — build ``replicas`` coscheduled ``RunTaskRequest``s sharing one
               pod-group, ``provider.sync`` to apply them (Kueue gates them
               until the whole group can be admitted), then poll until every
               task reaches SUCCEEDED.
  3. teardown — delete the gang's pods + Kueue Workloads; for kind delete the
               cluster, for coreweave optionally delete the NodePool.

The job entrypoint is a placeholder standing in for the real jax/levanter
distributed-init logic: it prints its rank/world from the Iris env vars, sleeps,
and exits 0.

Usage:
    cd lib/iris
    uv run --group dev python scripts/gpu_gang_smoke.py --config config/kind-gpu-smoke.yaml
    uv run --group dev python scripts/gpu_gang_smoke.py --config config/kind-gpu-smoke.yaml --keep
    # real CoreWeave (provisions 24 H100s — costs money):
    uv run --group dev python scripts/gpu_gang_smoke.py \
        --config config/coreweave-gpu-smoke.yaml --i-understand-the-cost
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import install_kueue
import yaml
from iris.cluster.config import make_provider as build_task_provider
from iris.cluster.controller.direct_provider import DirectProviderBatch
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.providers.k8s.controller import K8sControllerProvider
from iris.cluster.providers.k8s.service import CloudK8sService
from iris.cluster.providers.k8s.tasks import (
    _KUEUE_POD_GROUP_NAME,
    K8sTaskProvider,
)
from iris.cluster.providers.types import Labels, local_queue_name
from iris.cluster.types import JobName
from iris.rpc import config_pb2, job_pb2

# kind nodes carry none of CoreWeave's topology labels, so the harness stamps the
# same label keys the cw-ib ResourceFlavor + Topology CRs reference (see
# install_kueue.py) onto the worker nodes — putting them all in one synthetic
# leafgroup so TAS treats the kind cluster as a single IB fabric. flavor=infiniband
# is the ResourceFlavor node selector; fabric/superpod/leafgroup are Topology levels.
_KIND_NODE_TOPOLOGY_LABELS = {
    "backend.coreweave.cloud/flavor": "infiniband",
    "backend.coreweave.cloud/fabric": "fabric-0",
    "backend.coreweave.cloud/superpod": "superpod-0",
    "backend.coreweave.cloud/leafgroup": "leafgroup-0",
}
# label_prefix Iris uses on kind; the LocalQueue it reconciles is "{prefix}-lq".
_KIND_LABEL_PREFIX = "iris"

# CoreWeave smoke: a label_prefix DISTINCT from any real Iris cluster on the
# shared CKS cluster, so this smoke's NodePool / LocalQueue / managed-node labels
# never collide with — or get reaped alongside — production resources. The real
# marin cluster uses "iris-ci"; we deliberately use "iris" so iris-iris-managed /
# iris-iris-scale-group keys are isolated.
_CW_LABEL_PREFIX = "iris"
# Admin-provisioned ClusterQueue (install_kueue.py --variant coreweave
# --with-queues) that this smoke's namespaced LocalQueue binds to.
_CW_CLUSTER_QUEUE = "iris-cq"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("gpu_gang_smoke")

_PRIORITY_BANDS = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}

_TERMINAL_OK = {job_pb2.TASK_STATE_SUCCEEDED}
_TERMINAL_BAD = {
    job_pb2.TASK_STATE_FAILED,
    job_pb2.TASK_STATE_WORKER_FAILED,
    job_pb2.TASK_STATE_KILLED,
}

# Placeholder for the real jax/levanter distributed-init logic.
_PLACEHOLDER_PY = textwrap.dedent(
    """
    import os, socket, time
    task = os.environ.get("IRIS_TASK_ID", "?")
    world = os.environ.get("IRIS_NUM_TASKS", "?")
    host = socket.gethostname()
    print(f"[gang-smoke] task={{task}} world={{world}} host={{host}} starting", flush=True)
    print(f"[gang-smoke] (placeholder) would jax.distributed.initialize() across {{world}} hosts", flush=True)
    time.sleep({runtime})
    print(f"[gang-smoke] task={{task}} done", flush=True)
    """
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _parse_size(value: str | int) -> int:
    """Parse '256Mi' / '8Gi' / '512MB' / '0' / int into bytes (binary units)."""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s or s == "0":
        return 0
    m = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([kmgt]i?b?)?\s*", s)
    if not m:
        raise ValueError(f"cannot parse size {value!r}")
    num = float(m.group(1))
    unit = (m.group(2) or "").lower().rstrip("b")
    factor = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[unit.rstrip("i")]
    return int(num * factor)


def _parse_cpu_millicores(value: str | int | float) -> int:
    """Parse '0.25' / '32' cores or '250m' into millicores."""
    s = str(value).strip()
    if s.endswith("m"):
        return int(s[:-1])
    return int(float(s) * 1000)


@dataclass
class JobSpec:
    name: str
    replicas: int
    group_by: str
    priority_band: int
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    gpu_variant: str
    gpu_count: int
    runtime_seconds: int


@dataclass
class KueueSpec:
    install: bool
    version: str = "v0.11.0"
    cluster_queue_cpu: str = "12"
    cluster_queue_memory: str = "8Gi"


@dataclass
class KindSpec:
    cluster_name: str = "iris-gpu-smoke"
    workers: int = 3


@dataclass
class CoreweaveSpec:
    region: str = ""
    scale_group: str = ""
    instance_type: str = ""
    nodes: int = 3
    ensure_nodepools: bool = True
    delete_nodepools_on_teardown: bool = True


@dataclass
class SmokeConfig:
    target: str
    namespace: str
    kubeconfig_path: str
    local_queue: str
    image: str
    job: JobSpec
    kueue: KueueSpec
    kind: KindSpec = field(default_factory=KindSpec)
    coreweave: CoreweaveSpec = field(default_factory=CoreweaveSpec)


def load_config(path: Path) -> SmokeConfig:
    raw = yaml.safe_load(path.read_text())
    j = raw["job"]
    band = _PRIORITY_BANDS[j.get("priority_band", "batch")]
    job = JobSpec(
        name=j["name"],
        replicas=int(j["replicas"]),
        group_by=j.get("group_by", ""),
        priority_band=band,
        cpu_millicores=_parse_cpu_millicores(j.get("cpu", "0.25")),
        memory_bytes=_parse_size(j.get("memory", "256Mi")),
        disk_bytes=_parse_size(j.get("disk", "0")),
        gpu_variant=j.get("gpu_variant", ""),
        gpu_count=int(j.get("gpu_count", 0)),
        runtime_seconds=int(j.get("runtime_seconds", 8)),
    )
    kq = raw.get("kueue", {})
    kueue = KueueSpec(
        install=bool(kq.get("install", False)),
        version=kq.get("version", "v0.11.0"),
        cluster_queue_cpu=str(kq.get("cluster_queue_cpu", "12")),
        cluster_queue_memory=str(kq.get("cluster_queue_memory", "8Gi")),
    )
    return SmokeConfig(
        target=raw["target"],
        namespace=raw["namespace"],
        kubeconfig_path=str(Path(raw["kubeconfig_path"]).expanduser()),
        local_queue=raw["local_queue"],
        image=raw["image"],
        job=job,
        kueue=kueue,
        kind=KindSpec(**raw.get("kind", {})),
        coreweave=CoreweaveSpec(**raw.get("coreweave", {})),
    )


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], *, stdin: str | None = None, check: bool = True, quiet: bool = False) -> str:
    if not quiet:
        logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0 and check:
        logger.error(
            "command failed (%d): %s\nstdout:\n%s\nstderr:\n%s",
            proc.returncode,
            " ".join(cmd),
            proc.stdout,
            proc.stderr,
        )
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout


class Kubectl:
    """Thin kubectl wrapper bound to one kubeconfig (for cluster bootstrap only).

    The gang itself goes through K8sTaskProvider/CloudK8sService; this is just
    for create/install/apply/wait during setup and teardown.
    """

    def __init__(self, kubeconfig: str) -> None:
        self._base = ["kubectl", "--kubeconfig", kubeconfig]

    def apply(self, manifest: str, *, server_side: bool = False) -> None:
        cmd = [*self._base, "apply"]
        if server_side:
            cmd += ["--server-side", "--force-conflicts"]
        cmd += ["-f", "-"]
        _run(cmd, stdin=manifest)

    def get(self, *args: str) -> str:
        return _run([*self._base, "get", *args], check=False, quiet=True)

    def raw(self, *args: str, check: bool = True) -> str:
        return _run([*self._base, *args], check=check)


# ---------------------------------------------------------------------------
# Gang construction + run loop
# ---------------------------------------------------------------------------
def _build_requests(cfg: SmokeConfig) -> list[job_pb2.RunTaskRequest]:
    """Build `replicas` coscheduled RunTaskRequests sharing one pod-group.

    Sibling task ids share the parent path `/{job.name}` so the provider hashes
    them into a single Kueue pod-group-name (see tasks.py _pod_group_name).
    """
    reqs: list[job_pb2.RunTaskRequest] = []
    placeholder = _PLACEHOLDER_PY.format(runtime=cfg.job.runtime_seconds)
    for i in range(cfg.job.replicas):
        req = job_pb2.RunTaskRequest()
        req.task_id = f"/{cfg.job.name}/{i}"
        req.attempt_id = 0
        req.num_tasks = cfg.job.replicas
        req.entrypoint.run_command.argv.extend(["python", "-c", placeholder])
        req.resources.cpu_millicores = cfg.job.cpu_millicores
        req.resources.memory_bytes = cfg.job.memory_bytes
        if cfg.job.disk_bytes:
            req.resources.disk_bytes = cfg.job.disk_bytes
        if cfg.job.gpu_count > 0:
            req.resources.device.gpu.variant = cfg.job.gpu_variant or "auto"
            req.resources.device.gpu.count = cfg.job.gpu_count
        if cfg.job.group_by:
            req.coscheduling.group_by = cfg.job.group_by
        req.priority = cfg.job.priority_band
        reqs.append(req)
    return reqs


def _state_name(state: int) -> str:
    return job_pb2.TaskState.Name(state).replace("TASK_STATE_", "")


def run_gang(provider: K8sTaskProvider, cfg: SmokeConfig, *, timeout_seconds: int, poll_interval: float) -> bool:
    """Apply the gang, poll to completion. Returns True iff all tasks SUCCEEDED."""
    reqs = _build_requests(cfg)
    entries = [RunningTaskEntry(task_id=JobName.from_wire(r.task_id), attempt_id=0, coscheduled=True) for r in reqs]

    logger.info(
        "submitting gang %r: %d tasks, %s, group_by=%r, band=%s",
        cfg.job.name,
        cfg.job.replicas,
        f"{cfg.job.gpu_count}xGPU" if cfg.job.gpu_count else "cpu-only",
        cfg.job.group_by,
        _state_name(cfg.job.priority_band) if cfg.job.priority_band else "UNSPECIFIED",
    )
    # First sync applies the pods (Kueue gates them until the group is admitted).
    provider.sync(DirectProviderBatch(tasks_to_run=reqs, running_tasks=[]))

    states: dict[str, int] = {r.task_id: job_pb2.TASK_STATE_PENDING for r in reqs}
    deadline = time.monotonic() + timeout_seconds
    announced_admission = False
    last_line = ""
    while time.monotonic() < deadline:
        result = provider.sync(DirectProviderBatch(tasks_to_run=[], running_tasks=entries))
        for upd in result.updates:
            states[upd.task_id.to_wire()] = upd.new_state

        running = sum(1 for s in states.values() if s == job_pb2.TASK_STATE_RUNNING)
        if running and not announced_admission:
            logger.info("Kueue admitted the gang — %d/%d pods Running", running, cfg.job.replicas)
            announced_admission = True

        line = " ".join(f"{tid.rsplit('/', 1)[-1]}={_state_name(s)}" for tid, s in sorted(states.items()))
        if line != last_line:
            logger.info("states: %s", line)
            last_line = line

        if all(s in _TERMINAL_OK for s in states.values()):
            logger.info("gang SUCCEEDED (all %d tasks)", cfg.job.replicas)
            return True
        bad = {tid: s for tid, s in states.items() if s in _TERMINAL_BAD}
        if bad:
            logger.error("gang FAILED: %s", {t: _state_name(s) for t, s in bad.items()})
            return False
        time.sleep(poll_interval)

    logger.error("gang TIMED OUT after %ds; last states: %s", timeout_seconds, last_line)
    return False


def teardown_gang_pods(provider: K8sTaskProvider) -> None:
    """Delete the gang's pods + Kueue Workloads via the provider's stray-pod path."""
    # Empty desired set => every managed pod is stray and gets deleted, and its
    # backing Kueue Workload released (K8sTaskProvider._delete_stray_pods).
    provider.sync(DirectProviderBatch(tasks_to_run=[], running_tasks=[]))


def _print_pod_groups(kubectl: Kubectl, namespace: str) -> None:
    out = kubectl.get(
        "pods",
        "-n",
        namespace,
        "-l",
        _KUEUE_POD_GROUP_NAME,
        "-L",
        _KUEUE_POD_GROUP_NAME,
        "-L",
        "kueue.x-k8s.io/queue-name",
    )
    if out.strip():
        logger.info("gang pods:\n%s", out.rstrip())


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
class KindTarget:
    """Drives the SAME production code the CoreWeave path does, on a local kind cluster.

    The only kind-specific steps are creating the cluster and stamping CoreWeave
    topology/flavor labels on the workers (kind has none). Everything else is the
    real thing: scripts/install_kueue.py (upstream variant) installs the operator +
    Topology CRs + ResourceFlavor + ClusterQueue; K8sControllerProvider.ensure_kueue_queues
    reconciles the LocalQueue; config.make_provider builds the task provider.
    """

    def __init__(self, cfg: SmokeConfig) -> None:
        self.cfg = cfg
        self.kubectl = Kubectl(cfg.kubeconfig_path)
        self._iris_config_cache: config_pb2.IrisClusterConfig | None = None

    def setup(self) -> None:
        cfg = self.cfg
        name = cfg.kind.cluster_name
        existing = _run(["kind", "get", "clusters"], check=False, quiet=True).split()
        if name in existing:
            logger.info("kind cluster %r already exists — reusing it", name)
        else:
            kind_cfg = "kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nnodes:\n  - role: control-plane\n"
            kind_cfg += "".join("  - role: worker\n" for _ in range(cfg.kind.workers))
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                f.write(kind_cfg)
                kind_cfg_path = f.name
            _run(["kind", "create", "cluster", "--name", name, "--config", kind_cfg_path])
        _run(["kind", "export", "kubeconfig", "--name", name, "--kubeconfig", cfg.kubeconfig_path])

        self._label_nodes()
        if cfg.kueue.install:
            # Reuse the production install script (upstream variant) so the kind run
            # exercises the SAME operator Configuration + Topology CRs + ResourceFlavor
            # + ClusterQueue that install_kueue.py installs on CoreWeave (cks variant).
            install_kueue.run_install(
                variant=install_kueue.VARIANT_UPSTREAM,
                kubeconfig=cfg.kubeconfig_path,
                chart_version=cfg.kueue.version.lstrip("v"),
                with_queues=True,
                cluster_queue="iris-cq",
                cq_cpu=cfg.kueue.cluster_queue_cpu,
                cq_memory=cfg.kueue.cluster_queue_memory,
                cq_gpu="0",
                apply=True,
            )
        self.kubectl.apply(f"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: {cfg.namespace}\n")
        # The namespaced LocalQueue is the controller's responsibility — drive that
        # real path rather than stamping it here.
        self._ensure_local_queue()
        logger.info(
            "kind cluster ready: namespace=%s localQueue=%s", cfg.namespace, local_queue_name(_KIND_LABEL_PREFIX)
        )

    def _label_nodes(self) -> None:
        """Stamp CoreWeave topology + flavor + Iris-managed labels on the kind workers.

        Puts every worker in one synthetic leafgroup so the cw-ib ResourceFlavor
        selects them (flavor=infiniband) and TAS can satisfy a leafgroup-preferred
        podset within a single domain. The iris-managed label matches the node
        selector the provider stamps on pods (managed_label, derived from
        label_prefix) — CoreWeave NodePools carry it; on kind we add it by hand.
        """
        node_labels = dict(_KIND_NODE_TOPOLOGY_LABELS)
        node_labels[Labels(_KIND_LABEL_PREFIX).iris_managed] = "true"
        out = self.kubectl.get("nodes", "-l", "!node-role.kubernetes.io/control-plane", "-o", "name")
        nodes = [n for n in out.split() if n]
        label_args = [f"{key}={value}" for key, value in node_labels.items()]
        for node in nodes:
            self.kubectl.raw("label", node, *label_args, "--overwrite")
        logger.info("labeled %d kind worker node(s): %s", len(nodes), label_args)

    def _iris_config(self) -> config_pb2.IrisClusterConfig:
        """Build the IrisClusterConfig the provider + controller bootup consume.

        Kueue is enabled by kueue.cluster_queue (the admin ClusterQueue install_kueue.py
        created). The provider derives local_queue from label_prefix and the topology
        mapping from _CW_DEFAULT_TOPOLOGIES, so group_by=pool stamps the leafgroup
        podset-preferred-topology annotation.
        """
        if self._iris_config_cache is None:
            cfg = self.cfg
            proto = config_pb2.IrisClusterConfig()
            proto.platform.label_prefix = _KIND_LABEL_PREFIX
            kp = proto.kubernetes_provider
            kp.namespace = cfg.namespace
            kp.default_image = cfg.image
            kp.kubeconfig = cfg.kubeconfig_path
            kp.kueue.cluster_queue = "iris-cq"
            self._iris_config_cache = proto
        return self._iris_config_cache

    def _ensure_local_queue(self) -> None:
        cw = config_pb2.CoreweavePlatformConfig(namespace=self.cfg.namespace, kubeconfig_path=self.cfg.kubeconfig_path)
        controller = K8sControllerProvider(
            config=cw,
            label_prefix=_KIND_LABEL_PREFIX,
            kubectl=CloudK8sService(namespace=self.cfg.namespace, kubeconfig_path=self.cfg.kubeconfig_path),
        )
        controller.ensure_kueue_queues(self._iris_config())

    def make_provider(self) -> K8sTaskProvider:
        provider = build_task_provider(self._iris_config())
        assert isinstance(provider, K8sTaskProvider)
        return provider

    def teardown(self) -> None:
        _run(["kind", "delete", "cluster", "--name", self.cfg.kind.cluster_name], check=False)


class CoreweaveTarget:
    """Provision a dedicated H100 NodePool on a real CKS cluster and run one gang.

    Mirrors KindTarget: one IrisClusterConfig drives the production paths
    (ensure_nodepools, ensure_kueue_queues, config.make_provider). Kueue's operator
    and the cluster-scoped ClusterQueue/ResourceFlavor/Topologies are assumed
    admin-provisioned (scripts/install_kueue.py --variant coreweave --with-queues);
    this target creates only its own namespaced LocalQueue and NodePool, and tears
    the NodePool down (by exact name) so CKS stops billing for the H100s.
    """

    def __init__(self, cfg: SmokeConfig) -> None:
        self.cfg = cfg
        self.kubectl = Kubectl(cfg.kubeconfig_path)
        self._controller: K8sControllerProvider | None = None
        self._iris_config_cache: config_pb2.IrisClusterConfig | None = None

    def _controller_provider(self) -> K8sControllerProvider:
        if self._controller is None:
            cw = config_pb2.CoreweavePlatformConfig(
                region=self.cfg.coreweave.region,
                namespace=self.cfg.namespace,
                kubeconfig_path=self.cfg.kubeconfig_path,
            )
            self._controller = K8sControllerProvider(
                config=cw,
                label_prefix=_CW_LABEL_PREFIX,
                kubectl=CloudK8sService(namespace=self.cfg.namespace, kubeconfig_path=self.cfg.kubeconfig_path),
            )
        return self._controller

    def _iris_config(self) -> config_pb2.IrisClusterConfig:
        """One IrisClusterConfig for ensure_nodepools + ensure_kueue_queues + make_provider.

        Kueue is enabled by kueue.cluster_queue (the admin ClusterQueue); the
        provider derives local_queue from label_prefix and the topology mapping from
        _CW_DEFAULT_TOPOLOGIES, so group_by=pool stamps the leafgroup
        podset-preferred-topology and the managed-node selector pins pods to this
        smoke's NodePool.
        """
        if self._iris_config_cache is None:
            cfg = self.cfg
            sg = config_pb2.ScaleGroupConfig(
                num_vms=1,
                max_slices=cfg.coreweave.nodes,
                buffer_slices=cfg.coreweave.nodes,
                resources=config_pb2.ScaleGroupResources(
                    cpu_millicores=cfg.job.cpu_millicores,
                    memory_bytes=cfg.job.memory_bytes,
                    device_type=config_pb2.ACCELERATOR_TYPE_GPU,
                    device_variant=cfg.job.gpu_variant,
                    device_count=cfg.job.gpu_count,
                ),
                slice_template=config_pb2.SliceConfig(
                    num_vms=1,
                    coreweave=config_pb2.CoreweaveSliceConfig(
                        region=cfg.coreweave.region,
                        instance_type=cfg.coreweave.instance_type,
                        gpu_class=cfg.job.gpu_variant,
                        infiniband=True,
                    ),
                ),
            )
            cluster = config_pb2.IrisClusterConfig()
            cluster.platform.label_prefix = _CW_LABEL_PREFIX
            cluster.platform.coreweave.region = cfg.coreweave.region
            cluster.platform.coreweave.namespace = cfg.namespace
            cluster.platform.coreweave.kubeconfig_path = cfg.kubeconfig_path
            cluster.scale_groups[cfg.coreweave.scale_group].CopyFrom(sg)
            kp = cluster.kubernetes_provider
            kp.namespace = cfg.namespace
            kp.default_image = cfg.image
            kp.kubeconfig = cfg.kubeconfig_path
            kp.host_network = True
            kp.kueue.cluster_queue = _CW_CLUSTER_QUEUE
            self._iris_config_cache = cluster
        return self._iris_config_cache

    def _assert_namespace_unowned(self) -> None:
        """Refuse to run if another Iris controller already owns the target namespace.

        The standalone K8sTaskProvider reconciles the WHOLE namespace — it deletes
        every iris.managed pod not in the batch — so it must run in a namespace it
        exclusively owns. Bail out (before any mutation) if we find a live
        iris-controller Deployment or pre-existing iris.managed task pods, which is
        what pointing at a shared cluster namespace like iris-ci looks like. A
        not-yet-created namespace lists empty and passes.
        """
        ns = self.cfg.namespace
        controllers = self.kubectl.get("deploy", "-n", ns, "-l", "app=iris-controller", "-o", "name").split()
        managed = self.kubectl.get("pods", "-n", ns, "-l", "iris.managed=true", "-o", "name").split()
        if controllers or managed:
            raise RuntimeError(
                f"refusing to run: namespace {ns!r} is already owned by an Iris controller "
                f"(iris-controller deployments={len(controllers)}, iris.managed pods={len(managed)}). "
                "The smoke reconciles the whole namespace and would delete those pods. "
                "Point it at a dedicated namespace it exclusively owns."
            )

    def setup(self) -> None:
        cfg = self.cfg
        self._assert_namespace_unowned()
        controller = self._controller_provider()
        config = self._iris_config()
        controller.ensure_rbac()
        if cfg.coreweave.ensure_nodepools:
            logger.info(
                "ensuring NodePool %r (target %d nodes) — CKS will provision %s",
                controller._nodepool_name(cfg.coreweave.scale_group),
                cfg.coreweave.nodes,
                cfg.coreweave.instance_type,
            )
            controller.ensure_nodepools(config)
        # The namespaced LocalQueue is the controller's responsibility; drive that
        # real path (binds iris-ci -> the admin ClusterQueue).
        controller.ensure_kueue_queues(config)
        logger.info(
            "CoreWeave target ready: namespace=%s localQueue=%s clusterQueue=%s",
            cfg.namespace,
            local_queue_name(_CW_LABEL_PREFIX),
            _CW_CLUSTER_QUEUE,
        )

    def make_provider(self) -> K8sTaskProvider:
        provider = build_task_provider(self._iris_config())
        assert isinstance(provider, K8sTaskProvider)
        return provider

    def teardown(self) -> None:
        # Pods + Workloads are released by teardown_gang_pods(); here we drop the
        # NodePool(s) we created so CKS stops billing for the H100s. Delete by EXACT
        # name (not a label selector) — surgical on the shared cluster, and immune to
        # the label-key drift that a `-l` selector is prone to.
        if not self.cfg.coreweave.delete_nodepools_on_teardown:
            logger.warning("delete_nodepools_on_teardown=false: leaving NodePool up — it is STILL BILLING")
            return
        controller = self._controller_provider()
        for scale_group in self._iris_config().scale_groups:
            pool = controller._nodepool_name(scale_group)
            logger.info("deleting NodePool %s (CKS deprovisions asynchronously)", pool)
            self.kubectl.raw(
                "delete", "nodepools.compute.coreweave.com", pool, "--ignore-not-found", "--wait=false", check=False
            )


def make_target(cfg: SmokeConfig):
    if cfg.target == "kind":
        return KindTarget(cfg)
    if cfg.target == "coreweave":
        return CoreweaveTarget(cfg)
    raise ValueError(f"unknown target {cfg.target!r} (expected 'kind' or 'coreweave')")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="smoke config YAML")
    ap.add_argument("--keep", action="store_true", help="skip teardown (leave the cluster running)")
    ap.add_argument("--timeout", type=int, default=600, help="gang completion timeout (seconds)")
    ap.add_argument("--poll-interval", type=float, default=3.0, help="provider sync poll interval (seconds)")
    ap.add_argument(
        "--i-understand-the-cost", action="store_true", help="required for target=coreweave (provisions paid GPU nodes)"
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.target == "coreweave" and not args.i_understand_the_cost:
        logger.error("target=coreweave provisions paid GPU nodes; pass --i-understand-the-cost to proceed")
        return 2

    target = make_target(cfg)
    provider = None
    ok = False
    try:
        target.setup()
        provider = target.make_provider()
        ok = run_gang(provider, cfg, timeout_seconds=args.timeout, poll_interval=args.poll_interval)
        _print_pod_groups(target.kubectl, cfg.namespace)
    finally:
        if provider is not None:
            try:
                teardown_gang_pods(provider)
                provider.close()
            except Exception as e:
                logger.warning("gang pod teardown failed: %s", e)
        if args.keep:
            logger.info("--keep set: leaving cluster up (kubeconfig: %s)", cfg.kubeconfig_path)
        else:
            target.teardown()

    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
