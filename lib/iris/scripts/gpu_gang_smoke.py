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

import yaml
from iris.cluster.controller.direct_provider import DirectProviderBatch
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.providers.k8s.service import CloudK8sService
from iris.cluster.providers.k8s.tasks import (
    _KUEUE_POD_GROUP_NAME,
    K8sTaskProvider,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2

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

    def apply_url(self, url: str, *, server_side: bool = True) -> None:
        cmd = [*self._base, "apply"]
        if server_side:
            cmd += ["--server-side"]
        cmd += ["-f", url]
        _run(cmd)

    def wait(self, *args: str) -> None:
        _run([*self._base, "wait", *args], check=False)

    def rollout_restart(self, target: str, namespace: str) -> None:
        _run([*self._base, "rollout", "restart", target, "-n", namespace], check=False)

    def get(self, *args: str) -> str:
        return _run([*self._base, "get", *args], check=False, quiet=True)

    def raw(self, *args: str, check: bool = True) -> str:
        return _run([*self._base, *args], check=check)


# ---------------------------------------------------------------------------
# Kueue manifests
# ---------------------------------------------------------------------------
def _kueue_pod_integration_configmap() -> str:
    """ConfigMap that enables Kueue's plain-Pod integration.

    Kueue does not manage bare Pods unless 'pod' is in integrations.frameworks.
    The namespaceSelector excludes system namespaces; everything else (incl.
    our job namespace) is in scope. Requires a controller rollout to take effect.
    """
    config = {
        "apiVersion": "config.kueue.x-k8s.io/v1beta1",
        "kind": "Configuration",
        "health": {"healthProbeBindAddress": ":8081"},
        "metrics": {"bindAddress": ":8443"},
        "webhook": {"port": 9443},
        "manageJobsWithoutQueueName": False,
        "integrations": {
            "frameworks": ["batch/job", "pod"],
            "podOptions": {
                "namespaceSelector": {
                    "matchExpressions": [
                        {
                            "key": "kubernetes.io/metadata.name",
                            "operator": "NotIn",
                            "values": ["kube-system", "kueue-system"],
                        }
                    ]
                }
            },
        },
    }
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "kueue-manager-config", "namespace": "kueue-system"},
        "data": {"controller_manager_config.yaml": yaml.safe_dump(config, sort_keys=False)},
    }
    return yaml.safe_dump(cm, sort_keys=False)


def _kueue_queues_manifest(cfg: SmokeConfig) -> str:
    docs = [
        textwrap.dedent(
            """
            apiVersion: kueue.x-k8s.io/v1beta1
            kind: ResourceFlavor
            metadata:
              name: default-flavor
            """
        ).strip(),
        textwrap.dedent(
            f"""
            apiVersion: kueue.x-k8s.io/v1beta1
            kind: ClusterQueue
            metadata:
              name: iris-cq
            spec:
              namespaceSelector: {{}}
              resourceGroups:
                - coveredResources: ["cpu", "memory"]
                  flavors:
                    - name: default-flavor
                      resources:
                        - name: cpu
                          nominalQuota: "{cfg.kueue.cluster_queue_cpu}"
                        - name: memory
                          nominalQuota: "{cfg.kueue.cluster_queue_memory}"
            """
        ).strip(),
        textwrap.dedent(
            f"""
            apiVersion: kueue.x-k8s.io/v1beta1
            kind: LocalQueue
            metadata:
              name: {cfg.local_queue}
              namespace: {cfg.namespace}
            spec:
              clusterQueue: iris-cq
            """
        ).strip(),
    ]
    return "\n---\n".join(docs)


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
        "iris.managed=true",
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
    def __init__(self, cfg: SmokeConfig) -> None:
        self.cfg = cfg
        self.kubectl = Kubectl(cfg.kubeconfig_path)

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

        if cfg.kueue.install:
            self._install_kueue()
        self.kubectl.apply(f"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: {cfg.namespace}\n")
        self.kubectl.apply(_kueue_queues_manifest(cfg))
        logger.info("kind cluster ready: namespace=%s localQueue=%s", cfg.namespace, cfg.local_queue)

    def _install_kueue(self) -> None:
        ver = self.cfg.kueue.version
        url = f"https://github.com/kubernetes-sigs/kueue/releases/download/{ver}/manifests.yaml"
        logger.info("installing Kueue %s", ver)
        self.kubectl.apply_url(url, server_side=True)
        self.kubectl.wait(
            "--for=condition=Available",
            "deploy/kueue-controller-manager",
            "-n",
            "kueue-system",
            "--timeout=300s",
        )
        # Enable the plain-Pod integration, then restart the controller to load it.
        logger.info("enabling Kueue plain-Pod integration")
        self.kubectl.apply(_kueue_pod_integration_configmap())
        self.kubectl.rollout_restart("deploy/kueue-controller-manager", "kueue-system")
        self.kubectl.wait(
            "--for=condition=Available",
            "deploy/kueue-controller-manager",
            "-n",
            "kueue-system",
            "--timeout=300s",
        )
        # Webhook needs a moment to start serving after the restart.
        time.sleep(10)

    def make_provider(self) -> K8sTaskProvider:
        cfg = self.cfg
        kubectl = CloudK8sService(namespace=cfg.namespace, kubeconfig_path=cfg.kubeconfig_path)
        return K8sTaskProvider(
            kubectl=kubectl,
            namespace=cfg.namespace,
            default_image=cfg.image,
            local_queue=cfg.local_queue,
            controller_address=None,
        )

    def teardown(self) -> None:
        _run(["kind", "delete", "cluster", "--name", self.cfg.kind.cluster_name], check=False)


class CoreweaveTarget:
    def __init__(self, cfg: SmokeConfig) -> None:
        self.cfg = cfg
        self.kubectl = Kubectl(cfg.kubeconfig_path)
        self._controller = None

    def _controller_provider(self):
        if self._controller is None:
            from iris.cluster.providers.k8s.controller import K8sControllerProvider
            from iris.rpc import config_pb2

            cw = config_pb2.CoreweavePlatformConfig(
                region=self.cfg.coreweave.region,
                namespace=self.cfg.namespace,
                kubeconfig_path=self.cfg.kubeconfig_path,
            )
            self._controller = K8sControllerProvider(config=cw, label_prefix="iris")
        return self._controller

    def _cluster_config(self):
        """Build the minimal IrisClusterConfig ensure_nodepools needs."""
        from iris.rpc import config_pb2

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
        cluster.platform.coreweave.region = cfg.coreweave.region
        cluster.platform.coreweave.namespace = cfg.namespace
        cluster.scale_groups[cfg.coreweave.scale_group].CopyFrom(sg)
        return cluster

    def setup(self) -> None:
        cfg = self.cfg
        controller = self._controller_provider()
        controller.ensure_rbac()
        if cfg.coreweave.ensure_nodepools:
            logger.info(
                "ensuring NodePool %r (target %d nodes) — CKS will provision H100s",
                cfg.coreweave.scale_group,
                cfg.coreweave.nodes,
            )
            controller.ensure_nodepools(self._cluster_config())
        logger.info(
            "CoreWeave target ready: namespace=%s localQueue=%s (Kueue assumed admin-provisioned)",
            cfg.namespace,
            cfg.local_queue,
        )

    def make_provider(self) -> K8sTaskProvider:
        cfg = self.cfg
        kubectl = CloudK8sService(namespace=cfg.namespace, kubeconfig_path=cfg.kubeconfig_path)
        return K8sTaskProvider(
            kubectl=kubectl,
            namespace=cfg.namespace,
            default_image=cfg.image,
            local_queue=cfg.local_queue,
            host_network=True,
            controller_address=None,
        )

    def teardown(self) -> None:
        # Pods + Workloads are released by teardown_gang_pods(); here we drop the
        # NodePool so CKS stops billing for the H100s.
        if self.cfg.coreweave.delete_nodepools_on_teardown:
            self.kubectl.raw(
                "delete",
                "nodepools.compute.coreweave.com",
                "-l",
                "iris-scale-group",
                "--ignore-not-found",
                check=False,
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
