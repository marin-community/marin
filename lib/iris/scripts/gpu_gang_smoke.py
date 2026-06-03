#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up a real Iris controller, submit one coscheduled GPU gang through it, tear it down.

End-to-end smoke for Iris K8s-direct-provider **gang admission** via Kueue. Unlike a
bare provider test, this drives the *normal job path*: a live controller (the
``K8sTaskProvider`` runs inside it) admits a multi-replica coscheduled job through
Kueue, delivers the workspace bundle, runs ``uv sync --extra``, and the workload uses
``iris.runtime.initialize_jax`` (controller endpoint registry) to join one JAX mesh.

The workload (``scripts/gang_jax_smoke_workload.py``) trains a small causal transformer
data-parallel across the whole gang; its gradient all-reduce is the inter-host
NCCL/IB exercise.

Two targets, one code path:

  * ``kind``      — local cluster, **CPU-shaped** gang (no GPUs). Validates the entire
                    software path (controller boot, Kueue gang admission, bundle +
                    ``uv sync``, registry ``initialize_jax``, the model on CPU). The
                    controller image is built from this branch and ``kind load``ed; the
                    CoreWeave-only NodePool step is skipped.
  * ``coreweave`` — real H100 gang on CKS. Builds+pushes the controller image, provisions
                    a dedicated NodePool, runs the gang on 24 H100s. Costs money; gated
                    behind ``--i-understand-the-cost`` and a teardown-enabled check.

The task image is branch-agnostic (gang code lives in the controller), so it stays the
public ``iris-task:latest``; only the controller image is built from this branch.

Usage:
    cd lib/iris
    uv run --group dev python scripts/gpu_gang_smoke.py \
        --config config/kind-controller-gpu-smoke.yaml --target kind
    uv run --group dev python scripts/gpu_gang_smoke.py \
        --config config/coreweave-controller-gpu-smoke.yaml --target coreweave \
        --i-understand-the-cost
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import install_kueue
from iris.client import IrisClient
from iris.cluster.config import load_config
from iris.cluster.providers.k8s.controller import K8sControllerProvider, _build_controller_deployment
from iris.cluster.providers.k8s.service import CloudK8sService
from iris.cluster.providers.types import Labels
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import config_pb2, job_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("gpu_gang_smoke")

# Path (within the job bundle / repo root) of the multi-host JAX workload module.
_WORKLOAD = "lib/iris/scripts/gang_jax_smoke_workload.py"
# Coscheduling group_by → (by provider default) the leafgroup preferred-topology.
_GROUP_BY = "pool"
# CoreWeave topology/flavor labels stamped on kind workers so the cw-ib ResourceFlavor
# selects them (flavor=infiniband) and TAS can place the podset in one leafgroup.
_KIND_NODE_LABELS = {
    "backend.coreweave.cloud/flavor": "infiniband",
    "backend.coreweave.cloud/fabric": "fabric-0",
    "backend.coreweave.cloud/superpod": "superpod-0",
    "backend.coreweave.cloud/leafgroup": "leafgroup-0",
}


def repo_root() -> Path:
    # lib/iris/scripts/gpu_gang_smoke.py -> repo root.
    return Path(__file__).resolve().parents[3]


def _run(cmd: list[str], *, check: bool = True, quiet: bool = False, cwd: str | None = None) -> str:
    if not quiet:
        logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0 and check:
        logger.error(
            "command failed (%d): %s\nstdout:\n%s\nstderr:\n%s", proc.returncode, " ".join(cmd), proc.stdout, proc.stderr
        )
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def _wait_port(host: str, port: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
    raise RuntimeError(f"port {host}:{port} not reachable within {timeout}s")


# ---------------------------------------------------------------------------
# Controller lifecycle (shared)
# ---------------------------------------------------------------------------
class ControllerTarget:
    """Shared controller bring-up/teardown over the K8s direct provider."""

    def __init__(self, cfg: config_pb2.IrisClusterConfig, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.namespace = cfg.kubernetes_provider.namespace
        self.label_prefix = cfg.platform.label_prefix
        self.kubeconfig = str(Path(cfg.platform.coreweave.kubeconfig_path).expanduser())
        # k8s clients are built lazily by _connect() AFTER setup() finalizes the
        # kubeconfig — kind assigns a fresh API-server port on each cluster create,
        # so a client built against a stale kubeconfig would hit a dead port.
        self.kubectl: CloudK8sService | None = None
        self.controller: K8sControllerProvider | None = None
        self._tunnel: subprocess.Popen | None = None

    def _connect(self) -> None:
        """(Re)build the k8s clients against the current kubeconfig."""
        self.kubectl = CloudK8sService(namespace=self.namespace, kubeconfig_path=self.kubeconfig)
        self.controller = K8sControllerProvider(
            config=config_pb2.CoreweavePlatformConfig(
                region=self.cfg.platform.coreweave.region, namespace=self.namespace, kubeconfig_path=self.kubeconfig
            ),
            label_prefix=self.label_prefix,
            kubectl=self.kubectl,
        )

    def _kc(self, *args: str) -> list[str]:
        return ["kubectl", "--kubeconfig", self.kubeconfig, *args]

    def _ensure_namespace(self) -> None:
        _run(self._kc("create", "namespace", self.namespace), check=False, quiet=True)

    def deploy_controller(self, *, skip_nodepools: bool, local_image: bool, node_selector: dict[str, str]) -> str:
        """Reconcile RBAC, ConfigMap, (NodePools), LocalQueue, Deployment, Service.

        On kind we skip ensure_nodepools (no CoreWeave NodePool CRD) and use a
        kind-loaded image (imagePullPolicy IfNotPresent, no scale-group nodeSelector).
        """
        cfg = self.cfg
        c = self.controller
        port = cfg.controller.coreweave.port or 10000
        svc = cfg.controller.coreweave.service_name or "iris-controller-svc"

        c.ensure_rbac()
        cm = c._config_json_for_configmap(cfg)
        self.kubectl.apply_json(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "iris-cluster-config", "namespace": self.namespace},
                "data": {"config.json": cm},
            }
        )
        if not skip_nodepools:
            c.ensure_nodepools(cfg)
        c.ensure_kueue_queues(cfg)

        # fresh=False: a controller restart must NOT wipe the SQLite DB, or an
        # in-flight submitted job vanishes (client then sees the job NOT_FOUND).
        dep = _build_controller_deployment(
            namespace=self.namespace,
            image=cfg.controller.image,
            port=port,
            node_selector=node_selector,
            s3_env_vars=[],
            fresh=False,
        )
        if local_image:
            dep["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
        self.kubectl.apply_json(dep)
        self.kubectl.apply_json(
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": svc, "namespace": self.namespace},
                "spec": {
                    "selector": {"app": "iris-controller"},
                    "ports": [{"port": port, "targetPort": port}],
                    "type": "ClusterIP",
                },
            }
        )
        logger.info("controller manifests applied; waiting for deployment ready")
        c.wait_for_deployment_ready()
        addr = c.discover_controller(cfg.controller)
        logger.info("controller ready (in-cluster: %s)", addr)
        return addr

    def open_tunnel(self, local_port: int = 10000) -> str:
        port = self.cfg.controller.coreweave.port or 10000
        svc = self.cfg.controller.coreweave.service_name or "iris-controller-svc"
        self._tunnel = subprocess.Popen(
            self._kc("port-forward", "-n", self.namespace, f"svc/{svc}", f"{local_port}:{port}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_port("localhost", local_port, timeout=60)
        url = f"http://localhost:{local_port}"
        logger.info("controller tunnel open: %s", url)
        return url

    def close_tunnel(self) -> None:
        if self._tunnel is not None:
            self._tunnel.terminate()
            self._tunnel = None

    def stop_controller(self) -> None:
        self.controller.stop_controller(self.cfg)


# ---------------------------------------------------------------------------
# kind
# ---------------------------------------------------------------------------
class KindTarget(ControllerTarget):
    def setup(self) -> None:
        cfg = self.cfg
        name = self.args.kind_cluster
        existing = _run(["kind", "get", "clusters"], check=False, quiet=True).split()
        if name not in existing:
            kind_cfg = "kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nnodes:\n  - role: control-plane\n"
            kind_cfg += "".join("  - role: worker\n" for _ in range(self.args.replicas))
            path = Path("/tmp/kind-iris-gpu-smoke.yaml")
            path.write_text(kind_cfg)
            _run(["kind", "create", "cluster", "--name", name, "--config", str(path)])
        _run(["kind", "export", "kubeconfig", "--name", name, "--kubeconfig", self.kubeconfig])
        self._connect()  # kubeconfig is now final (fresh API-server port); safe to build clients
        self._ensure_namespace()
        self._label_nodes()
        # Build the controller image from THIS branch and load it into kind.
        logger.info("building controller image %s (cached layers make repeats fast)", cfg.controller.image)
        _run(
            ["docker", "build", "-f", "lib/iris/Dockerfile", "--target", "controller", "-t", cfg.controller.image, "."],
            cwd=str(repo_root()),
        )
        _run(["kind", "load", "docker-image", cfg.controller.image, "--name", name])
        # Kueue operator + cluster-scoped ClusterQueue/ResourceFlavor/Topology.
        install_kueue.run_install(
            variant=install_kueue.VARIANT_UPSTREAM,
            kubeconfig=self.kubeconfig,
            chart_version="0.11.0",
            with_queues=True,
            cluster_queue=cfg.kubernetes_provider.kueue.cluster_queue,
            apply=True,
        )

    def _label_nodes(self) -> None:
        labels = dict(_KIND_NODE_LABELS)
        labels[Labels(self.label_prefix).iris_managed] = "true"
        out = _run(
            self._kc(
                "get",
                "nodes",
                "-l",
                "!node-role.kubernetes.io/control-plane",
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ),
            quiet=True,
        )
        nodes = out.split()
        for node in nodes:
            _run(self._kc("label", "node", node, "--overwrite", *[f"{k}={v}" for k, v in labels.items()]), quiet=True)
        logger.info("labeled %d kind worker node(s)", len(nodes))

    def deploy(self) -> str:
        return self.deploy_controller(skip_nodepools=True, local_image=True, node_selector={})

    def job_resources(self) -> tuple[ResourceSpec, list[str]]:
        # kind has no GPUs: CPU-shaped gang, jax CPU via the `cpu` extra.
        return ResourceSpec(cpu=2, memory="8GB", disk="30GB", device=None), ["cpu"]

    def teardown(self) -> None:
        self.close_tunnel()
        if not self.args.keep:
            _run(["kind", "delete", "cluster", "--name", self.args.kind_cluster], check=False)


# ---------------------------------------------------------------------------
# CoreWeave
# ---------------------------------------------------------------------------
class CoreweaveTarget(ControllerTarget):
    def setup(self) -> None:
        cfg = self.cfg
        self._connect()  # coreweave kubeconfig is static
        self._assert_namespace_unowned()
        self._ensure_namespace()
        logger.info("building+pushing controller image %s", cfg.controller.image)
        _run(
            ["docker", "build", "-f", "lib/iris/Dockerfile", "--target", "controller", "-t", cfg.controller.image, "."],
            cwd=str(repo_root()),
        )
        _run(["docker", "push", cfg.controller.image])
        # Kueue operator + iris-cq are admin-provisioned on the shared CKS cluster.

    def _assert_namespace_unowned(self) -> None:
        out = _run(
            self._kc("get", "deploy", "-n", self.namespace, "-l", "app=iris-controller", "-o", "name"),
            check=False,
            quiet=True,
        ).split()
        # Our own controller from a prior run is fine; refuse only a foreign namespace.
        if out and self.namespace in ("iris-ci", "iris"):
            raise RuntimeError(f"refusing to run in shared namespace {self.namespace!r}; use a dedicated namespace")

    def deploy(self) -> str:
        # Use the canonical production bring-up: it provisions the S3 credentials
        # Secret, ConfigMap, NodePools (CPU controller pool + H100 gang pool),
        # Kueue queues, Deployment (pinned to the controller's CPU scale group via
        # nodeSelector), Service, and PDB, then waits for readiness. The kind path
        # can't use this (no NodePool CRD, kind-loaded image), but on real CKS this
        # is exactly what `iris cluster up` runs.
        addr = self.controller.start_controller(self.cfg)
        self._set_target_nodes(self.args.replicas)
        return addr

    def _set_target_nodes(self, n: int) -> None:
        # Kueue TAS can't scale from zero (see project_kueue_tas_scale_from_zero):
        # provision the H100 nodes up front so admission can place the gang. Only
        # GPU pools — the controller's CPU pool keeps its own buffer size.
        patch = json.dumps({"spec": {"targetNodes": n}})
        for name, sg in self.cfg.scale_groups.items():
            if sg.resources.device_type != config_pb2.ACCELERATOR_TYPE_GPU:
                continue
            pool = self.controller._nodepool_name(name)
            logger.info("setting NodePool %s targetNodes=%d", pool, n)
            _run(self._kc("patch", "nodepools.compute.coreweave.com", pool, "--type=merge", "-p", patch))

    def job_resources(self) -> tuple[ResourceSpec, list[str]]:
        variant, count = self.args.gpu.split(":")
        return ResourceSpec(cpu=32, memory="512GB", disk="100GB", device=gpu_device(variant, int(count))), ["gpu"]

    def teardown(self) -> None:
        self.close_tunnel()
        if self.args.keep:
            logger.warning("--keep set: leaving controller + NodePool up — H100s are STILL BILLING")
            return
        if self.controller is None:
            return  # setup failed before connecting; nothing provisioned
        try:
            self.stop_controller()
        except Exception as e:
            logger.warning("stop_controller failed: %s", e)
        for sg in self.cfg.scale_groups:
            pool = self.controller._nodepool_name(sg)
            logger.info("deleting NodePool %s (async)", pool)
            _run(
                self._kc("delete", "nodepools.compute.coreweave.com", pool, "--ignore-not-found", "--wait=false"),
                check=False,
            )


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------
def submit_gang(controller_url: str, target: ControllerTarget, args: argparse.Namespace) -> bool:
    resources, extras = target.job_resources()
    env = EnvironmentSpec(
        extras=extras,
        env_vars={"GANG_SMOKE_STEPS": str(args.steps), "GANG_SMOKE_PDB": str(args.per_device_batch)},
    )
    client = IrisClient.remote(controller_url, workspace=repo_root(), timeout_ms=300_000)
    logger.info(
        "submitting gang %r: %d replicas, %s, group_by=%s, extras=%s",
        args.job_name,
        args.replicas,
        f"{args.gpu}" if args.gpu else "cpu-only",
        _GROUP_BY,
        extras,
    )
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", _WORKLOAD),
        name=args.job_name,
        resources=resources,
        environment=env,
        replicas=args.replicas,
        coscheduling=CoschedulingConfig(group_by=_GROUP_BY),
    )
    status = job.wait(timeout=args.timeout, poll_interval=10.0, stream_logs=True, raise_on_failure=False)
    state = job_pb2.JobState.Name(status.state)
    logger.info("job %s finished in state %s", args.job_name, state)
    return status.state == job_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Iris cluster config YAML (kind or coreweave)")
    ap.add_argument("--target", required=True, choices=["kind", "coreweave"])
    ap.add_argument("--replicas", type=int, default=3, help="gang size (tasks)")
    ap.add_argument("--gpu", default="", help="GPU request as VARIANT:COUNT (coreweave), e.g. H100:8")
    ap.add_argument("--steps", type=int, default=20, help="training steps")
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--job-name", default="gpu-gang-smoke")
    ap.add_argument("--timeout", type=int, default=1800, help="job completion timeout (seconds)")
    ap.add_argument("--kind-cluster", default="iris-gpu-smoke")
    ap.add_argument("--keep", action="store_true", help="skip teardown")
    ap.add_argument("--i-understand-the-cost", action="store_true", help="required for --target coreweave (paid H100s)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.target == "coreweave":
        if not args.i_understand_the_cost:
            logger.error("target=coreweave provisions paid H100s; pass --i-understand-the-cost")
            return 2
        if args.keep:
            # Cleanup must be guaranteed so we never strand billing H100s.
            logger.error("--keep is not allowed for target=coreweave: teardown must run to deprovision H100s")
            return 2
        if not args.gpu:
            args.gpu = "H100:8"

    target = KindTarget(cfg, args) if args.target == "kind" else CoreweaveTarget(cfg, args)

    ok = False
    try:
        target.setup()
        target.deploy()
        url = target.open_tunnel()
        ok = submit_gang(url, target, args)
    finally:
        if args.keep:
            logger.info("--keep set: leaving cluster up")
            target.close_tunnel()
        else:
            target.teardown()

    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
