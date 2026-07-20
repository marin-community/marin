#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth snapshot + diff, to verify a `pulumi up` (or any other cluster operation) didn't
touch running nodes, NodePools, pods, or the controller.

Captures four things:
  - NodePool object identity (metadata.uid) — a changed uid means replace, not update, i.e. the
    reserved bare-metal fleet was deprovisioned and recreated. The single most important check.
  - Node identity (name + creationTimestamp) — a changed value means nodes cycled.
  - Every pod's creation time, across all namespaces (name + namespace + creationTimestamp) — a
    reset age means something restarted it.
  - The Iris controller's reported status (Running/Healthy/Version/Workers) — a changed Version
    means something redeployed the controller, which `pulumi up` should never do.

First run (no existing snapshot file): captures ground truth and writes it as the baseline.
Subsequent runs: captures ground truth again and diffs against the file, printing any changes.
Exits non-zero if there's a diff, so this composes into a shell `&&` chain or a CI gate.

Usage:
    uv run infra/iac/scripts/verify_no_drift.py --cluster cw-us-west-04a
    # ... do something to the cluster (pulumi up, etc.) ...
    uv run infra/iac/scripts/verify_no_drift.py --cluster cw-us-west-04a

    # Explicit snapshot path, or start over:
    uv run infra/iac/scripts/verify_no_drift.py --cluster cw-us-west-04a --snapshot /tmp/before.json
    rm /tmp/cw-us-west-04a-ground-truth.json  # discard the baseline, next run starts fresh
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from iac.config import IAC_CLUSTER_CONFIG_DIR, load_iris_config
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.provenance import provenance_from_proto
from iris.rpc import controller_pb2, job_pb2
from rigging.config_discovery import resolve_cluster_config

DEFAULT_SNAPSHOT_DIR = Path("/tmp")


def _kubectl_json(kubeconfig: str, context: str, *args: str) -> dict:
    cmd = ["kubectl", "--kubeconfig", kubeconfig, "--context", context, *args, "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"`{' '.join(cmd)}` failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _nodepool_uids(kubectl: CloudK8sService) -> dict[str, str]:
    items = kubectl.list_json(K8sResource.NODE_POOLS)
    return {item["metadata"]["name"]: item["metadata"]["uid"] for item in items}


def _node_creation_timestamps(kubectl: CloudK8sService) -> dict[str, str]:
    items = kubectl.list_json(K8sResource.NODES)
    return {item["metadata"]["name"]: item["metadata"]["creationTimestamp"] for item in items}


def _pod_creation_timestamps(kubeconfig: str, context: str) -> dict[str, str]:
    # Cross-namespace ("-A"): CloudK8sService.list_json always scopes namespaced
    # resources to a single namespace, so it can't express this — kubectl directly.
    data = _kubectl_json(kubeconfig, context, "get", "pods", "-A")
    return {
        f"{item['metadata']['namespace']}/{item['metadata']['name']}": item["metadata"]["creationTimestamp"]
        for item in data["items"]
    }


def _controller_status(cluster: str) -> dict[str, object]:
    """The controller's reported identity, read straight from its RPC surface.

    A changed Version between snapshots means the controller was redeployed, which a
    `pulumi up` over the static substrate must never cause. Talks to the controller
    through the Iris client rather than a CLI subprocess: the client returns typed
    protos, where CLI stdout is not a stable contract. Resolves the controller from
    the reviewed in-tree config (IAC_CLUSTER_CONFIG_DIR), matching the rest of capture().
    """
    config_file = Path(resolve_cluster_config(cluster, dirs=(IAC_CLUSTER_CONFIG_DIR,)))
    with connect_controller(config_file=config_file) as endpoint:
        with rpc_client(endpoint.url, endpoint.credentials) as client:
            proc = client.get_process_status(job_pb2.GetProcessStatusRequest()).process_info
            workers = client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
    return {
        "version": str(provenance_from_proto(proc.provenance)),
        "workers_healthy": sum(1 for w in workers if w.healthy),
        "workers_total": len(workers),
    }


def capture(cluster: str, namespace: str, kubeconfig: str, context: str) -> dict:
    kubectl = CloudK8sService(namespace=namespace, kubeconfig_path=kubeconfig, context=context)
    return {
        "nodepool_uids": _nodepool_uids(kubectl),
        "node_creation_timestamps": _node_creation_timestamps(kubectl),
        "pod_creation_timestamps": _pod_creation_timestamps(kubeconfig, context),
        "controller_status": _controller_status(cluster),
    }


def diff(before: dict, after: dict) -> list[str]:
    lines = []
    for section in before:
        before_section, after_section = before[section], after.get(section, {})
        for key in sorted(set(before_section) | set(after_section)):
            before_value = before_section.get(key, "<absent>")
            after_value = after_section.get(key, "<absent>")
            if before_value != after_value:
                lines.append(f"{section}.{key}: {before_value!r} -> {after_value!r}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", required=True, help="Iris cluster name, e.g. cw-us-west-04a")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Snapshot file path [default: /tmp/<cluster>-ground-truth.json]",
    )
    args = parser.parse_args()

    iris_config = load_iris_config(args.cluster)
    platform_coreweave = iris_config.platform.coreweave
    if platform_coreweave is None or not platform_coreweave.kubeconfig_path:
        raise SystemExit(f"cluster {args.cluster!r} has no platform.coreweave.kubeconfig_path")
    kubeconfig = os.path.expanduser(platform_coreweave.kubeconfig_path)
    context = platform_coreweave.kube_context
    if not context:
        raise SystemExit(f"cluster {args.cluster!r} has no platform.coreweave.kube_context")

    snapshot_path = args.snapshot or DEFAULT_SNAPSHOT_DIR / f"{args.cluster}-ground-truth.json"
    current = capture(args.cluster, platform_coreweave.namespace or "iris", kubeconfig, context)

    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"No existing snapshot at {snapshot_path} — wrote baseline ground truth.")
        return

    before = json.loads(snapshot_path.read_text())
    changes = diff(before, current)
    if not changes:
        print(f"No drift detected against {snapshot_path}.")
        return

    print(f"DRIFT DETECTED against {snapshot_path}:")
    for line in changes:
        print(f"  {line}")
    sys.exit(1)


if __name__ == "__main__":
    main()
