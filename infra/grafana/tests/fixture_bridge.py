# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic HTTP fixture for rendering the provisioned dashboard in Grafana."""

import json
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from config import K8S_CLUSTERS

_NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)
_CW_K8S_CLUSTERS = tuple(target.name for target in K8S_CLUSTERS)
_CW_NODE_WITH_DEADLOCK = "cw-us-east-08a"
_LANES = (
    ("tpu-ferry", "TPU ferry", "marin", "training"),
    ("cw-gpu-ferry", "CW ferry", "marin", "training"),
    ("grug-multislice", "Grug", "marin", "training"),
    ("datakit-t1", "Data T1", "marin", "data"),
    ("datakit-t2", "Data T2", "marin", "data"),
    ("datakit-t3", "Data T3", "marin", "data"),
    ("evalchemy", "Evalchemy", "forks", "evaluation"),
    ("harbor", "Harbor", "forks", "evaluation"),
    ("marinskyrl", "SkyRL", "forks", "rl"),
    ("vllm-gpu", "vLLM GPU", "forks", "inference"),
    ("tpu-inference", "TPU infer", "forks", "inference"),
)


def _nightlies() -> list[dict]:
    rows = []
    for lane_order, (lane_id, label, group, subgroup) in enumerate(_LANES):
        for offset in range(7):
            date = (_NOW - timedelta(days=offset)).strftime("%Y-%m-%d")
            failed = (lane_order + offset) % 17 == 0
            slow = (lane_order * 3 + offset) % 13 == 0
            rows.append(
                {
                    "date": date,
                    "lane_id": lane_id,
                    "lane": label,
                    "label": label,
                    "group": group,
                    "subgroup": subgroup,
                    "state": "run",
                    "duration_state": "slow" if slow else "normal",
                    "duration_seconds": 1800 + lane_order * 137 + offset * 83,
                    "conclusion": "failure" if failed else "success",
                    "url": f"https://github.com/marin-community/marin/actions/runs/{lane_order}{offset}",
                    "workflow_url": "https://github.com/marin-community/marin/actions",
                    "healthy": not failed,
                    "due": True,
                    "source_error": "",
                    "lane_order": lane_order,
                }
            )
    return rows


def _builds() -> list[dict]:
    rows = []
    for index in range(60):
        state = "FAILURE" if index in (7, 21, 42) else "PENDING" if index < 2 else "SUCCESS"
        rows.append(
            {
                "oid": f"{index:040x}",
                "short_oid": f"{index:07x}",
                "headline": "compact infra dashboard" if index == 0 else f"main branch change {index}",
                "author": "marin-bot",
                "avatar_url": "",
                "state": state,
                "committed_at": round((_NOW - timedelta(minutes=index * 38)).timestamp() * 1000),
                "url": f"https://github.com/marin-community/marin/commit/{index:040x}",
                "success_rate": 0.947,
            }
        )
    return rows


def _wandb(chart: str) -> list[dict]:
    titles = {"train-loss": "Train cross-entropy loss", "paloma-macro-loss": "Paloma macro loss", "mfu": "MFU (%)"}
    rows = []
    for run_index, run in enumerate(("hero-12d8b6f0-dee637",)):
        for index in range(40):
            tokens = (index + 1) * 250_000_000_000
            if chart == "mfu":
                value = 0.43 + run_index * 0.025 + index * 0.0009
            else:
                value = 3.2 - index * 0.035 + run_index * 0.08
            rows.append(
                {
                    "chart": titles[chart],
                    "run": run,
                    "tokens": tokens,
                    "value": value,
                    "report_title": "535B-A23B 18T Token Hero Run + Scaling Ladder",
                    "report_url": (
                        "https://wandb.ai/marin-community/marin_moe/reports/"
                        "535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng"
                    ),
                }
            )
    return rows


def _wandb_history(params: dict[str, list[str]]) -> list[dict]:
    """A whole-run curve: the point of the panel is that it starts at step 0."""
    run = params.get("run", ["hero-preview"])[0]
    return [
        {
            "run": run,
            "project": "marin_moe",
            "run_url": f"https://wandb.ai/marin-community/marin_moe/runs/{run}",
            "step": step,
            "value": 3.4 - 1.9 * (step / 40_000) ** 0.35,
        }
        for step in range(0, 40_000, 200)
    ]


def _finelog(query: str) -> list[dict]:
    sql = parse_qs(query).get("sql", [""])[0]
    if 'FROM "iris.task"' in sql:
        return [
            {
                "cluster": "cw-us-east-02a",
                "task": task,
                "pod": pod,
                "cpu_millicores": cpu,
                "memory_bytes": memory,
                "sampled_at": round((_NOW - timedelta(seconds=20)).timestamp() * 1000),
            }
            for task, pod, cpu, memory in (
                ("/alice/llama/0", "llama-0", 29_000, 310_000_000_000),
                ("/alice/llama/1", "llama-1", 14_000, 180_000_000_000),
                ("/bob/eval/0", "eval-0", 9_500, 92_000_000_000),
                ("/carol/embed/0", "embed-0", 5_200, 48_000_000_000),
                ("/ops/loader/0", "loader-0", 1_800, 12_000_000_000),
            )
        ]
    if "gpu_memory_total_bytes" in sql and "ROW_NUMBER" in sql:
        sampled_at = round((_NOW - timedelta(seconds=15)).timestamp() * 1000)
        values = {
            "gpu-a": {
                "node_cpu_utilization_percent": 61,
                "node_memory_used_bytes": 760_000_000_000,
                "node_memory_total_bytes": 1_100_000_000_000,
                "gpu_utilization_percent": 88,
                "gpu_memory_used_bytes": 590_000_000_000,
                "gpu_memory_total_bytes": 640_000_000_000,
            },
            "gpu-b": {
                "node_cpu_utilization_percent": 44,
                "node_memory_used_bytes": 430_000_000_000,
                "node_memory_total_bytes": 1_100_000_000_000,
                "gpu_utilization_percent": 55,
                "gpu_memory_used_bytes": 350_000_000_000,
                "gpu_memory_total_bytes": 640_000_000_000,
            },
            "gpu-c": {
                "node_cpu_utilization_percent": 28,
                "node_memory_used_bytes": 190_000_000_000,
                "node_memory_total_bytes": 550_000_000_000,
                "gpu_utilization_percent": 36,
                "gpu_memory_used_bytes": 91_000_000_000,
                "gpu_memory_total_bytes": 320_000_000_000,
            },
        }
        return [
            {
                "cluster": "cw-us-east-02a",
                "node": node,
                "name": name,
                "value": value,
                "sampled_at": sampled_at,
            }
            for node, metrics in values.items()
            for name, value in metrics.items()
        ]
    return []


def _finelog_k8s_rows(path: str) -> list[dict] | None:
    if path == "/k8s/finelog":
        return [
            {
                "cluster": cluster,
                "namespace": "iris",
                "deployment": server,
                "pod": f"{server}-abc",
                "node": "cpu-node-1",
                "phase": "Running",
                "ready": True,
                "restarts": 0,
                "last_exit_code": None,
                "last_exit_reason": "",
                "cpu_request": "2",
                "cpu_limit": "8",
                "memory_request": "16Gi",
                "memory_limit": "32Gi",
                "startup_probe": True,
                "readiness_probe": True,
                "liveness_probe": True,
                "pvc": f"{server}-cache",
                "storage_class": "shared-vast",
                "storage_capacity": "250Gi",
                "image": "ghcr.io/marin-community/finelog@sha256:abc",
                "error_class": "",
                "error": "",
            }
            for cluster, server in (
                ("cw-us-east-02a", "finelog-cw-use02a"),
                ("cw-us-east-08a", "finelog-cw-use08a"),
                ("cw-rno2a", "finelog-cw-rno2a"),
            )
        ]
    if path == "/k8s/finelog_events":
        return [
            {
                "cluster": "cw-rno2a",
                "namespace": "iris",
                "object": "Pod/finelog-cw-rno2a-abc",
                "reason": "Unhealthy",
                "message": "Readiness probe failed",
                "count": 3,
                "last_seen": round(_NOW.timestamp() * 1000),
            }
        ]
    return None


def _rows(path: str, query: str) -> list[dict] | dict:
    finelog_rows = _finelog_k8s_rows(path)
    if finelog_rows is not None:
        return finelog_rows
    if path == "/github/nightlies":
        return _nightlies()
    if path == "/github/builds":
        return _builds()
    if path == "/github/ferries":
        return [
            {
                "group": group,
                "tier": tier,
                "conclusion": "success",
                "status": "completed",
                "sha": "abc1234",
                "started_at": round((_NOW - timedelta(hours=index + 1)).timestamp() * 1000),
                "duration_seconds": 720 + index * 80,
                "success_rate": 0.96,
                "actor": "marin-bot",
                "html_url": "https://github.com/marin-community/marin/actions",
            }
            for index, (group, tier) in enumerate(
                (
                    ("Canary ferry", ""),
                    ("CW ferry", ""),
                    ("Datakit ferry", "tier1"),
                    ("Datakit ferry", "tier2"),
                    ("Datakit ferry", "tier3"),
                )
            )
        ]
    if path == "/iris/marin/health":
        return [{"reachable": True, "up": 1, "latency_ms": 18}]
    if path == "/iris/marin/peers":
        return [
            {
                "peer": peer,
                "controller_address": f"https://iris-{peer}.oa.dev",
                "state": state,
                "last_contact_age_seconds": age,
                "value": int(state == "unreachable"),
            }
            for peer, state, age in (
                ("cw-us-east-02a", "reachable", 12),
                ("cw-us-east-08a", "unreachable", 10_800),
                ("cw-rno2a", "reachable", 8),
            )
        ]
    if path == "/iris/marin/workers":
        return [
            {
                "region": region,
                "healthy": healthy,
                "cpu_millicores": healthy * 96_000,
                "memory_bytes": healthy * 412_316_860_416,
                "tpu_chips": chips,
            }
            for region, healthy, chips in (("us-east5", 84, 512), ("us-central2", 51, 256), ("cw-us-east", 37, 0))
        ]
    if path == "/iris/marin/job_counts":
        return [
            {"bucket": "inflight", "state": "running", "count": 43},
            {"bucket": "last24h", "state": "succeeded", "count": 318},
            {"bucket": "last24h", "state": "failed", "count": 9},
        ]
    if path == "/iris/marin/jobs":
        return [{"job": job} for job in ("/alice/llama", "/bob/eval", "/carol/embed", "/ops/loader", "/dave/train")]
    if path == "/finelog/marin/fleet_health":
        return [
            {
                "cluster": cluster,
                "server": server,
                "role": role,
                "responsive": True,
                "ready": 1,
                "desired": 1,
                "latency_ms": 22 if role == "hub" else None,
                "error_class": "",
                "error": "",
            }
            for cluster, server, role in (
                ("marin", "finelog-marin", "hub"),
                ("cw-us-east-02a", "finelog-cw-use02a", "mirror"),
                ("cw-us-east-08a", "finelog-cw-use08a", "mirror"),
                ("cw-rno2a", "finelog-cw-rno2a", "mirror"),
            )
        ]
    if path == "/k8s/health":
        return [
            {"cluster": cluster, "reachable": True, "up": 1, "latency_ms": 31, "error_class": ""}
            for cluster in _CW_K8S_CLUSTERS
        ]
    if path == "/k8s/alerts/unreachable":
        return [{"cluster": cluster, "error_class": "none", "value": 0} for cluster in _CW_K8S_CLUSTERS]
    if path == "/k8s/overview":
        return [{"pending_pods": 1, "crashlooping_containers": 1}]
    if path == "/k8s/control_plane":
        return [
            {
                "cluster": cluster,
                "kind": "component",
                "component": component,
                "ready": 1,
                "desired": 1,
                "restarts": 0,
                "waiting_reason": "",
            }
            for cluster in _CW_K8S_CLUSTERS
            for component in ("iris/iris-controller", "kueue-system/kueue-controller-manager")
        ]
    if path == "/k8s/nodes":
        selected = parse_qs(query).get("cluster", [""])[0]
        rows = [
            {
                "cluster": cluster,
                "node": node_name,
                "instance_type": instance_type,
                "node_pool": "training",
                "gpu_model": gpu_model,
                "gpu_capacity": gpu_count,
                "gpu_allocatable": gpu_count,
                "cpu_allocatable": cpu,
                "memory_allocatable": memory,
                "ready": True,
                "unschedulable": cluster == _CW_NODE_WITH_DEADLOCK,
                "kernel_deadlock": cluster == _CW_NODE_WITH_DEADLOCK,
                "deadlock_reason": "CPUSoftLockup" if cluster == _CW_NODE_WITH_DEADLOCK else "",
                "cordon_reason": "KernelDeadlock,NLCCPendingExitProduction" if cluster == _CW_NODE_WITH_DEADLOCK else "",
                "pending_phase": "production-reboot" if cluster == _CW_NODE_WITH_DEADLOCK else "",
                "deadlock_message": "watchdog: CPU stuck" if cluster == _CW_NODE_WITH_DEADLOCK else "",
            }
            for cluster, node_name, instance_type, gpu_model, gpu_count, cpu, memory in (
                ("cw-us-east-02a", "gpu-a", "h100-8", "NVIDIA H100 80GB HBM3", 8, "96", "1Ti"),
                ("cw-us-east-02a", "gpu-b", "h100-8", "NVIDIA H100 80GB HBM3", 8, "96", "1Ti"),
                ("cw-us-east-02a", "gpu-c", "h100-4", "NVIDIA H100 80GB HBM3", 4, "48", "512Gi"),
                ("cw-us-east-08a", "g8fd930", "gb200-4", "NVIDIA GB200", 4, "72", "768Gi"),
                ("cw-rno2a", "cw-rno2a-node-1", "h100-8", "NVIDIA H100 80GB HBM3", 8, "96", "1Ti"),
            )
        ]
        return [row for row in rows if not selected or row["cluster"] == selected]
    if path == "/k8s/workloads":
        selected = parse_qs(query).get("cluster", [""])[0]
        rows = [
            {
                "cluster": "cw-us-east-02a",
                "namespace": "iris",
                "pod": pod,
                "node": node_name,
                "job": job,
                "task": task,
                "phase": phase,
                "ready": phase == "Running",
                "priority_class": "iris-production",
                "age_seconds": age,
                "cpu_request_millicores": cpu,
                "memory_request_bytes": memory,
                "gpu_request_count": gpu,
                "gpu_variant": "H100" if gpu else "",
            }
            for pod, node_name, job, task, phase, age, cpu, memory, gpu in (
                ("llama-0", "gpu-a", "/alice/llama", "/alice/llama/0", "Running", 7300, 32_000, 343_597_383_680, 8),
                ("llama-1", "gpu-b", "/alice/llama", "/alice/llama/1", "Running", 7200, 16_000, 206_158_430_208, 4),
                ("eval-0", "gpu-b", "/bob/eval", "/bob/eval/0", "Running", 2800, 12_000, 103_079_215_104, 2),
                ("embed-0", "gpu-c", "/carol/embed", "/carol/embed/0", "Running", 1500, 8000, 68_719_476_736, 2),
                ("loader-0", "gpu-a", "/ops/loader", "/ops/loader/0", "Running", 600, 4000, 17_179_869_184, 0),
                ("train-queued", "", "/dave/train", "/dave/train/0", "Pending", 420, 16_000, 137_438_953_472, 4),
            )
        ]
        return [row for row in rows if not selected or row["cluster"] == selected]
    if path == "/k8s/pending":
        return [
            {
                "cluster": "cw-us-east-08a",
                "namespace": "iris",
                "pod": "trainer-queued",
                "state": "pending",
                "reason": "Unschedulable",
                "age_seconds": 420,
            }
        ]
    if path == "/k8s/crashloops":
        return [
            {
                "cluster": "cw-rno2a",
                "namespace": "training",
                "pod": "logger",
                "container": "logger",
                "reason": "CrashLoopBackOff",
                "restarts": 4,
                "scope": "workload",
                "error_class": "",
            }
        ]
    if path == "/k8s/termination_candidates":
        return [
            {
                "cluster": "cw-us-east-02a",
                "namespace": "training",
                "pod": "old-worker",
                "node": "gpu-node-1",
                "classification": "node-cleanup",
                "gpu_count": 8,
                "overdue_seconds": 900,
            }
        ]
    if path == "/k8s/kueue":
        return [{"cluster": "cw-us-east-08a", "queue": "training", "unadmitted": 6, "oldest_age_seconds": 540}]
    if path == "/k8s/gpu_racks":
        return [
            {
                "cluster": "cw-us-east-08a",
                "rack": rack,
                "rack_name": f"dh1-r{rack}-us-east-08a",
                "instance_type": "gb200-4x",
                "trays_total": total,
                "trays_ready": ready,
            }
            for rack, total, ready in (
                ("122", 17, 17),
                ("124", 17, 17),
                ("125", 17, 17),
                ("126", 18, 18),
                ("128", 16, 16),
                ("129", 18, 18),
                ("136", 17, 17),
                ("137", 16, 16),
                ("392", 16, 16),
                ("393", 16, 16),
                ("394", 16, 16),
                ("397", 15, 15),
            )
        ]
    if path == "/k8s/events":
        return [
            {
                "cluster": "cw-us-east-08a",
                "namespace": "training",
                "object": "Pod/trainer-queued",
                "reason": "FailedScheduling",
                "message": "waiting for H100 capacity",
                "count": 2,
                "last_seen": round(_NOW.timestamp() * 1000),
            }
        ]
    if path == "/wandb/history":
        return _wandb_history(parse_qs(query))
    if path.startswith("/wandb/report/"):
        return _wandb(path.rsplit("/", 1)[-1])
    if path == "/finelog/marin/query":
        return _finelog(query)
    return {"error": f"unknown fixture route {path}"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request = urlsplit(self.path)
        payload = _rows(request.path, request.query)
        status = 404 if isinstance(payload, dict) and "error" in payload else 200
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message_format: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8081), Handler).serve_forever()
