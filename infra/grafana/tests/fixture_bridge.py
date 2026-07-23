# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic HTTP fixture for rendering the provisioned dashboard in Grafana."""

import json
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

_NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)
_LANES = (
    ("tpu-ferry", "TPU ferry", "marin", "training"),
    ("cw-gpu-ferry", "CW ferry", "marin", "training"),
    ("grug-multislice", "Grug", "marin", "training"),
    ("datakit-t1", "Data T1", "marin", "data"),
    ("datakit-t2", "Data T2", "marin", "data"),
    ("datakit-t3", "Data T3", "marin", "data"),
    ("cluster-smoke", "Cluster", "marin", "cluster"),
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
    for run_index, run in enumerate(("67b-a2b-10t", "67b-a2b-resume")):
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
                    "run_state": "running" if run_index else "finished",
                    "tokens": tokens,
                    "value": value,
                    "report_title": "67B-A2B MoE on 10T tokens",
                    "report_url": (
                        "https://wandb.ai/marin-community/marin_moe/reports/"
                        "67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ"
                    ),
                }
            )
    return rows


def _finelog(query: str) -> list[dict]:
    sql = parse_qs(query).get("sql", [""])[0]
    if "probe_latency_ms" in sql and "ROW_NUMBER" not in sql:
        return [
            {
                "t": round((_NOW - timedelta(minutes=5 * index)).timestamp() * 1000),
                "label_probe": probe,
                "value": 22 + index % 8,
            }
            for probe in ("iris", "finelog", "kueue")
            for index in range(24)
        ]
    if "probe_up" in sql and "metric IN" not in sql:
        return [{"value": 1}, {"value": 1}, {"value": 1}]
    if "metric IN" in sql:
        return [
            {"probe": probe, "metric": metric, "value": value}
            for probe in ("iris", "finelog", "kueue")
            for metric, value in (("probe_up", 1), ("probe_latency_ms", 24))
        ]
    if "worker_healthy" in sql:
        return [
            {
                "t": round((_NOW - timedelta(minutes=10 * index)).timestamp() * 1000),
                "label_region": region,
                "value": base + index % 3,
            }
            for region, base in (("us-east5", 84), ("us-central2", 51), ("cw-us-east", 37))
            for index in range(24)
        ]
    if "provision_success_ratio" in sql:
        return [
            {"t": round((_NOW - timedelta(minutes=10 * index)).timestamp() * 1000), "value": 0.96 + (index % 4) * 0.008}
            for index in range(24)
        ]
    return []


def _rows(path: str, query: str) -> list[dict] | dict:
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
    if path == "/iris/marin/jobs":
        return [
            {"bucket": "inflight", "state": "running", "count": 43},
            {"bucket": "last24h", "state": "succeeded", "count": 318},
            {"bucket": "last24h", "state": "failed", "count": 9},
        ]
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
            for cluster in ("cw-us-east-02a", "cw-us-east-08a", "cw-rno2a")
        ]
    if path == "/k8s/alerts/unreachable":
        return [
            {"cluster": cluster, "error_class": "none", "value": 0}
            for cluster in ("cw-us-east-02a", "cw-us-east-08a", "cw-rno2a")
        ]
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
            for cluster in ("cw-us-east-02a", "cw-us-east-08a", "cw-rno2a")
            for component in ("iris/iris-controller", "kueue-system/kueue-controller-manager")
        ]
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
    if path.startswith("/wandb/"):
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
