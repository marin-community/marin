# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloud Function: posts top network consumers to Slack when a cross-region egress alert fires."""

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import monitoring_v3

PROJECT = "hai-gcp-models"
LOOKBACK_HOURS = 6
TOP_N = 15

METRICS = [
    ("VM sent", "compute.googleapis.com/instance/network/sent_bytes_count"),
    ("VM recv", "compute.googleapis.com/instance/network/received_bytes_count"),
    ("TPU sent", "tpu.googleapis.com/network/sent_bytes_count"),
    ("TPU recv", "tpu.googleapis.com/network/received_bytes_count"),
]


def fmt_bytes(b: int) -> str:
    if b >= 1024**4:
        return f"{b / 1024**4:.1f} TiB"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GiB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MiB"
    return f"{b / 1024:.1f} KiB"


def query_metric(client: monitoring_v3.MetricServiceClient, metric_type: str) -> dict[str, dict]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=LOOKBACK_HOURS)
    interval = monitoring_v3.TimeInterval(start_time=start, end_time=now)

    results = client.list_time_series(
        request={
            "name": f"projects/{PROJECT}",
            "filter": f'metric.type="{metric_type}"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            "aggregation": monitoring_v3.Aggregation(
                alignment_period={"seconds": LOOKBACK_HOURS * 3600},
                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_SUM,
            ),
        }
    )

    totals: dict[str, dict] = {}
    for ts in results:
        labels = ts.resource.labels
        name = labels.get("instance_name") or labels.get("node_id") or labels.get("instance_id", "unknown")
        zone = labels.get("zone", "unknown")
        key = f"{name}|{zone}"
        total = sum(p.value.int64_value for p in ts.points)
        if key in totals:
            totals[key]["bytes"] += total
        else:
            totals[key] = {"name": name, "zone": zone, "bytes": total}
    return totals


def top_consumers() -> list[dict]:
    client = monitoring_v3.MetricServiceClient()
    all_items: list[dict] = []
    for label, metric_type in METRICS:
        data = query_metric(client, metric_type)
        for val in data.values():
            all_items.append({**val, "direction": label})
    all_items.sort(key=lambda x: x["bytes"], reverse=True)
    return all_items[:TOP_N]


def build_slack_blocks(consumers: list[dict]) -> dict:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = f":rotating_light: *Cross-Region Egress Alert* ({now_str})"
    subheader = f"Top {TOP_N} network consumers over the last {LOOKBACK_HOURS}h in `{PROJECT}`:"

    lines = [f"{'Name':<45} {'Zone':<22} {'Dir':<10} {'Bytes':>10}"]
    lines.append("-" * 90)
    for c in consumers:
        lines.append(f"{c['name'][:45]:<45} {c['zone']:<22} {c['direction']:<10} {fmt_bytes(c['bytes']):>10}")

    table = "\n".join(lines)

    return {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": header}},
            {"type": "section", "text": {"type": "mrkdwn", "text": subheader}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```\n{table}\n```"}},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"<https://console.cloud.google.com/monitoring/alerting?project={PROJECT}"
                            "|View in Cloud Monitoring>"
                        ),
                    }
                ],
            },
        ],
    }


def post_to_slack(payload: dict):
    webhook_url = os.environ["SLACK_WEBHOOK_URL"]
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Slack responded with {resp.status}: {resp.read().decode()}")


@functions_framework.cloud_event
def handle_alert(cloud_event: CloudEvent):
    """Entry point for the Cloud Function. Triggered by Pub/Sub from a Monitoring alert."""
    print(f"Alert received: {cloud_event.data}")

    consumers = top_consumers()
    if not consumers:
        print("No consumers found in the lookback window.")
        return

    payload = build_slack_blocks(consumers)
    post_to_slack(payload)
    print(f"Posted {len(consumers)} consumers to Slack.")
