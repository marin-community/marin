#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "google-cloud-monitoring>=2.20",
# ]
# ///
"""CLI for monitoring and alerting on GCS cross-region egress in hai-gcp-models.

Usage:
    # Show top network consumers and bucket egress over the last 6 hours
    uv run scripts/egress_alert.py report --hours 6

    # Deploy full alerting infrastructure (Pub/Sub, alert policy, Cloud Function)
    uv run scripts/egress_alert.py install --slack-webhook-url https://hooks.slack.com/...

    # Fire a test alert to Slack (reads webhook from Secret Manager if omitted)
    uv run scripts/egress_alert.py test
"""

import json
import logging
import shlex
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
from google.cloud import monitoring_v3
from google.protobuf import duration_pb2

logger = logging.getLogger(__name__)

PROJECT = "hai-gcp-models"
REGION = "us-central1"
TOPIC_ID = "egress-alert-notifications"
SECRET_ID = "EGRESS_ALERT_SLACK_WEBHOOK"
ALERT_POLICY_DISPLAY_NAME = "Cross-Region Storage Egress"
NOTIFICATION_CHANNEL_DISPLAY_NAME = "Egress Alert Pub/Sub"
CLOUD_FUNCTION_NAME = "egress-alert-slack"
CLOUD_FUNCTION_DIR = Path(__file__).parent / "egress_alert_fn"

# Keep in sync with scripts/egress_alert_fn/main.py
METRICS = [
    ("VM sent", "compute.googleapis.com/instance/network/sent_bytes_count"),
    ("VM recv", "compute.googleapis.com/instance/network/received_bytes_count"),
    ("TPU sent", "tpu.googleapis.com/network/sent_bytes_count"),
    ("TPU recv", "tpu.googleapis.com/network/received_bytes_count"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def fmt_bytes(b: int) -> str:
    if b >= 1024**4:
        return f"{b / 1024**4:.1f} TiB"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GiB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MiB"
    return f"{b / 1024:.1f} KiB"


def query_metric(client: monitoring_v3.MetricServiceClient, metric_type: str, hours: int) -> dict[str, dict]:
    """Query a network bytes metric, return {key: {name, zone, bytes}}."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    interval = monitoring_v3.TimeInterval(start_time=start, end_time=now)

    results = client.list_time_series(
        request={
            "name": f"projects/{PROJECT}",
            "filter": f'metric.type="{metric_type}"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            "aggregation": monitoring_v3.Aggregation(
                alignment_period={"seconds": hours * 3600},
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


def query_bucket_egress(client: monitoring_v3.MetricServiceClient, hours: int) -> list[dict]:
    """Query GCS sent_bytes_count per bucket, return sorted list of {bucket, bytes}."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    interval = monitoring_v3.TimeInterval(start_time=start, end_time=now)

    results = client.list_time_series(
        request={
            "name": f"projects/{PROJECT}",
            "filter": 'metric.type="storage.googleapis.com/network/sent_bytes_count"' ' AND resource.type="gcs_bucket"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            "aggregation": monitoring_v3.Aggregation(
                alignment_period={"seconds": hours * 3600},
                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_SUM,
                cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
                group_by_fields=["resource.labels.bucket_name"],
            ),
        }
    )

    buckets: dict[str, int] = {}
    for ts in results:
        bucket = ts.resource.labels.get("bucket_name", "unknown")
        total = sum(p.value.int64_value for p in ts.points)
        buckets[bucket] = buckets.get(bucket, 0) + total

    items = [{"bucket": k, "bytes": v} for k, v in buckets.items()]
    items.sort(key=lambda x: x["bytes"], reverse=True)
    return items


def top_consumers(hours: int, limit: int = 15) -> list[dict]:
    """Query all four metrics and return the top N consumers sorted by bytes desc."""
    client = monitoring_v3.MetricServiceClient()
    all_items: list[dict] = []
    for label, metric_type in METRICS:
        data = query_metric(client, metric_type, hours)
        for val in data.values():
            all_items.append({**val, "direction": label})
    all_items.sort(key=lambda x: x["bytes"], reverse=True)
    return all_items[:limit]


# ---------------------------------------------------------------------------
# gcloud subprocess helper
# ---------------------------------------------------------------------------


def run_gcloud(
    cmd: str, *, check: bool = True, capture: bool = False, input_data: str | None = None
) -> subprocess.CompletedProcess:
    parts = shlex.split(cmd)
    logger.info("Running: %s", cmd)
    kwargs: dict = {"check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    if input_data is not None:
        kwargs["input"] = input_data.encode()
    return subprocess.run(parts, **kwargs)


def resource_exists(cmd: str) -> bool:
    return run_gcloud(cmd, check=False, capture=True).returncode == 0


# ---------------------------------------------------------------------------
# Install sub-steps
# ---------------------------------------------------------------------------


REQUIRED_APIS = [
    "eventarc.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
]


def enable_required_apis():
    apis = " ".join(REQUIRED_APIS)
    run_gcloud(f"gcloud services enable {apis} --project {PROJECT}")
    click.echo("  Required APIs enabled.")


def grant_pubsub_token_creator():
    """Grant the Pub/Sub SA the token creator role needed for authenticated push."""
    result = run_gcloud(
        f"gcloud projects describe {PROJECT} --format value(projectNumber)",
        capture=True,
    )
    project_number = result.stdout.strip()
    sa = f"service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com"
    run_gcloud(
        f"gcloud projects add-iam-policy-binding {PROJECT} "
        f"--member serviceAccount:{sa} "
        f"--role roles/iam.serviceAccountTokenCreator "
        f"--condition None "
        f"--quiet",
        check=False,
    )


def ensure_pubsub_topic():
    if resource_exists(f"gcloud pubsub topics describe {TOPIC_ID} --project {PROJECT}"):
        click.echo(f"  Pub/Sub topic {TOPIC_ID} already exists.")
        return
    run_gcloud(f"gcloud pubsub topics create {TOPIC_ID} --project {PROJECT}")
    click.echo(f"  Created Pub/Sub topic {TOPIC_ID}.")


def ensure_notification_channel() -> str:
    """Create or find a Pub/Sub notification channel. Returns the channel resource path."""
    client = monitoring_v3.NotificationChannelServiceClient()
    project_name = f"projects/{PROJECT}"
    topic = f"projects/{PROJECT}/topics/{TOPIC_ID}"

    for ch in client.list_notification_channels(name=project_name):
        if ch.display_name == NOTIFICATION_CHANNEL_DISPLAY_NAME and ch.type_ == "pubsub":
            click.echo(f"  Notification channel already exists: {ch.name}")
            return ch.name

    channel = monitoring_v3.NotificationChannel(
        type_="pubsub",
        display_name=NOTIFICATION_CHANNEL_DISPLAY_NAME,
        labels={"topic": topic},
    )
    created = client.create_notification_channel(name=project_name, notification_channel=channel)
    click.echo(f"  Created notification channel: {created.name}")
    return created.name


def ensure_alert_policy(notification_channel_name: str, threshold_gib_per_hour: int):
    """Create a Cloud Monitoring alert policy for cross-region storage egress."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{PROJECT}"

    for policy in client.list_alert_policies(name=project_name):
        if policy.display_name == ALERT_POLICY_DISPLAY_NAME:
            click.echo(f"  Alert policy already exists: {policy.name}")
            return

    threshold_bytes_per_sec = threshold_gib_per_hour * 1024**3 / 3600

    condition = monitoring_v3.AlertPolicy.Condition(
        display_name="Storage egress rate exceeds threshold",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter='metric.type="storage.googleapis.com/network/sent_bytes_count" AND resource.type="gcs_bucket"',
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period=duration_pb2.Duration(seconds=3600),
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE,
                    cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
                ),
            ],
            comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
            threshold_value=threshold_bytes_per_sec,
            duration=duration_pb2.Duration(seconds=0),
            trigger=monitoring_v3.AlertPolicy.Condition.Trigger(count=1),
        ),
    )

    policy = monitoring_v3.AlertPolicy(
        display_name=ALERT_POLICY_DISPLAY_NAME,
        conditions=[condition],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        notification_channels=[notification_channel_name],
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="Cross-region GCS storage egress has exceeded the configured threshold. "
            "Check the Slack #gcp-alerts channel for the top consumers.",
            mime_type="text/markdown",
        ),
    )
    created = client.create_alert_policy(name=project_name, alert_policy=policy)
    click.echo(f"  Created alert policy: {created.name}")


def ensure_slack_webhook_secret(webhook_url: str):
    if resource_exists(f"gcloud secrets describe {SECRET_ID} --project {PROJECT}"):
        click.echo(f"  Secret {SECRET_ID} already exists. Adding new version...")
        run_gcloud(
            f"gcloud secrets versions add {SECRET_ID} --project {PROJECT} --data-file=-",
            input_data=webhook_url,
        )
    else:
        run_gcloud(
            f"gcloud secrets create {SECRET_ID} --project {PROJECT} --replication-policy=automatic --data-file=-",
            input_data=webhook_url,
        )
    click.echo(f"  Slack webhook stored in secret {SECRET_ID}.")


def grant_iam_permissions():
    result = run_gcloud(
        f"gcloud iam service-accounts list --project {PROJECT} "
        f"--filter email:compute@developer.gserviceaccount.com --format value(email)",
        capture=True,
    )
    sa_email = result.stdout.strip()
    if not sa_email:
        click.echo("  WARNING: Could not find default compute service account.", err=True)
        return

    run_gcloud(
        f"gcloud projects add-iam-policy-binding {PROJECT} "
        f"--member serviceAccount:{sa_email} "
        f"--role roles/monitoring.viewer "
        f"--condition None "
        f"--quiet",
        check=False,
    )

    run_gcloud(
        f"gcloud secrets add-iam-policy-binding {SECRET_ID} "
        f"--project {PROJECT} "
        f"--member serviceAccount:{sa_email} "
        f"--role roles/secretmanager.secretAccessor",
        check=False,
    )
    click.echo(f"  Granted IAM roles to {sa_email}.")


def deploy_cloud_function():
    run_gcloud(
        f"gcloud functions deploy {CLOUD_FUNCTION_NAME} "
        f"--project {PROJECT} "
        f"--region {REGION} "
        f"--gen2 "
        f"--runtime python312 "
        f"--entry-point handle_alert "
        f"--trigger-topic {TOPIC_ID} "
        f"--source {CLOUD_FUNCTION_DIR} "
        f"--set-secrets SLACK_WEBHOOK_URL={SECRET_ID}:latest "
        f"--memory 512Mi "
        f"--timeout 120s "
        f"--max-instances 1",
    )
    click.echo(f"  Deployed Cloud Function {CLOUD_FUNCTION_NAME}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Monitor and alert on GCS cross-region egress in hai-gcp-models."""


@cli.command()
@click.option("--hours", default=24, show_default=True, help="Lookback window in hours.")
@click.option("--limit", default=30, show_default=True, help="Number of top consumers to show.")
def report(hours: int, limit: int):
    """Print a table of top network consumers (VMs + TPUs) over the last N hours."""
    click.echo(f"Querying Cloud Monitoring for last {hours} hours...")
    items = top_consumers(hours, limit)

    click.echo(f"\n{'=' * 110}")
    click.echo(f"  TOP {limit} NETWORK CONSUMERS (last {hours}h)")
    click.echo(f"{'=' * 110}")
    click.echo(f"  {'NAME':<50} {'ZONE':<22} {'DIRECTION':<12} {'BYTES':<15}")
    click.echo(f"  {'-' * 50} {'-' * 22} {'-' * 12} {'-' * 15}")
    for r in items:
        click.echo(f"  {r['name']:<50} {r['zone']:<22} {r['direction']:<12} {fmt_bytes(r['bytes']):<15}")

    eu_recv = sum(r["bytes"] for r in items if "europe" in r["zone"] and "recv" in r["direction"])
    eu_sent = sum(r["bytes"] for r in items if "europe" in r["zone"] and "sent" in r["direction"])
    if eu_recv or eu_sent:
        click.echo(f"\n  EU total in top {limit}: recv={fmt_bytes(eu_recv)}, sent={fmt_bytes(eu_sent)}")

    # Bucket egress
    click.echo("\nQuerying GCS bucket egress...")
    client = monitoring_v3.MetricServiceClient()
    buckets = query_bucket_egress(client, hours)
    if buckets:
        click.echo(f"\n{'=' * 80}")
        click.echo(f"  GCS BUCKET EGRESS (last {hours}h)")
        click.echo(f"{'=' * 80}")
        click.echo(f"  {'BUCKET':<55} {'EGRESS':<15}")
        click.echo(f"  {'-' * 55} {'-' * 15}")
        for b in buckets[:limit]:
            click.echo(f"  {b['bucket']:<55} {fmt_bytes(b['bytes']):<15}")
        total = sum(b["bytes"] for b in buckets)
        click.echo(f"  {'-' * 55} {'-' * 15}")
        click.echo(f"  {'TOTAL':<55} {fmt_bytes(total):<15}")


@cli.command()
@click.option("--slack-webhook-url", prompt="Slack webhook URL for #gcp-alerts", help="Slack Incoming Webhook URL.")
@click.option("--threshold-gib-per-hour", default=500, show_default=True, help="Egress threshold in GiB/hour.")
def install(slack_webhook_url: str, threshold_gib_per_hour: int):
    """Deploy the full GCP alerting infrastructure for cross-region egress monitoring.

    Creates: Pub/Sub topic, notification channel, alert policy, Secret Manager
    secret, Cloud Function, and IAM bindings.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    click.echo("=== Egress Alert Infrastructure Setup ===\n")

    click.echo("[1/8] Enabling required APIs...")
    enable_required_apis()

    click.echo("[2/8] Pub/Sub token creator role...")
    grant_pubsub_token_creator()

    click.echo("[3/8] Pub/Sub topic...")
    ensure_pubsub_topic()

    click.echo("[4/8] Notification channel...")
    channel_name = ensure_notification_channel()

    click.echo("[5/8] Alert policy...")
    ensure_alert_policy(channel_name, threshold_gib_per_hour)

    click.echo("[6/8] Slack webhook secret...")
    ensure_slack_webhook_secret(slack_webhook_url)

    click.echo("[7/8] IAM permissions...")
    grant_iam_permissions()

    click.echo("[8/8] Cloud Function...")
    deploy_cloud_function()

    click.echo(
        f"\nDone. The alert will fire when cross-region storage egress "
        f"exceeds {threshold_gib_per_hour} GiB/hour and post culprits to #gcp-alerts."
    )


def get_slack_webhook_from_secret() -> str:
    """Read the Slack webhook URL from Secret Manager."""
    result = run_gcloud(
        f"gcloud secrets versions access latest --secret={SECRET_ID} --project={PROJECT}",
        capture=True,
    )
    url = result.stdout.strip()
    if not url:
        raise click.ClickException(f"Secret {SECRET_ID} is empty. Run 'install' first or pass --slack-webhook-url.")
    return url


@cli.command()
@click.option(
    "--slack-webhook-url",
    default=None,
    help="Slack webhook URL. If omitted, reads from Secret Manager.",
)
@click.option("--hours", default=6, show_default=True, help="Lookback window in hours.")
@click.option("--limit", default=15, show_default=True, help="Number of top consumers to include.")
def test(slack_webhook_url: str | None, hours: int, limit: int):
    """Fire a test alert to Slack with the current top consumers."""
    if not slack_webhook_url:
        click.echo("Reading Slack webhook from Secret Manager...")
        slack_webhook_url = get_slack_webhook_from_secret()

    click.echo(f"Querying top {limit} consumers over last {hours}h...")
    items = top_consumers(hours, limit)
    if not items:
        click.echo("No consumers found.")
        return

    # Reuse the same Slack formatting as the Cloud Function
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"{'Name':<45} {'Zone':<22} {'Dir':<10} {'Bytes':>10}"]
    lines.append("-" * 90)
    for c in items:
        lines.append(f"{c['name'][:45]:<45} {c['zone']:<22} {c['direction']:<10} {fmt_bytes(c['bytes']):>10}")
    table = "\n".join(lines)

    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f":test_tube: *Cross-Region Egress Alert (TEST)* ({now_str})"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Top {limit} network consumers over the last {hours}h in `{PROJECT}`:",
                },
            },
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```\n{table}\n```"}},
        ],
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        slack_webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Slack responded with {resp.status}: {resp.read().decode()}")

    click.echo("Posted to Slack.")


if __name__ == "__main__":
    cli()
