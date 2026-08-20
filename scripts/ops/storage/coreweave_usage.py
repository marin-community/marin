# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record CoreWeave object-storage usage and zone quotas in Finelog."""

import datetime as dt
import json
import logging
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar

import click
import requests
from finelog.client import FlushResult, LogClient, StoragePolicy
from finelog.deploy.config import load_finelog_config, tunnel_target_for
from rigging.tunnel import open_tunnel

logger = logging.getLogger(__name__)

PROVIDER = "coreweave"
STORAGE_CLASS = "STANDARD"
STORAGE_USAGE_NAMESPACE = "storage.usage"
STORAGE_USAGE_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_PROMETHEUS_URL = "https://observe.coreweave.com"
DEFAULT_FINELOG_CONFIG = "marin"
REQUEST_TIMEOUT = 60.0
FLUSH_TIMEOUT = 30.0
_BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

USAGE_QUERY = (
    "sum by (bucket_name, zone, storage_class) "
    f'(billing:object_storage_used_bytes:total{{storage_class="{STORAGE_CLASS}"}})'
)
QUOTA_QUERY = (
    "max by (quota_zone, storage_class) "
    '(cwobject_quota_info{active="true", '
    'measurement_type="MEASUREMENT_TYPE_USAGE_BYTES", '
    f'storage_class="{STORAGE_CLASS}"}})'
)


class StorageMetric(StrEnum):
    """The byte gauge stored in a storage usage row."""

    USED_BYTES = "used_bytes"
    QUOTA_BYTES = "quota_bytes"


class StorageUsageError(RuntimeError):
    """The CoreWeave API did not return valid storage telemetry."""


@dataclass(frozen=True)
class StorageUsage:
    """One observed object-storage usage or quota gauge."""

    key_column: ClassVar[str] = "zone"

    provider: str
    metric: str
    zone: str
    bucket: str | None
    storage_class: str
    value_bytes: float
    observed_at: dt.datetime
    collected_at: dt.datetime


def _query(session: requests.Session, base_url: str, query: str) -> list[dict[str, Any]]:
    try:
        response = session.get(
            f"{base_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise StorageUsageError(f"CoreWeave storage query failed: {exc}") from exc

    if response.status_code in (401, 403):
        raise StorageUsageError(f"CoreWeave returned HTTP {response.status_code}. The token needs Observability Viewer.")
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StorageUsageError(f"CoreWeave storage query returned an invalid response: {exc}") from exc
    if payload.get("status") != "success":
        raise StorageUsageError(f"CoreWeave storage query returned status {payload.get('status')!r}")
    result = payload.get("data", {}).get("result", [])
    if not isinstance(result, list):
        raise StorageUsageError("CoreWeave storage query did not return a vector")
    return result


def _label(labels: Mapping[str, Any], name: str) -> str:
    value = labels.get(name)
    if not isinstance(value, str) or not value:
        raise StorageUsageError(f"CoreWeave storage series has no {name!r} label")
    return value


def _value(item: Mapping[str, Any]) -> tuple[dt.datetime, float]:
    sample = item.get("value")
    if not isinstance(sample, list) or len(sample) != 2:
        raise StorageUsageError("CoreWeave storage series has no instant value")
    try:
        observed_at = dt.datetime.fromtimestamp(float(sample[0]), tz=dt.UTC)
        value_bytes = float(sample[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise StorageUsageError("CoreWeave storage series has an invalid instant value") from exc
    if not math.isfinite(value_bytes) or value_bytes < 0:
        raise StorageUsageError(f"CoreWeave storage series has invalid byte value {value_bytes!r}")
    return observed_at, value_bytes


def collect_storage_usage(
    session: requests.Session,
    base_url: str,
    collected_at: dt.datetime,
) -> list[StorageUsage]:
    """Collect current bucket usage and active zone quotas from CoreWeave."""
    if collected_at.tzinfo is None:
        raise ValueError("collected_at must include a timezone")

    usage_series = _query(session, base_url, USAGE_QUERY)
    if not usage_series:
        raise StorageUsageError("CoreWeave usage query returned no series")

    samples: list[StorageUsage] = []
    usage_zones: set[tuple[str, str]] = set()
    for item in usage_series:
        labels = item.get("metric", {})
        if not isinstance(labels, Mapping):
            raise StorageUsageError("CoreWeave usage series has invalid labels")
        zone = _label(labels, "zone")
        storage_class = _label(labels, "storage_class")
        observed_at, value_bytes = _value(item)
        usage_zones.add((zone, storage_class))
        samples.append(
            StorageUsage(
                provider=PROVIDER,
                metric=str(StorageMetric.USED_BYTES),
                zone=zone,
                bucket=_label(labels, "bucket_name"),
                storage_class=storage_class,
                value_bytes=value_bytes,
                observed_at=observed_at,
                collected_at=collected_at,
            )
        )

    quota_series = _query(session, base_url, QUOTA_QUERY)
    quota_zones: set[tuple[str, str]] = set()
    for item in quota_series:
        labels = item.get("metric", {})
        if not isinstance(labels, Mapping):
            raise StorageUsageError("CoreWeave quota series has invalid labels")
        zone = _label(labels, "quota_zone")
        storage_class = _label(labels, "storage_class")
        observed_at, value_bytes = _value(item)
        if value_bytes <= 0:
            raise StorageUsageError(f"CoreWeave quota for {zone} must be positive")
        quota_zones.add((zone, storage_class))
        samples.append(
            StorageUsage(
                provider=PROVIDER,
                metric=str(StorageMetric.QUOTA_BYTES),
                zone=zone,
                bucket=None,
                storage_class=storage_class,
                value_bytes=value_bytes,
                observed_at=observed_at,
                collected_at=collected_at,
            )
        )

    missing_quotas = sorted(usage_zones - quota_zones)
    if missing_quotas:
        zones = ", ".join(f"{zone}/{storage_class}" for zone, storage_class in missing_quotas)
        raise StorageUsageError(f"CoreWeave quota query returned no series for {zones}")

    return sorted(samples, key=lambda sample: (sample.zone, sample.metric, sample.bucket or ""))


def _session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": _BROWSER_USER_AGENT})
    return session


@contextmanager
def _finelog_client(finelog_url: str | None, finelog_config: str) -> Iterator[LogClient]:
    if finelog_url:
        logger.info("Connecting to Finelog at %s", finelog_url)
        with closing(LogClient.connect(finelog_url)) as client:
            yield client
        return

    config = load_finelog_config(finelog_config)
    target = tunnel_target_for(config)
    logger.info("Opening a tunnel to Finelog %r", config.name)
    with open_tunnel(target, timeout=60.0) as tunnel_url, closing(LogClient.connect(tunnel_url)) as client:
        yield client


def _write(client: LogClient, samples: list[StorageUsage]) -> None:
    table = client.get_table(
        STORAGE_USAGE_NAMESPACE,
        StorageUsage,
        storage_policy=StoragePolicy(max_bytes=STORAGE_USAGE_MAX_BYTES),
    )
    table.write(samples)
    result = table.flush(timeout=FLUSH_TIMEOUT)
    if result is not FlushResult.SUCCEEDED:
        raise RuntimeError(f"Finelog flush did not complete within {FLUSH_TIMEOUT:.0f} seconds: {result}")


@click.command(help=__doc__)
@click.option("--prometheus-url", default=DEFAULT_PROMETHEUS_URL, show_default=True)
@click.option("--finelog-url", default=None, help="Connect directly to this Finelog URL.")
@click.option("--finelog-config", default=DEFAULT_FINELOG_CONFIG, show_default=True)
@click.option("--dry-run/--no-dry-run", default=False, help="Print rows and do not write to Finelog.")
def main(prometheus_url: str, finelog_url: str | None, finelog_config: str, dry_run: bool) -> None:
    """Collect and record one storage telemetry snapshot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    token = os.environ.get("COREWEAVE_API_TOKEN")
    if not token:
        raise click.ClickException("COREWEAVE_API_TOKEN is not set")

    collected_at = dt.datetime.now(dt.UTC)
    with _session(token) as session:
        samples = collect_storage_usage(session, prometheus_url, collected_at)

    if dry_run:
        for sample in samples:
            print(json.dumps(asdict(sample), default=str, sort_keys=True))
    else:
        with _finelog_client(finelog_url, finelog_config) as client:
            _write(client, samples)
    logger.info("Recorded %d CoreWeave storage rows (dry_run=%s)", len(samples), dry_run)


if __name__ == "__main__":
    main()
