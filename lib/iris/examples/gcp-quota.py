#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCP TPU quota and usage viewer.

Shows TPU counts and quotas across all zones.

Usage:
    uv run examples/gcp-quota.py
    uv run examples/gcp-quota.py --project=my-project
    uv run examples/gcp-quota.py --zones=europe-west4-b,us-central1-a
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tabulate import tabulate

# All known TPU types by generation
TPU_TYPES = {
    "v4": [f"v4-{n}" for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]],
    "v5e": [f"v5litepod-{n}" for n in [1, 4, 8, 16, 32, 64, 128, 256]],
    "v5p": [f"v5p-{n}" for n in [8, 16, 32, 64, 128, 256, 384, 512]],
    "v6e": [f"v6e-{n}" for n in [1, 4, 8, 16, 32, 64, 128, 256]],
}

# Flatten for easy lookup
ALL_TPU_TYPES = [t for types in TPU_TYPES.values() for t in types]

# Quota metrics by generation (preemptible)
QUOTA_METRICS = {
    "v4": "tpu.googleapis.com/tpu-v4-preemptible",
    "v5e": "tpu.googleapis.com/tpu-v5s-litepod-preemptible",
    "v5p": "tpu.googleapis.com/tpu-v5p-preemptible",
    "v6e": "tpu.googleapis.com/tpu-v6e-preemptible",
}

DEFAULT_ZONES = [
    "europe-west4-a",
    "europe-west4-b",
    "us-central1-a",
    "us-central2-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "us-west4-a",
]


def get_generation(accelerator_type: str) -> str | None:
    """Get TPU generation from accelerator type."""
    for gen, types in TPU_TYPES.items():
        if accelerator_type in types:
            return gen
    return None


def get_chip_count(accelerator_type: str) -> int:
    """Extract chip count from accelerator type."""
    match = re.search(r"-(\d+)$", accelerator_type)
    return int(match.group(1)) if match else 0


@dataclass
class ZoneData:
    zone: str
    tpu_counts: dict[str, int] = field(default_factory=dict)
    quotas: dict[str, int] = field(default_factory=dict)  # gen -> limit
    usage: dict[str, int] = field(default_factory=dict)  # gen -> chips used


def run_gcloud(args: list[str]) -> dict | list | None:
    """Run a gcloud command and return JSON output."""
    cmd = ["gcloud", *args, "--format=json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def get_quotas(project: str) -> dict[str, dict[str, int]]:
    """Get TPU quotas. Returns zone -> {gen -> limit}."""
    data = run_gcloud(
        [
            "alpha",
            "services",
            "quota",
            "list",
            f"--consumer=projects/{project}",
            "--service=tpu.googleapis.com",
        ]
    )
    if not data:
        return {}

    quotas: dict[str, dict[str, int]] = defaultdict(dict)

    for item in data:
        metric = item.get("metric", "")
        gen = None
        for g, m in QUOTA_METRICS.items():
            if metric == m:
                gen = g
                break
        if not gen:
            continue

        for limit_info in item.get("consumerQuotaLimits", []):
            for bucket in limit_info.get("quotaBuckets", []):
                dims = bucket.get("dimensions", {})
                eff = bucket.get("effectiveLimit", bucket.get("defaultLimit"))
                if eff is None or eff == "-1" or eff == -1:
                    continue
                zone = dims.get("zone", "")
                if zone:
                    quotas[zone][gen] = int(eff)
                elif not dims:
                    quotas["_default"][gen] = int(eff)

    return dict(quotas)


def get_tpus(project: str, zone: str) -> list[dict]:
    """Get all TPU VMs in a zone."""
    data = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--zone={zone}",
            f"--project={project}",
        ]
    )
    return data if isinstance(data, list) else []


def get_zone_data(project: str, zones: list[str]) -> list[ZoneData]:
    """Get TPU data for all zones in parallel."""
    print("Fetching quotas...", file=sys.stderr)
    all_quotas = get_quotas(project)
    default_quotas = all_quotas.get("_default", {})

    def fetch_zone(zone: str) -> ZoneData:
        tpus = get_tpus(project, zone)
        counts: dict[str, int] = defaultdict(int)
        usage: dict[str, int] = defaultdict(int)
        for tpu in tpus:
            acc_type = tpu.get("acceleratorType", "")
            counts[acc_type] += 1
            gen = get_generation(acc_type)
            if gen:
                usage[gen] += get_chip_count(acc_type)

        zone_quotas = all_quotas.get(zone, {})
        merged_quotas = {**default_quotas, **zone_quotas}

        return ZoneData(zone=zone, tpu_counts=dict(counts), quotas=merged_quotas, usage=dict(usage))

    print(f"Checking {len(zones)} zones in parallel...", file=sys.stderr)
    results: dict[str, ZoneData] = {}
    with ThreadPoolExecutor(max_workers=len(zones)) as pool:
        futures = {pool.submit(fetch_zone, zone): zone for zone in zones}
        for future in as_completed(futures):
            zone = futures[future]
            results[zone] = future.result()

    return [results[zone] for zone in zones]


def format_table(zone_data: list[ZoneData]) -> str:
    """Format as table with per-type columns."""
    # Find which types have any TPUs running
    types_with_tpus: set[str] = set()
    for zd in zone_data:
        types_with_tpus.update(zd.tpu_counts.keys())

    # Filter to only types that exist somewhere
    active_types = [t for t in ALL_TPU_TYPES if t in types_with_tpus]

    # Build headers
    headers = ["Zone"]
    for gen in TPU_TYPES:
        headers.append(f"{gen}:quota")
    for t in active_types:
        # Shorten: v5litepod-128 -> v5e-128, v4-8 -> v4-8
        if t.startswith("v5litepod-"):
            short = t.replace("v5litepod-", "v5e-")
        else:
            short = t
        headers.append(short)

    # Build rows
    rows = []
    for zd in zone_data:
        row = [zd.zone]
        # Usage/quota per generation
        for gen in TPU_TYPES:
            q = zd.quotas.get(gen, 0)
            u = zd.usage.get(gen, 0)
            if q:
                row.append(f"{u}/{q}")
            else:
                row.append("-")
        # Counts per type
        for t in active_types:
            count = zd.tpu_counts.get(t, 0)
            row.append(str(count) if count else "-")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="simple")


def main():
    parser = argparse.ArgumentParser(description="View GCP TPU quotas and usage")
    parser.add_argument("--project", default="hai-gcp-models", help="GCP project ID")
    parser.add_argument("--zones", help="Comma-separated list of zones to check")
    args = parser.parse_args()

    zones = args.zones.split(",") if args.zones else DEFAULT_ZONES
    data = get_zone_data(args.project, zones)

    print(f"\nTPU Status for project: {args.project}")
    print("Quotas are per-generation (chips). Counts are TPU instances.\n")
    print(format_table(data))


if __name__ == "__main__":
    main()
