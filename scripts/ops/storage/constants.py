# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and tiny helpers for the storage tooling."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

SCRIPT_PATH = Path(__file__).resolve()
STORAGE_DIR = SCRIPT_PATH.parent
REPO_ROOT = STORAGE_DIR.parent.parent

# ---------------------------------------------------------------------------
# Buckets & regions
# ---------------------------------------------------------------------------

MARIN_BUCKETS = [
    "marin-us-central1",
    "marin-us-central2",
    "marin-eu-west4",
    "marin-us-east1",
    "marin-us-east5",
    "marin-us-west4",
]

BUCKET_LOCATIONS = {
    "marin-eu-west4": "EUROPE-WEST4",
    "marin-us-central1": "US-CENTRAL1",
    "marin-us-central2": "US-CENTRAL2",
    "marin-us-east1": "US-EAST1",
    "marin-us-east5": "US-EAST5",
    "marin-us-west4": "US-WEST4",
}

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# (id, name, US $/GiB/month, EU $/GiB/month)
STORAGE_CLASS_PRICING: list[tuple[int, str, float, float]] = [
    (1, "STANDARD", 0.020, 0.023),
    (2, "NEARLINE", 0.010, 0.013),
    (3, "COLDLINE", 0.004, 0.006),
    (4, "ARCHIVE", 0.0012, 0.0025),
]

STORAGE_CLASS_IDS = {name: sc_id for sc_id, name, _, _ in STORAGE_CLASS_PRICING}

GCS_DISCOUNT = 0.30
DISCOUNT_FACTOR = 1.0 - GCS_DISCOUNT

NON_STANDARD_STORAGE_CLASSES = frozenset({"NEARLINE", "COLDLINE", "ARCHIVE"})

# ---------------------------------------------------------------------------
# GCS scan constants
# ---------------------------------------------------------------------------

GCS_MAX_PAGE_SIZE = 5000
GCS_LIST_TIMEOUT = 120
BLOB_FIELDS = "items(name,size,storageClass,timeCreated,updated),prefixes,nextPageToken"

ADAPTIVE_SPLIT_THRESHOLD = 25_000
ADAPTIVE_MAX_DEPTH = 2

# ---------------------------------------------------------------------------
# Arrow schema for objects parquet
# ---------------------------------------------------------------------------

OBJECTS_ARROW_SCHEMA = pa.schema(
    [
        ("bucket", pa.string()),
        ("name", pa.string()),
        ("size_bytes", pa.int64()),
        ("storage_class_id", pa.int32()),
        ("created", pa.timestamp("us", tz="UTC")),
        ("updated", pa.timestamp("us", tz="UTC")),
    ]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def continent_for_region(region: str) -> str:
    return "EUROPE" if region.startswith("eu-") else "US"


def region_from_bucket(bucket: str) -> str:
    return bucket.removeprefix("marin-") if bucket.startswith("marin-") else bucket


def plan_rows() -> list[dict[str, str]]:
    """One row per bucket with region and location."""
    return [{"region": region_from_bucket(b), "bucket": b, "location": BUCKET_LOCATIONS[b]} for b in MARIN_BUCKETS]
