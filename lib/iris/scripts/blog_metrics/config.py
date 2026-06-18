# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for the Iris blog-metrics pipeline.

Paths, finelog source coordinates, the TPU FLOPS table, the worker-variant ->
chip/FLOPS mapping, and the milestone annotations all live here so the fetch /
extract / chart steps agree on one source of truth.

The peak-FLOPS numbers are vendored from ``lib/fray/src/fray/device_flops.py``
(``DEVICE_FLOPS``) because ``fray`` is not importable from the iris
environment. Keep them in sync with that file if the canonical table changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from iris.cluster.tpu_topology import TPU_TOPOLOGIES, get_tpu_topology

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "data")


@dataclass(frozen=True)
class Paths:
    """Resolved on-disk layout for one pipeline run, rooted at ``data_dir``."""

    data_dir: str

    @property
    def raw_dir(self) -> str:
        """Bulk-copied finelog parquet + extracted controller audit lines."""
        return os.path.join(self.data_dir, "raw")

    @property
    def worker_parquet(self) -> str:
        return os.path.join(self.raw_dir, "iris.worker")

    @property
    def task_parquet(self) -> str:
        return os.path.join(self.raw_dir, "iris.task")

    @property
    def controller_audit_parquet(self) -> str:
        return os.path.join(self.raw_dir, "controller_audit.parquet")

    @property
    def wandb_runs_parquet(self) -> str:
        """Cached bulk pull of W&B run metadata (one row per run)."""
        return os.path.join(self.raw_dir, "wandb_runs.parquet")

    @property
    def daily_dir(self) -> str:
        """Small per-day CSV rollups — cheap to recompute (gitignored)."""
        return os.path.join(self.data_dir, "daily")

    @property
    def charts_dir(self) -> str:
        return os.path.join(self.data_dir, "charts")


def resolve_paths(data_dir: str | None) -> Paths:
    return Paths(data_dir=os.path.abspath(data_dir or DEFAULT_DATA_DIR))


# --------------------------------------------------------------------------- #
# Finelog source
# --------------------------------------------------------------------------- #
# The finelog deployment name (``finelog gcs-query <name> ...``) and the GCS
# prefix it archives to. iris.worker / iris.task carry structured per-heartbeat
# rows; the ``log`` namespace's ``/system/controller`` key holds the
# controller's audit lines (see audit_logging.py).
FINELOG_DEPLOYMENT = "marin"
REMOTE_LOG_DIR = "gs://marin-us-central2/finelog/marin"
WORKER_NAMESPACE = "iris.worker"
TASK_NAMESPACE = "iris.task"
CONTROLLER_LOG_KEY = "/system/controller"

# Time bucket used to turn dense heartbeats into a concurrency estimate: a
# worker present in a bucket (>= 1 heartbeat) counts once. Daily "mean" is the
# time-weighted average over buckets; daily "peak" is the busiest bucket.
CONCURRENCY_BUCKET_MINUTES = 10
CONCURRENCY_BUCKET = f"{CONCURRENCY_BUCKET_MINUTES} minutes"

# --------------------------------------------------------------------------- #
# W&B long-history source
# --------------------------------------------------------------------------- #
# Iris structured logs only reach back to the finelog Rust migration
# (~2026-05-06). W&B run history reaches back to the project's start (~2024-04)
# and is the only source for pre-Iris compute. Each run's summary carries
# num_devices, _runtime, parameter_count (N) and throughput/total_tokens (D), so
# realized training FLOPs are estimated as 6*N*D with no device-type lookup.
WANDB_ENTITY = "marin-community"
WANDB_PROJECTS = ["marin"]
# Multiplier in the standard transformer training-FLOPs estimate 6*N*D
# (forward+backward MAC*2). Eval/inference runs over-count under this (they do
# ~2ND), so realized FLOPs from W&B are an upper bound for non-training work.
TRAINING_FLOPS_PER_PARAM_TOKEN = 6.0

# The Iris stats window overlaps W&B from this date; used to calibrate the
# W&B-derived series (realized) against the cluster's provisioned capacity.
CALIBRATION_START = "2026-05-07"  # first full day of iris stats data

# --------------------------------------------------------------------------- #
# FLOPS table (peak bf16 FLOP/s per chip)
# --------------------------------------------------------------------------- #
# Vendored from lib/fray/src/fray/device_flops.py::DEVICE_FLOPS (TPU bf16
# entries). Keyed by the family token produced by FAMILY_OF (variant.split("-")[0]).
FAMILY_FLOPS_BF16: dict[str, float] = {
    "v3": 123e12 / 2,  # per-chip halved: a JAX device is one core
    "v4": 275e12,
    "v5litepod": 197e12,  # v5e
    "v5p": 459e12,
    "v6e": 918e12,
}

# Human-facing family label for charts/legends.
FAMILY_DISPLAY: dict[str, str] = {
    "v3": "v3",
    "v4": "v4",
    "v5litepod": "v5e",
    "v5p": "v5p",
    "v6e": "v6e",
}


def family_of(variant: str) -> str:
    """Family token for a worker ``device_variant`` (e.g. ``v5p-32`` -> ``v5p``)."""
    return variant.split("-")[0]


@dataclass(frozen=True)
class VariantInfo:
    """Per-worker (one VM) chip count and peak FLOPS for a TPU slice variant."""

    variant: str
    family: str
    family_display: str
    chips_per_vm: int
    flops_per_chip: float


def build_variant_table() -> list[VariantInfo]:
    """Resolve every known TPU topology to its per-VM chip count and FLOPS.

    Each ``iris.worker`` row reports the slice ``device_variant`` of the VM it
    runs on; one worker == one VM, so its chip contribution is ``chips_per_vm``.
    Variants whose family has no FLOPS entry are dropped (with the family token
    left discoverable via the topology list).
    """
    rows: list[VariantInfo] = []
    for topo in TPU_TOPOLOGIES:
        variant = topo.name.lower()
        family = family_of(variant)
        flops = FAMILY_FLOPS_BF16.get(family)
        if flops is None:
            continue
        rows.append(
            VariantInfo(
                variant=variant,
                family=family,
                family_display=FAMILY_DISPLAY.get(family, family),
                chips_per_vm=topo.chips_per_vm,
                flops_per_chip=flops,
            )
        )
    return rows


def variant_chips(variant: str) -> int:
    """Per-VM chip count for a single ``device_variant`` (0 if unknown)."""
    try:
        return get_tpu_topology(variant).chips_per_vm
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
# The first path component of a task id (``/<user>/<job>/.../<idx>``) is the
# submitting user. These are automation namespaces, not people — excluded from
# the "human active users" count but kept in the raw count.
BOT_USERS: frozenset[str] = frozenset({"infra-probes", "runner", "canary"})

# --------------------------------------------------------------------------- #
# Milestones
# --------------------------------------------------------------------------- #
# (ISO date, short label) annotations drawn as vertical markers. Dates outside
# the data window are dropped by the chart step. Sourced from git history of
# lib/iris; edit freely.
MILESTONES: list[tuple[str, str]] = [
    ("2024-04-10", "First W&B-tracked runs"),
    ("2026-04-01", "User budgets + priority bands (#4096)"),
    ("2026-04-08", "v4-reserved pool (#4528)"),
    ("2026-04-22", "Ray → Iris cutover (#5076)"),
    ("2026-05-08", "Scheduling queue (#5563)"),
    ("2026-06-16", "Empirical availability constraint (#6438)"),
]
