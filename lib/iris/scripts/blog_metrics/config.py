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
    def audit_raw_dir(self) -> str:
        """GCP audit-log create/delete event CSVs + the current live-fleet snapshot."""
        return os.path.join(self.raw_dir, "audit")

    @property
    def live_tpus_csv(self) -> str:
        """Current live TPU node names (``gcloud compute tpus tpu-vm list``)."""
        return os.path.join(self.audit_raw_dir, "live_tpus.csv")

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

# A single day chosen to illustrate *intraday* compute migration across regions:
# preemptible capacity sloshes between regions over the day (on 2026-06-01,
# us-east5 dominates the morning then hands off to europe-west4 in the
# afternoon) while the reserved v4 pool (us-central2) holds flat. Bucketed finer
# than the daily series so the within-day movement is visible.
INTRADAY_CANDIDATE_DAY = "2026-06-01"
INTRADAY_BUCKET = "30 minutes"

# --------------------------------------------------------------------------- #
# W&B long-history source
# --------------------------------------------------------------------------- #
# Iris structured logs only reach back to the finelog Rust migration
# (~2026-05-06). W&B run history reaches back to the project's start (~2024-04)
# and is the only source for pre-Iris compute. Each run's summary carries
# num_devices, _runtime, parameter_count (N) and throughput/total_tokens (D), so
# realized training FLOPs are estimated as 6*N*D, capped at physical capacity
# (see REALIZED_FLOPS_CEILING_* below).
WANDB_ENTITY = "marin-community"
# The flagship pretraining lives in "marin"; the rest are the sibling projects
# that log device-bearing runs *with* N and D (so they contribute FLOPs): the
# MoE training projects and the two largest optimizer sweeps. Eval / post-train
# projects log devices but no token counts, so they cannot form 6*N*D and are
# omitted.
WANDB_PROJECTS = ["marin", "dial_moe", "marin_moe", "optimizer-scaling", "Hyperball"]
# Multiplier in the standard transformer training-FLOPs estimate 6*N*D
# (forward+backward MAC*2).
TRAINING_FLOPS_PER_PARAM_TOKEN = 6.0

# W&B's throughput/total_tokens (D) is a CUMULATIVE counter that resumed and
# cooldown runs inherit from their parent, so 6*N*D counts a flagship lineage
# many times over: a 1-hour 32B cooldown re-reports the parent's ~6T tokens and
# 6*N*D then attributes ~1e24 FLOPs to it -> an implied >1000x MFU. We therefore
# cap each run's 6*N*D at the most its *own* hardware-time could physically
# deliver, ``num_devices * runtime * peak * mfu``. Clean runs (6*N*D below the
# ceiling) keep their exact estimate; only the cumulative-counter runs are
# bounded, and the bound uses each run's own device count so it honors a fleet
# whose size varied over time. The peak is the generous v5p per-chip rate so
# genuine v5p/v6e runs are never wrongly capped; mfu is a hardware ceiling, not a
# typical efficiency.
REALIZED_FLOPS_CEILING_PEAK_BF16 = 459e12  # v5p per-chip bf16 FLOP/s
REALIZED_FLOPS_CEILING_MFU = 0.6

# Calibration compares W&B realized compute against iris *provisioned* capacity,
# so it is only valid where all training actually ran on iris. That holds from
# early June: April's v4 runs were still on Ray (no iris logs), and May is mixed
# — iris was bootstrapping (~May 6) and some runs ran elsewhere, so W&B
# device-hours *exceed* iris's on May 7-8 (an impossible >100% coverage). Start
# in June for an apples-to-apples window.
CALIBRATION_START = "2026-06-01"

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

# --------------------------------------------------------------------------- #
# GCP audit-log source (long-history TPU provisioning)
# --------------------------------------------------------------------------- #
# finelog/W&B only reach the Iris era (~2026-05-06). GCP Admin-Activity audit
# logs in the locked ``_Required`` bucket retain 400 days and record every TPU
# CreateNode/DeleteNode and CreateQueuedResource/DeleteQueuedResource, so they
# reconstruct *provisioned* TPU capacity ~13 months back — the only native GCP
# source that does, since this project's billing export is inaccessible (the
# billing account is Stanford-managed). See audit_logs.py for the method.
GCP_PROJECT = "hai-gcp-models"

# Earliest day we pull; the 400-day audit retention floor sat here when the
# reconstruction was first run, so going earlier returns truncated history.
AUDIT_WINDOW_START = "2025-05-14"

# Long-history charts are truncated to this start for the blog (the pre-2026
# Ray era is slice-count-only — see the size limitation in audit_logs.py).
AUDIT_CHART_START = "2026-01-01"

# Successful-completion audit method names, across both API versions in use over
# the window (older months emit v2alpha1, not v2 — a v2-only pull undercounts
# ~50x). Reserved capacity arrives via QueuedResource, not Node.
AUDIT_NODE_CREATE_METHODS = (
    "google.cloud.tpu.v2.Tpu.CreateNode",
    "google.cloud.tpu.v2alpha1.Tpu.CreateNode",
    "google.cloud.tpu.v1.Tpu.CreateNode",
)
AUDIT_NODE_DELETE_METHODS = (
    "google.cloud.tpu.v2.Tpu.DeleteNode",
    "google.cloud.tpu.v2alpha1.Tpu.DeleteNode",
    "google.cloud.tpu.v1.Tpu.DeleteNode",
)
AUDIT_QR_CREATE_METHODS = (
    "google.cloud.tpu.v2.Tpu.CreateQueuedResource",
    "google.cloud.tpu.v2alpha1.Tpu.CreateQueuedResource",
)
AUDIT_QR_DELETE_METHODS = (
    "google.cloud.tpu.v2.Tpu.DeleteQueuedResource",
    "google.cloud.tpu.v2alpha1.Tpu.DeleteQueuedResource",
)

# Zones queried for the current live fleet (refines the right edge of the
# reconstruction: create-only nodes still present here are clamped to "now"
# rather than capped at the class-median lifetime). A zone with no TPUs simply
# returns nothing, so an over-broad list is harmless.
LIVE_TPU_ZONES = (
    "us-central2-a",
    "us-central2-b",
    "us-east5-a",
    "us-east5-b",
    "us-east5-c",
    "us-central1-a",
    "us-east1-d",
    "europe-west4-a",
    "europe-west4-b",
    "us-west4-a",
)

# Ray-era node names (``ray-marin-<token>-worker-...``) encode the region token
# but not the slice size; each regional Ray cluster was homogeneous, so the
# token maps deterministically to a TPU family (sizes stay unknown -> slice
# counts only, no chips, before the Iris era).
RAY_REGION_FAMILY: dict[str, str] = {
    "us-central2": "v4",
    "us-central2-vllm": "v4",
    "us-east5-a": "v5p",
    "us-east5-b": "v6e",
    "us-east5": "v6e",
    "us-central1": "v5p",
    "us-central1-vllm": "v5p",
    "eu-west4": "v5e",
    "eu-west4-a": "v6e",
    "eu-west4-vllm": "v5e",
    "us-east1": "v6e",
    "us-east1-d": "v6e",
}


def slice_chips(family: str, n: int) -> float | None:
    """Total chips in a sized slice ``<family>-<n>`` (None if family is unknown).

    The accelerator-type suffix ``n`` counts TensorCores for the older pod
    families (two per chip) and chips directly for the single-core lite pods:
    ``v2/v3/v4/v5p -> n/2``; ``v5e/v6e -> n``. Verified against live topology
    (e.g. ``v4-512`` = 4x8x8 = 256 chips).
    """
    if family in ("v2", "v3", "v4", "v5p"):
        return n / 2.0
    if family in ("v5e", "v6e"):
        return float(n)
    return None
