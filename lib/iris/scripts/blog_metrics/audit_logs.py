# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP audit-log source: long-history TPU provisioning (back to ~2025-05).

The finelog/W&B sources only reach the Iris era (~2026-05-06). GCP's
Admin-Activity audit logs in the locked ``_Required`` bucket retain 400 days and
record every TPU ``CreateNode``/``DeleteNode`` and
``CreateQueuedResource``/``DeleteQueuedResource``, so they reconstruct
*provisioned* TPU capacity ~13 months back — the only native GCP source that
does, since this project's billing export is inaccessible.

Two entry points, mirroring ``wandb_history`` (one source, driven by the
fetch/extract orchestrators):

* ``fetch_events`` (fetch step) shells out to ``gcloud logging read`` once per
  month window for each event kind, plus ``gcloud compute tpus tpu-vm list`` for
  the current fleet, into ``raw/audit/``. Cached; pass ``force`` to refresh.
* ``write_daily`` (extract step) matches create→delete by node name,
  reconstructs each slice's lifetime, and integrates provisioned chip-hours and
  slice-hours per day into the daily CSVs.

Reconstruction gotchas handled (each silently breaks the count):

* failed creates (~77%, zone stockouts) are excluded server-side (``NOT
  severity>=ERROR``);
* older months emit the ``v2alpha1`` API, not ``v2`` (a v2-only pull undercounts
  ~50x) — see ``config.AUDIT_*_METHODS``;
* reserved capacity is provisioned via ``CreateQueuedResource`` (no
  ``CreateNode``);
* ~11% of preemptible nodes have no ``DeleteNode`` (GCP preemption) → cap
  create-only orphans at the class-median lifetime, except nodes still in the
  live fleet (clamped to "now") — clamping all orphans to now inflated ~30x.

Ray-era limitation: ``ray-marin-<token>-worker-...`` names encode the region
token (→ family, since each regional cluster was homogeneous) but **not** the
slice size, and the audit request/response payloads are empty, so chips are
unrecoverable before ~Mar 2026; those nodes contribute to the slice-count series
only.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import config

logger = logging.getLogger(__name__)

# Parallel gcloud calls (one per month window); matches the original shell -P 5.
_FETCH_WORKERS = 5

# Sized Iris-era names: marin-tpu-v5p-256-... / iris-tpu_v5e_32-... (- or _ seps).
SIZED_RE = re.compile(r"tpu[-_](v\d+[a-z]?)[-_](?:[a-z]+[-_])*?(\d+)(?:[-_]|$)")
RAY_RE = re.compile(r"^ray-marin-(.+?)-(?:vllm-)?worker-")


# --------------------------------------------------------------------------- #
# Fetch (gcloud)
# --------------------------------------------------------------------------- #
def _month_windows(start_iso: str, now: datetime) -> list[tuple[str, str]]:
    """(start, end) RFC3339 pairs splitting [start_iso, now] on month boundaries."""
    start = datetime.fromisoformat(start_iso).replace(tzinfo=UTC)
    edges: list[datetime] = [start]
    cur = datetime(start.year, start.month, 1, tzinfo=UTC)
    while True:
        cur = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=UTC)
        if cur >= now:
            break
        edges.append(cur)
    edges.append(now)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return [(edges[i].strftime(fmt), edges[i + 1].strftime(fmt)) for i in range(len(edges) - 1)]


def _logging_read(methods: tuple[str, ...], start: str, end: str) -> list[tuple[str, str]]:
    """Successful (timestamp, resourceName) completions of ``methods`` in [start, end)."""
    method_clause = " OR ".join(f'"{m}"' for m in methods)
    filter_expr = (
        f"protoPayload.methodName=({method_clause}) AND operation.last=true "
        f'AND NOT severity>=ERROR AND timestamp>="{start}" AND timestamp<"{end}"'
    )
    cmd = [
        "gcloud",
        "logging",
        "read",
        filter_expr,
        f"--project={config.GCP_PROJECT}",
        "--limit=2000000",
        "--format=csv[no-heading](timestamp,protoPayload.resourceName)",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        ts, sep, resource = line.partition(",")
        if sep and resource:
            rows.append((ts, resource))
    return rows


def _fetch_kind(methods: tuple[str, ...], windows: list[tuple[str, str]], label: str) -> list[tuple[str, str]]:
    """Pull one event kind across all month windows in parallel."""
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        per_window = list(pool.map(lambda w: _logging_read(methods, w[0], w[1]), windows))
    rows = [r for window_rows in per_window for r in window_rows]
    logger.info("  %s: %d events across %d windows", label, len(rows), len(windows))
    return rows


def _write_events(path: str, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _fetch_live_fleet(paths: config.Paths) -> None:
    """Snapshot current live TPU node names across the active zones (best-effort)."""
    names: list[str] = []
    for zone in config.LIVE_TPU_ZONES:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={config.GCP_PROJECT}",
            f"--zone={zone}",
            "--format=value(name)",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.warning("live-fleet list failed for %s (skipping): %s", zone, exc.stderr.strip())
            continue
        names.extend(line for line in proc.stdout.splitlines() if line)
    with open(paths.live_tpus_csv, "w", newline="") as fh:
        csv.writer(fh).writerows([name] for name in names)
    logger.info("live fleet: %d TPU nodes across %d zones", len(names), len(config.LIVE_TPU_ZONES))


def fetch_events(paths: config.Paths, *, force: bool = False) -> None:
    """Pull TPU create/delete audit events + the live fleet into ``raw/audit/``."""
    os.makedirs(paths.audit_raw_dir, exist_ok=True)
    creates = os.path.join(paths.audit_raw_dir, "creates.csv")
    if os.path.exists(creates) and not force:
        logger.info("audit event cache present (%s); skipping pull (use --force to refresh)", paths.audit_raw_dir)
        return

    now = datetime.now(UTC)
    windows = _month_windows(config.AUDIT_WINDOW_START, now)
    logger.info("pulling TPU audit events %s → %s (%d windows)", config.AUDIT_WINDOW_START, now.date(), len(windows))
    for fname, methods, label in (
        ("creates.csv", config.AUDIT_NODE_CREATE_METHODS, "CreateNode"),
        ("deletes.csv", config.AUDIT_NODE_DELETE_METHODS, "DeleteNode"),
        ("qr_creates.csv", config.AUDIT_QR_CREATE_METHODS, "CreateQueuedResource"),
        ("qr_deletes.csv", config.AUDIT_QR_DELETE_METHODS, "DeleteQueuedResource"),
    ):
        _write_events(os.path.join(paths.audit_raw_dir, fname), _fetch_kind(methods, windows, label))
    _fetch_live_fleet(paths)


# --------------------------------------------------------------------------- #
# Reconstruct (event matching → daily chip/slice hours)
# --------------------------------------------------------------------------- #
def _parse_resource(resource: str) -> tuple[str, str, str] | None:
    """(zone, region, node_name) from a ``.../locations/<zone>/{nodes,queuedResources}/<name>``."""
    m = re.search(r"/locations/([^/]+)/(?:nodes|queuedResources)/(.+)$", resource)
    if not m:
        return None
    zone = m.group(1)
    region = re.sub(r"-[a-z]$", "", zone)
    return zone, region, m.group(2)


def _parse_ts(s: str) -> datetime:
    s = s.strip().rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"  # gcloud emits nanoseconds; datetime takes micros
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=UTC)
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def _classify(name: str, region: str) -> tuple[str, float | None]:
    """(family, chips) for a node; chips is None when the size is unknown (Ray era)."""
    if name.startswith("ray-marin"):
        m = RAY_RE.match(name)
        token = m.group(1) if m else region
        family = config.RAY_REGION_FAMILY.get(token) or config.RAY_REGION_FAMILY.get(region) or "ray?"
        return family, None
    m = SIZED_RE.search(name)
    if m:
        family = m.group(1)
        return family, config.slice_chips(family, int(m.group(2)))
    if name.startswith("tpu-ci") or name.startswith("ci-"):
        return "ci?", None
    return "other?", None


def _life_class(name: str) -> str:
    """Coarse class whose median observed lifetime caps that class's missed-delete orphans."""
    if "preemptible" in name:
        return "preemptible"
    if "reserved" in name:
        return "reserved"
    if name.startswith("ray"):
        return "ray"
    return "other"


def _add_interval(acc: dict, start: datetime, end: datetime, key: tuple, weight: float) -> None:
    """Spread ``weight`` (chips or 1 slice) over each calendar day the interval touches."""
    if end <= start:
        return
    cur = start
    while cur < end:
        day = cur.date()
        day_end = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
        seg_end = min(end, day_end)
        hours = (seg_end - cur).total_seconds() / 3600.0
        acc[(day.isoformat(), *key)] += hours * weight
        cur = seg_end


def _load(path: str, keep: str) -> tuple[dict[str, datetime], dict[str, tuple[str, str]]]:
    """node_name → (kept) timestamp and (zone, region); ``keep`` is "min" (create) or "max" (delete)."""
    times: dict[str, datetime] = {}
    meta: dict[str, tuple[str, str]] = {}
    if not os.path.exists(path):
        return times, meta
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            parsed = _parse_resource(row[1])
            if not parsed:
                continue
            zone, region, name = parsed
            try:
                t = _parse_ts(row[0])
            except ValueError:
                continue
            if name not in times:
                times[name] = t
                meta[name] = (zone, region)
            else:
                times[name] = min(times[name], t) if keep == "min" else max(times[name], t)
    return times, meta


def _merge(base_t: dict, base_m: dict, other_t: dict, other_m: dict, keep: str) -> None:
    for name, t in other_t.items():
        if name not in base_t:
            base_t[name] = t
            base_m[name] = other_m[name]
        else:
            base_t[name] = min(base_t[name], t) if keep == "min" else max(base_t[name], t)


def reconstruct(paths: config.Paths) -> tuple[dict, dict]:
    """Integrate provisioned chip-hours and slice-hours per (day, region, family).

    Returns ``(chip_hours, slice_hours)``; ``chip_hours`` covers only sized
    (Iris-era) nodes, ``slice_hours`` counts every node regardless of known size.
    """
    create_t, create_m = _load(os.path.join(paths.audit_raw_dir, "creates.csv"), "min")
    qc_t, qc_m = _load(os.path.join(paths.audit_raw_dir, "qr_creates.csv"), "min")
    _merge(create_t, create_m, qc_t, qc_m, "min")
    delete_t, delete_m = _load(os.path.join(paths.audit_raw_dir, "deletes.csv"), "max")
    qd_t, qd_m = _load(os.path.join(paths.audit_raw_dir, "qr_deletes.csv"), "max")
    _merge(delete_t, delete_m, qd_t, qd_m, "max")

    live: set[str] = set()
    if os.path.exists(paths.live_tpus_csv):
        with open(paths.live_tpus_csv, newline="") as fh:
            live = {row[0] for row in csv.reader(fh) if row}

    # Class-median observed lifetime, used to cap missed-delete orphans.
    matched: dict[str, list[float]] = defaultdict(list)
    for name, t in create_t.items():
        if name in delete_t and delete_t[name] > t:
            matched[_life_class(name)].append((delete_t[name] - t).total_seconds() / 3600.0)
    median = {cls: statistics.median(v) for cls, v in matched.items()}

    window_start = datetime.fromisoformat(config.AUDIT_WINDOW_START).replace(tzinfo=UTC)
    now = datetime.now(UTC)
    chip_hours: dict = defaultdict(float)
    slice_hours: dict = defaultdict(float)
    capped = clamped = 0

    for name in set(create_t) | set(delete_t):
        _zone, region = create_m.get(name) or delete_m.get(name) or ("unknown", "unknown")
        family, chips = _classify(name, region)
        cap = timedelta(hours=median.get(_life_class(name), 1.0))
        has_create, has_delete = name in create_t, name in delete_t
        if has_create and has_delete:
            start, end = create_t[name], delete_t[name]
        elif has_create:  # missed delete: alive → now, else preempted → class-median cap
            start = create_t[name]
            if name in live:
                end = now
                clamped += 1
            else:
                end = start + cap
                capped += 1
        else:  # delete only: created before our capture window → cap backwards
            end = delete_t[name]
            start = max(window_start, end - cap)

        _add_interval(slice_hours, start, end, (region, family), 1.0)
        if chips is not None:
            _add_interval(chip_hours, start, end, (region, family), chips)

    logger.info(
        "reconstructed %d nodes (%d alive→now, %d capped missed-deletes); class-median lifetimes (h): %s",
        len(set(create_t) | set(delete_t)),
        clamped,
        capped,
        {cls: round(v, 2) for cls, v in median.items()},
    )
    return chip_hours, slice_hours


def _write_rollup(rollup: dict, path: str, group_index: int, group_col: str, value_col: str) -> None:
    """Collapse a (day, region, family)->hours accumulator on one axis and write avg-concurrent CSV.

    ``group_index`` selects region (1) or family (2); the daily value is hours/24
    = mean concurrent over the calendar day, additive across groups (stackable).
    """
    daily: dict[tuple[str, str], float] = defaultdict(float)
    for (day, region, family), hours in rollup.items():
        group = (region, family)[group_index - 1]
        daily[(day, group)] += hours
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", group_col, value_col])
        for (day, group), hours in sorted(daily.items()):
            writer.writerow([day, group, round(hours / 24.0, 4)])
    logger.info("wrote %s", path)


def write_daily(paths: config.Paths) -> None:
    """Write the four long-history provisioning CSVs from the reconstructed intervals."""
    chip_hours, slice_hours = reconstruct(paths)
    daily = paths.daily_dir
    _write_rollup(chip_hours, os.path.join(daily, "tpu_provisioning_by_family_daily.csv"), 2, "family", "mean_chips")
    _write_rollup(chip_hours, os.path.join(daily, "tpu_provisioning_by_region_daily.csv"), 1, "region", "mean_chips")
    _write_rollup(slice_hours, os.path.join(daily, "tpu_slices_by_family_daily.csv"), 2, "family", "mean_slices")
    _write_rollup(slice_hours, os.path.join(daily, "tpu_slices_by_region_daily.csv"), 1, "region", "mean_slices")
