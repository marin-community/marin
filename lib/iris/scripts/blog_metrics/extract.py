# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline step 2: roll the raw parquet up into small per-day CSVs.

All aggregation runs in DuckDB over the locally-mirrored parquet (no network),
so this step is cheap to re-run while iterating on metric definitions. Outputs
land in ``<data_dir>/daily`` as per-day CSVs:

* ``accelerators_daily.csv`` — total concurrent chips / workers / PFLOP/s /
  running tasks per day (mean = time-weighted over 10-min buckets, peak =
  busiest bucket).
* ``accelerators_by_family_daily.csv`` — the same mean concurrency split by TPU
  family (additive across families -> stackable).
* ``users_daily.csv`` — distinct active users and tasks that actually ran.
* ``submissions_daily.csv`` — *only if* controller audit lines were fetched:
  jobs/tasks submitted and distinct submitting users per day.

Concurrency model: each ``iris.worker`` heartbeat marks its VM present in a
10-min bucket; one worker contributes ``chips_per_vm`` chips. Summing distinct
present workers per bucket and averaging over the day's buckets gives a
time-weighted mean concurrent footprint that is robust to heartbeat cadence and
to preemption churn within a day.
"""

from __future__ import annotations

import glob
import logging
import os

import audit_logs
import config
import duckdb

logger = logging.getLogger(__name__)


def _register_variant_info(con: duckdb.DuckDBPyConnection) -> None:
    """Create the variant -> (family, chips_per_vm, flops_per_chip) lookup table."""
    con.execute(
        "CREATE TABLE variant_info ("
        " variant VARCHAR, family VARCHAR, family_display VARCHAR,"
        " chips_per_vm INTEGER, flops_per_chip DOUBLE)"
    )
    con.executemany(
        "INSERT INTO variant_info VALUES (?, ?, ?, ?, ?)",
        [
            (vi.variant, vi.family, vi.family_display, vi.chips_per_vm, vi.flops_per_chip)
            for vi in config.build_variant_table()
        ],
    )


def _log_variant_coverage(con: duckdb.DuckDBPyConnection, worker_glob: str) -> None:
    """Warn about TPU variants present in the data but missing from the FLOPS table."""
    rows = con.execute(
        f"""
        SELECT lower(device_variant) AS variant, count(DISTINCT worker_id) AS workers
        FROM read_parquet('{worker_glob}')
        WHERE lower(device_type) = 'tpu' AND device_variant <> ''
          AND lower(device_variant) NOT IN (SELECT variant FROM variant_info)
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    if rows:
        logger.warning("dropping %d unmapped TPU variant(s): %s", len(rows), rows)


def _build_bucket_tables(con: duckdb.DuckDBPyConnection, worker_glob: str) -> None:
    """Materialize per-bucket concurrency tables shared by the accelerator rollups."""
    bucket = config.CONCURRENCY_BUCKET
    # Distinct (worker, variant) presence per time bucket. device_variant is
    # static per worker, so this is "which VMs emitted a heartbeat this bucket".
    con.execute(
        f"""
        CREATE TEMP TABLE hb AS
        SELECT worker_id, lower(device_variant) AS variant,
               time_bucket(INTERVAL '{bucket}', ts) AS bucket
        FROM read_parquet('{worker_glob}')
        WHERE lower(device_type) = 'tpu' AND device_variant <> ''
        GROUP BY 1, 2, 3
        """
    )
    # Per (bucket, family): concurrent workers, chips, PFLOP/s capacity.
    con.execute(
        """
        CREATE TEMP TABLE bucket_family AS
        SELECT h.bucket, vi.family_display AS family,
               count(*) AS workers,
               sum(vi.chips_per_vm) AS chips,
               sum(vi.chips_per_vm * vi.flops_per_chip) / 1e15 AS pflops
        FROM hb h JOIN variant_info vi ON h.variant = vi.variant
        GROUP BY 1, 2
        """
    )
    # Per-bucket totals across all families.
    con.execute(
        """
        CREATE TEMP TABLE bucket_total AS
        SELECT bucket, sum(workers) AS workers, sum(chips) AS chips, sum(pflops) AS pflops
        FROM bucket_family GROUP BY 1
        """
    )
    # Number of populated buckets per day -> denominator for time-weighted means.
    con.execute(
        """
        CREATE TEMP TABLE day_buckets AS
        SELECT bucket::DATE AS day, count(*) AS nb
        FROM bucket_total GROUP BY 1
        """
    )
    # Per-bucket concurrent running tasks, from the host-level running_task_count.
    con.execute(
        f"""
        CREATE TEMP TABLE bucket_tasks AS
        SELECT bucket, sum(rtc) AS tasks FROM (
            SELECT time_bucket(INTERVAL '{bucket}', ts) AS bucket, worker_id,
                   max(running_task_count) AS rtc
            FROM read_parquet('{worker_glob}')
            GROUP BY 1, 2
        ) GROUP BY 1
        """
    )
    # Per bucket: provisioned chips split into busy (worker running >=1 task)
    # vs idle (running_task_count == 0). Drives the fleet-occupancy chart — the
    # real "are accelerators doing work" signal, independent of W&B.
    con.execute(
        f"""
        CREATE TEMP TABLE bucket_occupancy AS
        SELECT wr.bucket,
               sum(vi.chips_per_vm) AS chips_total,
               sum(CASE WHEN wr.rtc > 0 THEN vi.chips_per_vm ELSE 0 END) AS chips_busy
        FROM (
            SELECT time_bucket(INTERVAL '{bucket}', ts) AS bucket, worker_id,
                   lower(device_variant) AS variant, max(running_task_count) AS rtc
            FROM read_parquet('{worker_glob}')
            WHERE lower(device_type) = 'tpu' AND device_variant <> ''
            GROUP BY 1, 2, 3
        ) wr JOIN variant_info vi ON wr.variant = vi.variant
        GROUP BY 1
        """
    )
    # Per (bucket, region): concurrent chips, for the regional footprint chart.
    # Region = zone with its trailing "-<letter>" stripped (us-east5-a -> us-east5).
    con.execute(
        """
        CREATE TEMP TABLE bucket_region AS
        SELECT h.bucket, regexp_replace(coalesce(w.zone, ''), '-[a-z]$', '') AS region,
               sum(vi.chips_per_vm) AS chips
        FROM hb h
        JOIN variant_info vi ON h.variant = vi.variant
        JOIN (SELECT DISTINCT worker_id, zone FROM read_parquet('{glob}')) w
          ON h.worker_id = w.worker_id
        GROUP BY 1, 2
        """.replace(
            "{glob}", worker_glob
        )
    )


def _write_accelerators(con: duckdb.DuckDBPyConnection, daily_dir: str) -> None:
    total_csv = os.path.join(daily_dir, "accelerators_daily.csv")
    con.execute(
        f"""
        COPY (
            SELECT bt.bucket::DATE AS day,
                   avg(bt.chips) AS mean_chips, max(bt.chips) AS peak_chips,
                   avg(bt.workers) AS mean_workers, max(bt.workers) AS peak_workers,
                   avg(bt.pflops) AS mean_pflops, max(bt.pflops) AS peak_pflops,
                   avg(coalesce(tk.tasks, 0)) AS mean_tasks,
                   max(coalesce(tk.tasks, 0)) AS peak_tasks
            FROM bucket_total bt
            LEFT JOIN bucket_tasks tk USING (bucket)
            GROUP BY 1 ORDER BY 1
        ) TO '{total_csv}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", total_csv)

    family_csv = os.path.join(daily_dir, "accelerators_by_family_daily.csv")
    # Divide by the day's bucket count so an idle bucket counts as zero for a
    # family (time-weighted mean, additive across families for stacking).
    con.execute(
        f"""
        COPY (
            SELECT bf.bucket::DATE AS day, bf.family,
                   sum(bf.chips) / max(db.nb) AS mean_chips,
                   sum(bf.pflops) / max(db.nb) AS mean_pflops,
                   sum(bf.workers) / max(db.nb) AS mean_workers
            FROM bucket_family bf
            JOIN day_buckets db ON bf.bucket::DATE = db.day
            GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{family_csv}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", family_csv)

    region_csv = os.path.join(daily_dir, "accelerators_by_region_daily.csv")
    con.execute(
        f"""
        COPY (
            SELECT br.bucket::DATE AS day, br.region,
                   sum(br.chips) / max(db.nb) AS mean_chips
            FROM bucket_region br
            JOIN day_buckets db ON br.bucket::DATE = db.day
            GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{region_csv}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", region_csv)


def _write_utilization(con: duckdb.DuckDBPyConnection, daily_dir: str) -> None:
    """Daily fleet occupancy: provisioned chips that were running a task vs idle.

    Occupancy (allocation) is distinct from efficiency (MFU): a chip with a task
    assigned counts as busy even if that task runs at low MFU. The calibration
    chart covers the MFU side; this covers the "is anything scheduled on it"
    side. Requires the ``bucket_occupancy`` temp table.
    """
    out = os.path.join(daily_dir, "utilization_daily.csv")
    con.execute(
        f"""
        COPY (
            SELECT bucket::DATE AS day,
                   avg(chips_busy) AS mean_busy_chips,
                   avg(chips_total - chips_busy) AS mean_idle_chips,
                   avg(chips_total) AS mean_total_chips,
                   sum(chips_busy) / nullif(sum(chips_total), 0) AS occupancy
            FROM bucket_occupancy GROUP BY 1 ORDER BY 1
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", out)
    summary = con.execute("SELECT sum(chips_busy) / nullif(sum(chips_total), 0) FROM bucket_occupancy").fetchone()
    if summary and summary[0] is not None:
        logger.info("fleet occupancy: %.1f%% of provisioned chip-time was running a task", 100 * summary[0])


def _write_intraday_regions(con: duckdb.DuckDBPyConnection, daily_dir: str, worker_glob: str) -> None:
    """Fine-grained per-region chips for one day, to show intraday cross-region movement.

    Buckets ``config.INTRADAY_CANDIDATE_DAY`` at ``config.INTRADAY_BUCKET`` and
    sums concurrent chips per region per bucket — the within-day analogue of the
    daily regional footprint. Heartbeats are ~10s apart so even 30-min buckets
    are densely populated.
    """
    day = config.INTRADAY_CANDIDATE_DAY
    out = os.path.join(daily_dir, "intraday_regions.csv")
    con.execute(
        f"""
        COPY (
            WITH hb AS (
                SELECT worker_id, lower(device_variant) AS variant,
                       regexp_replace(coalesce(zone, ''), '-[a-z]$', '') AS region,
                       time_bucket(INTERVAL '{config.INTRADAY_BUCKET}', ts) AS bucket
                FROM read_parquet('{worker_glob}')
                WHERE lower(device_type) = 'tpu' AND device_variant <> ''
                  AND ts::DATE = DATE '{day}'
                GROUP BY 1, 2, 3, 4
            )
            SELECT hb.bucket, hb.region, sum(vi.chips_per_vm) AS chips
            FROM hb JOIN variant_info vi ON hb.variant = vi.variant
            GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s (intraday regions for %s)", out, day)


def _write_users(con: duckdb.DuckDBPyConnection, daily_dir: str, task_glob: str) -> None:
    bots = ", ".join(f"'{u}'" for u in sorted(config.BOT_USERS))
    users_csv = os.path.join(daily_dir, "users_daily.csv")
    con.execute(
        f"""
        COPY (
            SELECT ts::DATE AS day,
                   count(DISTINCT u) AS active_users,
                   count(DISTINCT CASE WHEN u NOT IN ({bots}) THEN u END) AS active_users_human,
                   count(DISTINCT task_id) AS active_tasks_ran
            FROM (
                SELECT ts, task_id, split_part(ltrim(task_id, '/'), '/', 1) AS u
                FROM read_parquet('{task_glob}')
            )
            GROUP BY 1 ORDER BY 1
        ) TO '{users_csv}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", users_csv)


def _write_submissions(con: duckdb.DuckDBPyConnection, daily_dir: str, audit_parquet: str) -> None:
    bots = ", ".join(f"'{u}'" for u in sorted(config.BOT_USERS))
    subs_csv = os.path.join(daily_dir, "submissions_daily.csv")
    con.execute(
        rf"""
        COPY (
            SELECT day,
                   count(*) AS jobs_submitted,
                   sum(num_tasks) AS tasks_submitted,
                   count(DISTINCT u) AS submitting_users,
                   count(DISTINCT CASE WHEN u NOT IN ({bots}) THEN u END) AS submitting_users_human
            FROM (
                SELECT to_timestamp(epoch_ms / 1000)::DATE AS day,
                       split_part(ltrim(regexp_extract(data, 'entity=(\S+)', 1), '/'), '/', 1) AS u,
                       TRY_CAST(regexp_extract(data, 'num_tasks=(\d+)', 1) AS INTEGER) AS num_tasks
                FROM read_parquet('{audit_parquet}')
                WHERE data LIKE '%event=job_submitted%'
            )
            GROUP BY 1 ORDER BY 1
        ) TO '{subs_csv}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", subs_csv)


def _write_iris_capacity(con: duckdb.DuckDBPyConnection, daily_dir: str) -> None:
    """Daily *provisioned* footprint from iris: device-hours and peak-capacity FLOPs.

    These are what the cluster could deliver (workers present x peak per chip),
    the denominator the W&B realized series is calibrated against.
    """
    mins = config.CONCURRENCY_BUCKET_MINUTES
    con.execute(
        f"""
        CREATE TEMP TABLE iris_capacity_daily AS
        SELECT bucket::DATE AS day,
               sum(chips) * ({mins} / 60.0) AS device_hours,
               sum(pflops) * 1e15 * ({mins} * 60) AS capacity_flops
        FROM bucket_total GROUP BY 1 ORDER BY 1
        """
    )
    out = os.path.join(daily_dir, "iris_capacity_daily.csv")
    con.execute(f"COPY iris_capacity_daily TO '{out}' (HEADER, DELIMITER ',')")
    logger.info("wrote %s", out)


def _write_wandb_compute(con: duckdb.DuckDBPyConnection, daily_dir: str, wandb_parquet: str) -> None:
    """Daily *realized* compute from W&B runs: device-hours and capped 6*N*D FLOPs.

    Each run's FLOPs are estimated as ``6*N*D`` but **capped at the most its own
    hardware-time could deliver** (``num_devices * runtime * peak * mfu``),
    because ``total_tokens`` is a cumulative counter that resumed/cooldown runs
    inherit — see ``config.REALIZED_FLOPS_CEILING_*``. The capped run total and
    the device-hours are then spread uniformly across the wall-clock days the run
    spans, so a multi-day run lands on every day it touched. ``backend``
    separates TPU from GPU/CPU work.
    """
    k = config.TRAINING_FLOPS_PER_PARAM_TOKEN
    peak = config.REALIZED_FLOPS_CEILING_PEAK_BF16
    mfu = config.REALIZED_FLOPS_CEILING_MFU
    con.execute(
        f"""
        CREATE TEMP TABLE wandb_exploded AS
        WITH r AS (
            SELECT created_at AS start_ts,
                   created_at + (greatest(coalesce(runtime_s, 0), 1) * INTERVAL '1 second') AS end_ts,
                   greatest(coalesce(runtime_s, 0), 1) AS dur,
                   num_devices,
                   least({k} * coalesce(parameter_count, 0) * coalesce(total_tokens, 0),
                         num_devices * greatest(coalesce(runtime_s, 0), 1) * {peak} * {mfu}) AS run_flops,
                   lower(coalesce(backend, 'unknown')) AS backend
            FROM read_parquet('{wandb_parquet}')
            WHERE created_at IS NOT NULL AND num_devices > 0
        )
        SELECT r.backend, d::DATE AS day,
               num_devices * day_secs / 3600.0 AS device_hours,
               run_flops * day_secs / dur AS realized_flops
        FROM r,
             unnest(generate_series(date_trunc('day', start_ts),
                                    date_trunc('day', end_ts),
                                    INTERVAL '1 day')) AS g(d),
             LATERAL (SELECT greatest(0, date_diff('second', greatest(start_ts, d),
                                      least(end_ts, d + INTERVAL '1 day'))) AS day_secs) o
        WHERE day_secs > 0
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE wandb_compute_daily AS
        SELECT day,
               sum(device_hours) AS device_hours,
               sum(realized_flops) AS realized_flops,
               sum(CASE WHEN backend = 'tpu' THEN device_hours ELSE 0 END) AS device_hours_tpu,
               sum(CASE WHEN backend = 'tpu' THEN realized_flops ELSE 0 END) AS realized_flops_tpu
        FROM wandb_exploded GROUP BY 1 ORDER BY 1
        """
    )
    out = os.path.join(daily_dir, "wandb_compute_daily.csv")
    con.execute(f"COPY wandb_compute_daily TO '{out}' (HEADER, DELIMITER ',')")
    logger.info("wrote %s", out)

    by_backend = os.path.join(daily_dir, "wandb_compute_by_backend_daily.csv")
    con.execute(
        f"""
        COPY (
            SELECT day, backend, sum(device_hours) AS device_hours, sum(realized_flops) AS realized_flops
            FROM wandb_exploded GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{by_backend}' (HEADER, DELIMITER ',')
        """
    )
    logger.info("wrote %s", by_backend)


def _write_calibration(con: duckdb.DuckDBPyConnection, daily_dir: str) -> None:
    """Calibrate W&B realized compute against iris provisioned capacity in the overlap.

    Requires the ``iris_capacity_daily`` and ``wandb_compute_daily`` temp tables.
    Two distinct ratios, kept separate because conflating them is misleading:

    * **coverage** = W&B-logged-TPU device-hours / iris device-hours — what
      fraction of the standing fleet ran W&B-logged *training* (the rest is
      non-training / non-logged work, so this is well below 100%);
    * **on-active MFU** = realized FLOPs per W&B device-hour / fleet peak FLOPs
      per device-hour — the efficiency of the logged jobs *while they ran*,
      independent of how much of the fleet they occupied.

    Their product is the "effective MFU vs total fleet", which collapses whenever
    coverage drops even though efficiency is steady — so we report on-active MFU
    as the efficiency headline and coverage separately.
    """
    start = config.CALIBRATION_START
    con.execute(
        f"""
        CREATE TEMP TABLE calibration_daily AS
        SELECT i.day,
               w.device_hours_tpu AS wandb_tpu_device_hours,
               i.device_hours AS iris_device_hours,
               w.device_hours_tpu / nullif(i.device_hours, 0) AS device_hours_coverage,
               w.realized_flops_tpu AS wandb_tpu_realized_flops,
               i.capacity_flops AS iris_capacity_flops,
               w.realized_flops_tpu / nullif(i.capacity_flops, 0) AS effective_mfu,
               w.realized_flops_tpu * i.device_hours
                 / nullif(i.capacity_flops * w.device_hours_tpu, 0) AS on_active_mfu
        FROM iris_capacity_daily i
        JOIN wandb_compute_daily w USING (day)
        WHERE i.day >= DATE '{start}'
        ORDER BY i.day
        """
    )
    out = os.path.join(daily_dir, "calibration_daily.csv")
    con.execute(f"COPY calibration_daily TO '{out}' (HEADER, DELIMITER ',')")
    logger.info("wrote %s", out)

    summary = con.execute(
        """
        SELECT sum(wandb_tpu_device_hours) / nullif(sum(iris_device_hours), 0) AS coverage,
               sum(wandb_tpu_realized_flops) * sum(iris_device_hours)
                 / nullif(sum(iris_capacity_flops) * sum(wandb_tpu_device_hours), 0) AS on_active_mfu
        FROM calibration_daily
        """
    ).fetchone()
    if summary and summary[0] is not None:
        logger.info(
            "calibration (all-on-iris window >= %s): W&B-logged training = %.1f%% of fleet device-hours "
            "(coverage); those jobs ran at %.1f%% MFU while active (on-active MFU)",
            start,
            100 * summary[0],
            100 * (summary[1] or 0),
        )


def run(paths: config.Paths) -> None:
    """Compute every daily rollup CSV from the locally-mirrored raw inputs.

    Each source's rollups are gated on its raw inputs being present, so a partial
    fetch (e.g. ``--no-wandb``, or audit-only when finelog GCS is unreachable)
    still produces the CSVs it can.
    """
    os.makedirs(paths.daily_dir, exist_ok=True)
    worker_glob = os.path.join(paths.worker_parquet, "*.parquet")
    task_glob = os.path.join(paths.task_parquet, "*.parquet")

    con = duckdb.connect()
    if glob.glob(worker_glob):
        _register_variant_info(con)
        _log_variant_coverage(con, worker_glob)
        _build_bucket_tables(con, worker_glob)
        _write_accelerators(con, paths.daily_dir)
        _write_utilization(con, paths.daily_dir)
        _write_intraday_regions(con, paths.daily_dir, worker_glob)
        _write_users(con, paths.daily_dir, task_glob)
        _write_iris_capacity(con, paths.daily_dir)

        if os.path.exists(paths.wandb_runs_parquet):
            _write_wandb_compute(con, paths.daily_dir, paths.wandb_runs_parquet)
            _write_calibration(con, paths.daily_dir)
        else:
            logger.info("no W&B run cache; skipping long-history compute + calibration")

        if os.path.exists(paths.controller_audit_parquet):
            _write_submissions(con, paths.daily_dir, paths.controller_audit_parquet)
        else:
            logger.info("no controller audit parquet; skipping submissions_daily.csv")
    else:
        logger.info("no iris.worker parquet; skipping finelog rollups (run `fetch` for the Iris-era charts)")
    con.close()

    if os.path.exists(os.path.join(paths.audit_raw_dir, "creates.csv")):
        audit_logs.write_daily(paths)
    else:
        logger.info("no audit events; skipping long-history TPU provisioning rollups")
