# Iris blog metrics pipeline

Reconstructs **TPU usage, active users, running tasks, and aggregate FLOPS over
time** from the Iris cluster's finelog archives, for the Marin blog post.

It is a three-step pipeline with on-disk artifacts between steps, so you can
re-run a later step (tweak a metric, restyle a chart) without redoing the
expensive fetch.

```
fetch    ──>  data/raw/      (multi-GB parquet + W&B cache)
extract  ──>  data/daily/    (small per-day CSVs)
charts   ──>  data/charts/   (SVG + PNG)
```

All of `data/` is gitignored — every artifact is reproducible from the scripts,
so nothing here is committed. Re-run the relevant step to regenerate.

## Data sources

Everything comes from the finelog `marin` deployment, archived at
`gs://marin-us-central2/finelog/marin`:

| Source | What we use it for |
|---|---|
| `iris.worker` | One row per worker heartbeat with `device_variant`, `running_task_count`, `zone`. Drives accelerators, FLOPS, concurrent tasks, regions. |
| `iris.task` | One row per task-attempt resource update; `task_id` is `/<user>/<job>/.../<idx>`, so the first path component gives the **submitting user**. Drives active users + distinct tasks. |
| `log` / `/system/controller` | The controller's audit lines (`event=job_submitted ...`). *Optional* — only needed for submission/demand series. The namespace is ~33 GB; fetching it is the slow path. |
| **W&B** `marin-community/marin` | One row per run (`num_devices`, `_runtime`, `parameter_count`, `throughput/total_tokens`, `backend`). The **only source reaching before Iris** (back to ~2024-04). Drives the long-history compute charts + calibration. |

**FLOPS** are computed from the slice `device_variant` (e.g. `v5p-32`):
`get_tpu_topology(variant).chips_per_vm` gives chips per worker VM, multiplied by
the peak bf16 FLOP/s per chip vendored from
`lib/fray/src/fray/device_flops.py` into `config.FAMILY_FLOPS_BF16`. This is
**provisioned (peak) capacity**, not realized utilization — `iris.task`'s
`accelerator_util_pct` is null across this window, so realized MFU is not
available here.

### Concurrency model

Heartbeats are dense and workers churn (preemption mints new worker ids), so raw
"distinct worker ids per day" overcounts badly. Instead each heartbeat marks its
VM **present in a 10-minute bucket**; one present worker contributes
`chips_per_vm` chips. Per day we report:

- **mean** = time-weighted average over the day's buckets (idle buckets count as
  zero), additive across families/regions so stacks are honest;
- **peak** = the single busiest bucket.

## Data window & history

The **Iris structured** charts (accelerators, users, tasks, regions) only exist
from **~2026-05-06** onward — the finelog Rust store was bootstrapped then and
earlier `iris.*` data was not carried over. The controller `log` namespace is
retention-capped only ~1 week earlier (~2026-04-29), and GCP has no billing
export in this project, so neither extends the structured charts.

The **W&B-derived** compute charts reach back to **~2024-04**, covering the
pre-Iris (Ray-era) history. They measure *realized training compute* from runs
that logged to W&B (`6 * parameter_count * total_tokens`), which:

- covers training + eval runs but **misses non-W&B work** (data/crawl jobs), so
  it is a compute-usage *proxy*, not the whole cluster;
- over-counts eval/inference (they do ~2ND, not 6ND);
- needs no device-type lookup (FLOPs come straight from N and D).

### Calibration

In the **overlap** (from `CALIBRATION_START`), the W&B realized series is
checked against the Iris-provisioned capacity (`calibration_daily.csv`):

- **coverage** = W&B-TPU device-hours / Iris provisioned device-hours — what
  fraction of the standing fleet the W&B-logged jobs occupied;
- **effective MFU** = W&B realized FLOPs / Iris peak-capacity FLOPs — realized
  vs theoretical peak. The extract step logs both headline ratios.

## Running

Run from `lib/iris`. `extract` uses the iris env (duckdb is already a dep);
`fetch` additionally needs `wandb` (and `WANDB_API_KEY`), `charts` needs
matplotlib.

```bash
# 1. Mirror iris.worker + iris.task parquet (~10 GB) + pull W&B run history
#    (~tens of thousands of runs, minutes; cached, so re-runs skip it).
uv run --with wandb python scripts/blog_metrics/pipeline.py fetch

# 2. Roll up into per-day CSVs (fast; re-run freely while tweaking metrics).
uv run python scripts/blog_metrics/pipeline.py extract

# 3. Render charts.
uv run --with matplotlib python scripts/blog_metrics/pipeline.py charts

# …or all three:
uv run --with wandb --with matplotlib python scripts/blog_metrics/pipeline.py all
```

Pass `--no-wandb` to skip the W&B pull (Iris-only charts), or `--force` to
refresh both caches.

Optional controller-log extraction (slow ~33 GB remote scan), which adds
`submissions_daily.csv` (jobs/tasks submitted, distinct submitting users):

```bash
uv run python scripts/blog_metrics/pipeline.py fetch --with-controller
```

`--data-dir DIR` relocates all artifacts; `--force` makes `fetch` pass
`rsync -d` to prune local segments dropped remotely.

## Outputs

`data/daily/`:

- `accelerators_daily.csv` — total chips / workers / PFLOP/s / running tasks (mean + peak).
- `accelerators_by_family_daily.csv` — mean chips / PFLOP/s / workers per TPU family.
- `accelerators_by_region_daily.csv` — mean chips per GCP region.
- `users_daily.csv` — active users (raw + human), distinct tasks run.
- `iris_capacity_daily.csv` — provisioned device-hours + peak-capacity FLOPs (calibration denominator).
- `wandb_compute_daily.csv` / `..._by_backend_daily.csv` — realized device-hours + FLOPs back to 2024.
- `calibration_daily.csv` — overlap-window coverage + effective MFU.
- `submissions_daily.csv` — only with `--with-controller`.

`data/charts/` (`.svg` for the blog, `.png` for preview): `accelerators_chips`,
`flops_pflops`, `users`, `tasks`, `accelerators_by_region`, `overview`,
`compute_history` (2024→now, + cumulative), `calibration`. Time-axis charts
annotate the `config.MILESTONES` that fall inside the window.

## Editing

- **Metrics / definitions** → `extract.py` (DuckDB SQL).
- **FLOPS table, bucket size, milestones, bot-user filter** → `config.py`.
- **Chart styling** → `charts.py`.
