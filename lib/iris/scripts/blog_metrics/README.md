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
that logged to W&B as `6 * parameter_count * total_tokens`, **capped at each
run's physical hardware capacity** (`num_devices * runtime * peak * mfu`,
`config.REALIZED_FLOPS_CEILING_*`). The cap matters a lot:

- W&B's `throughput/total_tokens` (D) is a **cumulative counter that resumed and
  cooldown runs inherit** from their parent. Uncapped, `6*N*D` counts a flagship
  lineage many times over — a 1-hour 32B cooldown re-reports the parent's ~6T
  tokens, so naive `6*N*D` charges it ~1e24 FLOPs (an implied **>1000x MFU**).
  Summed over all runs that was ~47e24 FLOPs — physically impossible against a
  fleet that peaked at 2048 chips.
- Capping each run at the most its *own* `num_devices * runtime` could deliver
  bounds the corrupted runs at hardware reality while leaving honest runs (whose
  `6*N*D` is already below the ceiling) at their exact value. The corrected total
  is **~3e24 FLOPs** — a few final-runs-worth. Because the bound uses each run's
  own device count, it honors a fleet whose size varied over time.
- It still **misses non-W&B work** (data/crawl jobs) and projects that don't log
  token counts, so it is a compute-usage *proxy*, not the whole cluster.

Projects pulled: the flagship `marin` project plus the device-bearing sibling
projects that log N and D — the MoE training projects (`dial_moe`, `marin_moe`)
and the two largest optimizer sweeps (`optimizer-scaling`, `Hyperball`). Eval /
post-training projects log devices but no token counts, so they can't form
`6*N*D` and are omitted (`config.WANDB_PROJECTS`).

### Calibration

The W&B realized series is checked against the Iris-provisioned capacity over
the window where **all training actually ran on Iris** (`CALIBRATION_START`,
June onward). Earlier is apples-to-oranges: April's v4 runs were still on Ray
(no Iris logs), and in early-bootstrap May some runs ran elsewhere — W&B
device-hours *exceed* Iris's on May 7–8 (an impossible >100% coverage), which is
exactly why the window starts in June.

Two ratios, kept **separate** because their product is misleading
(`calibration_daily.csv`):

- **coverage** = W&B-logged-TPU device-hours / Iris device-hours — what fraction
  of the standing fleet ran W&B-logged *training*. This is well under 100% (~16%
  in June) because most fleet work is non-training / not-6ND-loggable (data
  jobs, RL, evals);
- **on-active MFU** = realized FLOPs per W&B device-hour ÷ fleet peak per
  device-hour — the efficiency of the logged jobs *while they ran* (~32% in
  June, steady).

Their product, "effective MFU vs the whole fleet", craters whenever coverage
dips (the big multi-day MoE runs ended ~May 19, so it falls ~30× into June) even
though on-active MFU is steady — so we report on-active MFU as the efficiency
headline and coverage separately, and the chart plots on-active MFU only.

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
- `intraday_regions.csv` — concurrent chips per region at fine (30-min) buckets for `config.INTRADAY_CANDIDATE_DAY`, to show within-day cross-region migration.
- `utilization_daily.csv` — fleet occupancy: mean busy vs idle chips and the busy fraction (chips running ≥1 task / chips provisioned). Allocation, not efficiency — the MFU side is `calibration_daily.csv`.
- `users_daily.csv` — active users (raw + human), distinct tasks run.
- `iris_capacity_daily.csv` — provisioned device-hours + peak-capacity FLOPs (calibration denominator).
- `wandb_compute_daily.csv` / `..._by_backend_daily.csv` — realized device-hours + FLOPs back to 2024.
- `calibration_daily.csv` — June-onward coverage + on-active MFU (all-on-iris window).
- `submissions_daily.csv` — only with `--with-controller`.

`data/charts/` (`.svg` for the blog, `.png` for preview): `accelerators_chips`,
`flops_pflops`, `users`, `tasks`, `fleet_utilization` (occupancy: busy vs idle
chips), `preemptible` (preemptible non-v4 vs reserved v4 capacity), `accelerators_by_region`,
`intraday_regions` (within-day cross-region migration for the candidate day),
`overview`, `compute_history` (2024→now, + cumulative), `calibration`. Only the
wide `compute_history` annotates the `config.MILESTONES`; the cramped
Iris-window charts omit them.

The `preemptible` chart needs the real per-device generation, which only the
iris structured data carries (~May 6 on) — W&B logs `num_devices` but not the
TPU type, so the v4-vs-non-v4 split **cannot** be reconstructed for the pre-iris
(Jan–Apr) history.

## Editing

- **Metrics / definitions** → `extract.py` (DuckDB SQL).
- **FLOPS table, bucket size, milestones, bot-user filter** → `config.py`.
- **Chart styling** → `charts.py`.
