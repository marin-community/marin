# Iris blog metrics pipeline

Reconstructs **TPU usage, active users, running tasks, and aggregate FLOPS over
time** for the Marin blog post, from three sources with complementary reach:
finelog archives (detailed per-heartbeat occupancy, Iris era), W&B run history
(realized training compute back to ~2024), and GCP Admin-Activity audit logs
(provisioned TPU capacity back to ~2025-05 — the long-history backbone).

It is a three-step pipeline with on-disk artifacts between steps, so you can
re-run a later step (tweak a metric, restyle a chart) without redoing the
expensive fetch.

```
fetch    ──>  data/raw/      (multi-GB parquet + W&B cache + audit-event CSVs)
extract  ──>  data/daily/    (small per-day CSVs)
charts   ──>  data/charts/   (SVG + PNG)
```

All of `data/` is gitignored — every artifact is reproducible from the scripts,
so nothing here is committed. Re-run the relevant step to regenerate.

## Data sources

The finelog rows come from the `marin` deployment archived at
`gs://marin-us-central2/finelog/marin`; W&B and the GCP audit logs are pulled
from their respective APIs:

| Source | What we use it for |
|---|---|
| `iris.worker` | One row per worker heartbeat with `device_variant`, `running_task_count`, `zone`. Drives accelerators, FLOPS, concurrent tasks, regions. |
| `iris.task` | One row per task-attempt resource update; `task_id` is `/<user>/<job>/.../<idx>`, so the first path component gives the **submitting user**. Drives active users + distinct tasks. |
| `log` / `/system/controller` | The controller's audit lines (`event=job_submitted ...`). *Optional* — only needed for submission/demand series. The namespace is ~33 GB; fetching it is the slow path. |
| **W&B** `marin-community/marin` | One row per run (`num_devices`, `_runtime`, `parameter_count`, `throughput/total_tokens`, `backend`). The **only source for realized compute before Iris** (back to ~2024-04). Drives the long-history compute charts + calibration. |
| **GCP Admin-Activity audit logs** (`hai-gcp-models`) | Every TPU `Create/DeleteNode` and `Create/DeleteQueuedResource` completion, from the locked `_Required` log bucket (400-day retention). Reconstructs **provisioned TPU capacity back to ~2025-05** — the only native source reaching that far, since the billing export is inaccessible (Stanford-managed billing account). See `audit_logs.py`. |

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
retention-capped only ~1 week earlier (~2026-04-29). For TPU *provisioning* the
long history instead comes from the GCP audit logs (next section), which reach
back ~13 months; the billing export that would give exact $/SKU is inaccessible.

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

### TPU provisioning (audit-log reconstruction)

`audit_logs.py` reconstructs *provisioned* TPU capacity (what GCP bills:
creation→deletion) from Admin-Activity audit events. Each successful
`CreateNode`/`CreateQueuedResource` is matched to its `Delete…` by the unique
node name; the lifetime is integrated into chip-hours and slice-hours per day,
with type+size parsed from the node name and zone from the resource path. Chips:
`v2/v3/v4/v5p → N/2`, `v5e/v6e → N` (verified against live topology, e.g.
`v4-512` = 256 chips). Four non-obvious gotchas, each of which silently breaks
the count if missed:

1. **~77% of `CreateNode` events are failed stockouts** ("no capacity in the
   zone"); they never bill, so we keep only successful completions (`operation.last`,
   `NOT severity>=ERROR`).
2. **Older months emit the `v2alpha1` API**, not `v2` — a v2-only pull
   under-counted Dec-2025 by ~50× (`config.AUDIT_*_METHODS` lists both, plus `v1`).
3. **Reserved capacity is a Queued Resource**, emitting `CreateQueuedResource`
   (no `CreateNode`) — the steady reserved-v4 pool in us-central2-b was invisible
   until QR events were added.
4. **~11% of preemptible nodes have no `DeleteNode`** (GCP preemption). Clamping
   those create-only nodes to "now" inflated totals ~30×; instead they are capped
   at the class-median observed lifetime (≈0.5 h), except nodes still in the live
   fleet (`gcloud compute tpus tpu-vm list`), which are clamped to now.

Validated against the live fleet: reconstructed "chips alive now" lands within
~7% of `gcloud compute tpus` (≈2,420 vs 2,608 chips).

**Ray-era size limitation.** Before the Iris cutover (~2026-04), TPUs were
provisioned by the Ray autoscaler (`ray-marin-<region>-worker-…`). Those names
encode the region — hence the TPU family, since each regional Ray cluster was
homogeneous (`config.RAY_REGION_FAMILY`) — but **not the slice size**, and the
audit payloads are empty. So before ~Mar 2026 we can chart **slice counts and
type/region mix, but not chips**; the chip charts are accurate only for the
Iris-era sized names. Long-history charts truncate to `config.AUDIT_CHART_START`.

## Running

Run from `lib/iris`. `extract` uses the iris env (duckdb is already a dep);
`fetch` additionally needs `wandb` (and `WANDB_API_KEY`) and a `gcloud` login
with read access to `hai-gcp-models` (for the audit pull); `charts` needs
`seaborn` (which pulls in matplotlib).

```bash
# 1. Mirror iris.worker + iris.task parquet (~10 GB), pull W&B run history
#    (~tens of thousands of runs, minutes), and pull the GCP TPU audit events
#    (~13 months, a few minutes). All cached, so re-runs skip what's present.
uv run --with wandb python scripts/blog_metrics/pipeline.py fetch

# 2. Roll up into per-day CSVs (fast; re-run freely while tweaking metrics).
uv run python scripts/blog_metrics/pipeline.py extract

# 3. Render charts.
uv run --with seaborn python scripts/blog_metrics/pipeline.py charts

# …or all three:
uv run --with wandb --with seaborn python scripts/blog_metrics/pipeline.py all
```

Pass `--no-wandb` to skip the W&B pull, `--no-audit` to skip the GCP audit pull
(each source's charts are simply omitted when its inputs are absent), or
`--force` to refresh all caches.

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
- `tpu_provisioning_by_family_daily.csv` / `..._by_region_daily.csv` — audit-log mean concurrent **chips** (Iris-era sizes) by family / region.
- `tpu_slices_by_family_daily.csv` / `..._by_region_daily.csv` — audit-log mean concurrent **slice count** (full ~13-month window) by family / region.

`data/charts/` (`.svg` for the blog, `.png` for preview): `accelerators_chips`,
`flops_pflops`, `users`, `tasks`, `fleet_utilization` (occupancy: busy vs idle
chips), `preemptible` (preemptible non-v4 vs reserved v4 capacity), `accelerators_by_region`,
`intraday_regions` + `intraday_regions_share` (within-day cross-region migration
for the candidate day, absolute and share-of-total),
`overview`, `compute_history` (2024→now, + cumulative), `calibration`, and the
long-history audit-log provisioning charts `tpu_provisioning_chips`
(+ `_no_v4`, `_by_region`) and `tpu_slices` (+ `_no_v4`, `_by_region`). The
wide `compute_history` and the long-history provisioning charts annotate the
`config.MILESTONES`; the cramped Iris-window charts omit them.

The `preemptible` chart needs the real per-device generation, which only the
iris structured data carries (~May 6 on) — W&B logs `num_devices` but not the
TPU type, so the v4-vs-non-v4 split **cannot** be reconstructed for the pre-iris
(Jan–Apr) history. (The audit logs *do* carry the type, hence the long-history
`tpu_provisioning_*` charts — but not the slice size before the Iris era.)

## Editing

- **Finelog metrics / definitions** → `extract.py` (DuckDB SQL).
- **W&B long-history compute** → `wandb_history.py` (fetch) + `extract.py` (rollup).
- **GCP audit-log TPU provisioning** → `audit_logs.py` (fetch + reconstruct).
- **FLOPS table, bucket size, milestones, bot-user filter, audit knobs** → `config.py`.
- **Chart styling** → `charts.py`.
