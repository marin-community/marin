# exp117 sweep — region / slice throughput findings

Empirical throughput of the contacts-v1 1.5B recipe (global batch 128, seq 8192) across the
region/slice targets exercised during the exp117 rung-0 sweep (2026-07-15 → 07-17). Computed from the
orchestrator's periodic `run_progress` observations (`scratch/exp117-adaptive-sweep-s01.sqlite`), one snapshot
per ~30-min heartbeat, aggregated over each target's whole life in the sweep. Simple ratios only — no
time-weighting or decay.

## Metric definitions

One training step processes **`128 × 8192 = 1,048,576` tokens** (`~1.05M tok/step`). A full 8-epoch run is
**35,680 steps**. All rates are plain totals:

| metric | definition |
|---|---|
| **active steps/min** | total steps advanced ÷ wall-time **while progressing** (excludes hung / preempted / pending intervals) — the slice's speed when actually training |
| **effective steps/min** | total steps advanced ÷ **total** wall-time observed on the target (includes every stall) — the lifetime average you actually get |
| **stall %** | `1 − active_time / total_time` — share of observed time with zero progress (preemption + hangs + capacity waits) |
| **%/hr** | effective steps/min × 60 ÷ 35,680 × 100 — run-completion rate (a full run = 100%) |
| **MTok/s** | effective steps/min × 1.05M ÷ 60 — token throughput, millions of tokens per second |
| **steps/min/chip** | active steps/min ÷ chips — chip efficiency |

## Throughput by region × slice (lifetime aggregate)

Sorted by effective (lifetime-average) throughput. "obs-min" = total observed minutes feeding the average.

| region | slice | chips | obs-min | active steps/min | **effective steps/min** | %/hr | MTok/s | steps/min/chip | stall % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| europe-west4 | v5litepod-64 | 64 | 2 878 | 20.3 | **20.1** | 3.4 | 0.35 | 0.317 | 1 % |
| europe-west4 | v6e-32 | 32 | 7 463 | 11.8 | **11.7** | 2.0 | 0.20 | 0.370 | 2 % |
| us-east5 | v6e-32 | 32 | 3 436 | 10.6 | **7.9** | 1.3 | 0.14 | 0.330 | 25 % |
| us-west4 | v5litepod-64 | 64 | 237 | 14.4 | **7.2** | 1.2 | 0.13 | 0.226 | 50 % |
| europe-west4 | v6e-16 | 16 | 6 507 | 7.3 | **7.1** | 1.2 | 0.12 | 0.455 | 3 % |
| us-east5 | v5p-64 | 32 | 511 | 11.4 | **5.3** | 0.9 | 0.09 | 0.356 | 54 % |
| us-west4 | v5litepod-32 | 32 | 1 771 | 5.6 | **5.3** | 0.9 | 0.09 | 0.175 | 5 % |
| us-east1 | v6e-32 | 32 | 450 | 5.6 | **1.8** | 0.3 | 0.03 | 0.174 | 68 % |
| us-east5 | v6e-16 | 16 | 128 | — | **0.0** | 0.0 | 0.00 | — | 100 % |

*Low-sample rows (obs-min < ~500: us-west4 v5litepod-64, us-east5 v5p-64, us-east1 v6e-32, us-east5 v6e-16)
are directional, not precise.*

## Clean slice comparison (europe-west4, the low-stall region)

Isolating the reliable region removes preemption noise, so the **inherent size/efficiency tradeoff** is
clean here. Raw speed and chip-efficiency pull in opposite directions:

| slice | chips | steps/min | %/hr | MTok/s | steps/min/chip | to finish one run |
|---|---:|---:|---:|---:|---:|---:|
| **v5litepod-64** | 64 | **20.1** | 3.4 | **0.35** | 0.317 | **~30 h** |
| v6e-32 | 32 | 11.7 | 2.0 | 0.20 | 0.370 | ~51 h |
| v6e-16 | 16 | 7.1 | 1.2 | 0.12 | **0.455** | ~84 h |

- **Fastest wall-clock: v5litepod-64** (~2× v6e-32, ~3× v6e-16) — more chips ⇒ faster per step at fixed
  global batch. This is the slice to use when a single run must finish ASAP.
- **Most chip-efficient: v6e-16** (0.455 steps/min/chip vs 0.317 for v5litepod-64). "Bigger is not always
  better": going 16→32→64 chips *raises* wall-clock speed but *lowers* per-chip return, because a fixed
  batch of 128 spread over more chips means less work per chip and more cross-chip communication.
- **v6e is the most chip-efficient family** overall (v6e ≈ 0.33–0.46 /chip, v5p ≈ 0.36 /chip,
  v5litepod ≈ 0.18–0.32 /chip). v5litepod-64 wins on wall-clock only by throwing 64 chips at it.

## Changes in throughput over time

- **europe-west4 stayed flat and reliable** the entire sweep (1–3 % stall on all slices). Its
  active≈effective throughput is why the effective numbers above are trustworthy for that region.
- **us-east5 degraded over the run.** v6e-32 there trained at a healthy 10.6 steps/min when up, but
  recurring "silent hangs" (worker alive to iris, no progress) cut its *effective* rate to 7.9 (25 %
  stall); one us-east5 trial spent a ~5 h stretch at ~15 % of normal throughput before self-recovering.
- **us-west4 v5litepod-64** shows the SIGSEGV effect: 14.4 active but only 7.2 effective (50 % stall) —
  crashes, each recovered by checkpoint-resume, roughly halved its realized throughput.
- **SIGSEGV rate rose through 2026-07-17** (~20 crashes, cluster-wide across *every* region and slice
  family, exit 139). Not localized; each recovered on an on-target retry from checkpoint.

## Region × slice availability & errors (brief)

| target | note |
|---|---|
| europe-west4 v6e-{8..128}, v5litepod-{32..128} | reliable workhorse; **v6e-64 had no capacity** when requested on 07-17 (16-host allocation unavailable) though v6e-16/32 and v5litepod-64 scheduled fine |
| us-east5 v6e-32 | schedules, but prone to **silent hangs** (no crash, no progress; iris still "running") |
| us-east5 v6e-16 | **never scheduled** in ~2 h+ of attempts (no matching workers) |
| us-east5 v5p-64 | **capacity-scarce** — repeatedly unschedulable |
| us-central1 v5p-{16..} | **capacity-starved** — never scheduled at all (trial relocated out) |
| us-east1 v6e-32 | **hangs + SIGSEGV**; worst realized throughput → trial relocated to EU |
| us-west4 v5litepod-{32,64} | schedules and runs, but hit repeated **SIGSEGV** on 07-17 |

Slice-in-one-region-not-another: **v6e-16 ran cleanly in europe-west4 but never scheduled in us-east5**;
**v5p-64 ran in us-east5 (briefly) but was chronically unschedulable in us-central1**.

## Aggregate

- Total observed progress across the sweep so far: **~233k training steps ≈ 244 B tokens** (lower bound —
  30-min sampling misses sub-interval detail and monitoring gaps).
- **Practical takeaway for placement:** europe-west4 is the reliability anchor; for a single run that must
  finish fast, europe-west4 **v5litepod-64** is the throughput winner; for chip-efficient parallel
  breadth, **v6e-16/32** give the most run-progress per chip.
