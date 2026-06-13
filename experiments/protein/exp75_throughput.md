# exp75 — realized throughput by slice × band

Reference for slice selection. Throughput is **measured** from W&B
(`throughput/tokens_per_second`, steady post-warmup) on the exp75 1.5B recipe
(global batch 128, seq 8192 → ~1.05M tokens/step on every slice/band, so numbers
are directly comparable). `s/it` is realized wall time per step (`_runtime/_step`,
includes JIT/restart overhead — the basis for ETAs). Update as runs report in.

Per-step tokens are constant, so **tok/s ∝ steps/h**; an epoch is 4,460 steps.

## Interactive (single-host, counts toward budget)

| slice | chips | tok/s | s/it | h / epoch | n | notes |
|---|---|---:|---:|---:|---:|---|
| `v6e-8` | 8 | ~73,500 | 14.7 | ~18 | 4 | fastest single run |
| `v5p-8` | 4 | ~53,500 | 20.7 | ~26 | 3 | best **tok/s per chip** (~13.4k) |
| `v6e-4` | 4 | ~40,600 | 26.5 | ~33 | 3 | ≤ 2-epoch runs only |

## Batch (multi-host, off-budget)

_TBD — fill from the 5-slice probe (1 job per type). Bigger slices should finish a
run faster (more chips, no grad-accum) but are scarcer and more bug-prone; we don't
know a priori which wins on realized throughput or schedulability._

| slice | chips | hosts | tok/s | s/it | h / epoch | n | notes |
|---|---|---|---:|---:|---:|---:|---|
| `v5p-16` | 8 | 2 | — | — | — | 0 | |
| `v5p-32` | 16 | 4 | — | — | — | 0 | |
| `v5p-64` | 32 | 8 | — | — | — | 0 | |
| `v6e-16` | 16 | 4 | — | — | — | 0 | |
| `v6e-32` | 32 | 8 | — | — | — | 0 | |

## Per-chip efficiency (for budget-bound decisions)

When the **budget** (chips) is the binding constraint, what matters is tok/s **per
chip**, not per run. Measured interactive: v5p-8 ~13.4k > v6e-4 ~10.2k > v6e-8
~9.2k. Fill the batch rows to see whether a big slice beats v5p-8 per chip (it
shouldn't change per-chip much within a family — the win from big batch slices is
**wall-clock per run** and **off-budget capacity**, not per-chip efficiency).
