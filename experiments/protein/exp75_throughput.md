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

First probe (E2 runs, ~13:50Z, **still ramping** — early steps, tok/s climbs as the
compile amortizes; ranking is already clear). Needed the `claim_and_run` lock fix
(every host now trains; see issue #6365). **`v5p-64` is the throughput champion**
(~4.5× v6e-8), the v5p family is most chip-efficient (no grad-accum), and v5p-16
already beats v6e-8 at equal chips. Favor **v5p-64 → v5p-32** for new batch jobs.

| slice | chips | hosts | tok/s | s/it | h / epoch | n | notes |
|---|---|---|---:|---:|---:|---:|---|
| `v5p-64` | 32 | 8 | ~333k | 3.4 | ~4.2 | 1 | **fastest absolute**; ramping |
| `v5p-32` | 16 | 4 | ~199k | 5.6 | ~6.9 | 1 | ramping |
| `v6e-16` | 16 | 4 | ~130k | 8.8 | ~11 | 1 | ramping |
| `v5p-16` | 8 | 2 | ~102k | 12.1 | ~15 | 1 | > v6e-8 at equal chips |
| `v6e-32` | 32 | 8 | — | — | — | 0 | gang-pending (8 hosts scarce) |

## Per-chip efficiency (for budget-bound decisions)

When the **budget** (chips) is the binding constraint, what matters is tok/s **per
chip**, not per run. Measured interactive: v5p-8 ~13.4k > v6e-4 ~10.2k > v6e-8
~9.2k. Fill the batch rows to see whether a big slice beats v5p-8 per chip (it
shouldn't change per-chip much within a family — the win from big batch slices is
**wall-clock per run** and **off-budget capacity**, not per-chip efficiency).
