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
| `v6e-8` | 8 | ~73,400 | 14.8 | ~18 | 5 | steady; fastest single run |
| `v5p-8` | 4 | ~53,600 | 20.6 | ~25 | 4 | steady; best **tok/s per chip** (~13.4k) |
| `v6e-4` | 4 | ~40,600 | 26.3 | ~33 | 3 | steady; ≤ 2-epoch runs only |

## Batch (multi-host, off-budget)

All five batch slices now report (E2 runs; needed the `claim_and_run` lock fix so
every host trains — see issue #6365). Rates have stabilized vs the first ~13:50Z
probe. **`v5p-64` is the throughput champion** (~4.5× v6e-8), the v5p family is most
chip-efficient (no grad-accum), and even v5p-16 beats v6e-8 at equal chips. Favor
**v5p-64 → v5p-32** for new batch jobs. The v6e slices report much lower MFU
(~10-12% vs ~31-38% on v5p) — grad-accum / interconnect overhead on v6e at scale.

| slice | chips | hosts | tok/s | s/it | h / epoch | n | mfu | notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| `v5p-64` | 32 | 8 | ~333k | 3.3 | ~4.0 | 1 | ~31% | **fastest absolute**; steady (step ~1.1k) |
| `v6e-32` | 32 | 8 | ~213k | 5.2 | ~6.5 | 1 | ~10% | steady; low mfu |
| `v5p-32` | 16 | 4 | ~198k | 6.0 | ~7.4 | 1 | ~37% | steady |
| `v6e-16` | 16 | 4 | ~130k | 8.4 | ~10 | 1 | ~12% | steady |
| `v5p-16` | 8 | 2 | ~102k | 11.1 | ~14 | 1 | ~38% | steady; > v6e-8 at equal chips |

## Per-chip efficiency (for budget-bound decisions)

When the **budget** (chips) is the binding constraint, what matters is tok/s **per
chip**, not per run. Measured tok/s per chip across all running slices:

- batch `v5p-16` ~12.7k ≈ `v5p-32` ~12.4k > interactive `v5p-8` ~13.4k (the v5p
  family clusters at ~12-13k/chip; v5p-8 edges ahead per-chip at small scale)
- batch `v5p-64` ~10.4k > interactive `v6e-4` ~10.2k
- interactive `v6e-8` ~9.2k > batch `v6e-16` ~8.1k > batch `v6e-32` ~6.7k

So per-chip efficiency is governed by **family**, not slice size: v5p ~10-13k/chip
beats v6e ~7-9k/chip everywhere, and v6e degrades with scale (mfu ~10-12% on the
big v6e batch slices vs ~31-38% on v5p). The win from big batch slices is
**wall-clock per run** and **off-budget capacity**; for per-chip efficiency, stay
in the v5p family.
