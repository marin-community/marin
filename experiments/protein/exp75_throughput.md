# exp75 — throughput by slice × band

Reference for slice selection. Same recipe on every slice/band (global batch 128,
seq 8192 → ~1.05M tokens/step), so all numbers are directly comparable.

## Metrics — track all four, **decide on `wts`**

Per W&B run (`eric-czech/marin`, group `exp75-contacts-v1-tune`):

| metric | what | from W&B keys |
|---|---|---|
| **`wts`** | **wall-clock tok/s** — tokens / total elapsed, incl. queue/preemption/availability | `throughput/total_tokens` ÷ (`heartbeatAt` − `createdAt`) |
| `ats` | active tok/s — tokens / time the run was connected/active | `throughput/total_tokens` ÷ `_runtime` |
| `tts` | training tok/s — compute throughput during steps | `throughput/tokens_per_second` |
| `mfu` | model FLOPs utilization | `throughput/mfu` |

Ordering always `tts ≥ ats ≥ wts` (compute ⊂ active ⊂ wall). **`wts` is the
decision metric** — it is the real "how fast will this finish" rate, so it already
folds in slice speed *and* scarcity/preemption. Use the others only as diagnostics:
`tts` = the upside a slice would give *if always available* (the case for chasing a
scarce fast slice), `ats` isolates in-run overhead (eval/loading), **MFU is mostly
irrelevant** but worth a glance. `wts ≪ tts` flags a slice that's fast when up but
rarely scheduled. Re-measure periodically; never assume bigger = faster.

## Interactive (single-host, counts toward budget)

| slice | chips | tts | ats | wts | mfu | n | notes |
|---|---|---:|---:|---:|---:|---:|---|
| `v6e-8` | 8 | ~73.4k | — | — | ~14% | 5 | fastest single run |
| `v5p-8` | 4 | ~53.6k | — | — | — | 4 | best tts/chip (~13.4k) |
| `v6e-4` | 4 | ~40.6k | — | — | — | 3 | slowest |

## Batch (multi-host, off-budget)

| slice | chips | hosts | tts | ats | wts | mfu | n | notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| `v5p-64` | 32 | 8 | ~333k | — | — | ~31% | 1 | fastest `tts`; 8-host gang = preemption-prone |
| `v6e-32` | 32 | 8 | ~213k | — | — | ~10% | 1 | low mfu; preemption-prone |
| `v5p-32` | 16 | 4 | ~198k | — | — | ~37% | 1 | reliable 4-host |
| `v6e-16` | 16 | 4 | ~130k | — | — | ~12% | 1 | low mfu |
| `v5p-16` | 8 | 2 | ~102k | — | — | ~38% | 1 | > v6e-8 at equal chips |

`tts` is measured (steady, post-warmup). **`ats`/`wts` are TODO** — backfill with
the formulas above; `wts` is expected well below `tts` on the big v6e/8-host gangs
(frequent preemption) and is what should actually drive slice choice. v5p MFU
~31–38% vs v6e ~10–12% (the only reason MFU is worth tracking: it confirms why v6e
big slices underperform their chip count).
