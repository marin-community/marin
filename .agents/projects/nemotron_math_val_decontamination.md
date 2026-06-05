# Plan: Decontaminated nemotron math val set + full-family contamination scan

Owner: ahmed. Branch: `nemotron-math-contamination`. Logbook: `.agents/logbooks/nemotron_math_data.md`.

## Problem

The 12,500-window math val carve-out has J>=0.75 near-dups in train for 18.6% of
val tokens (full-corpus envelope), and the window split also leaks other windows
of the same source doc into train. Both inflate large-scale val performance.

All prior scanning covered only `nemotron_cc_math_v1/4plus` (45.1M docs, 92 GB).
The val set has never been compared against the rest of the family.

## Recall limit (drives the plan)

LSH was 286 perms / 26 bands / r=11. Candidate recall P = 1-(1-J^r)^b:
J=0.9 -> 100%, 0.75 -> 67%, 0.6 -> 9%, 0.5 -> 1.3%. The verified ">=0.5" pair
list is nearly complete at 0.9, ~2/3 at 0.75, almost empty below 0.6. For "no
overlap as much as possible" rebucket val signatures at r=4 / b=71 (284 perms):
P(0.5)=99%, P(0.4)=84%. Tradeoffs: more candidates -> verification stage grows.

## Phase 0 — local, no jobs

Build val v0 from existing artifacts: 57,243 doc ids minus
all verified J>=0.5 (~20,850 docs) minus boundary-spill windows
(window not fully inside val docs). Output `scratch/decon_val_v0_windows.npy`
(~2,300 windows, ~9.3M tokens), token-exact cache slice next.

## Phase 1 — 4plus high-recall rescan (us-east5, preemptible CPU)

1. Recompute MinHash (284 perms / 71 bands / r=4, 5-char, seed 42) for full
   4plus; collect val buckets (57,243 x 71 ~= 4.1M keys).
2. Bucket join -> candidates -> verify >=0.5 -> drop list update -> val v1.
Scope: 92 GB; prior identical pipeline ran ~30 min wall + verify ~5 min.
Driver: `scripts/analysis/nemotron_math_val_full_scan.py --subset 4plus`.

## Phase 2 — remaining math subsets (us-central1, where data lives)

Scope is nemotron MATH only (not the CC text/code family):

| subset | size | region |
|---|---:|---|
| math_v1/4plus (4plus_b05688a8) | 92 GB | us-east5 (phase 1) |
| math_v1/3 (3_f8007d22) | 159 GB | us-central1 |
| math_v1/4plus_mind (4plus_mind_1173e12a) | 144 GB | us-central1 |

Same pipeline as phase 1: per-shard MinHash -> keep rows hitting val bucket
keys -> candidates -> exact-Jaccard verify. Copy val docs (~150 MB) to
us-central1 scratch once; never read normalized parquet cross-region.
Driver: `scripts/analysis/nemotron_math_val_full_scan.py --subset {3,4plus_mind}`.

## Phase 3 — clean val + re-eval

Final drop list = union of phases. Rebuild val (target >=2k clean windows),
re-eval scaling checkpoints, refit scaling laws. Future: fuzzy dedup before
carve-out (logbook follow-up).

## Status

- [ ] Phase 0
- [ ] Phase 1
- [ ] Phase 2
- [ ] Phase 3
