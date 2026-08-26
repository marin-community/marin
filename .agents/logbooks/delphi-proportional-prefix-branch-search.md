# Delphi proportional-prefix branch search

## Current status

**TL;DR:** Exact phase-boundary checkpoints exist for three independently trained proportional prefixes. Wave 1 is being frozen as a single-prefix, outcome-blind continuation panel for v6e-8 in `us-east5-b`; no branch job has been submitted yet.

**Primary question:** Can optimizing only phase 1 from a proportional phase-0 state match or beat the validated cap-4 two-phase frontier on Uncheatable BPB?

**Primary comparator:** The exact tied continuation from the same prefix under common random numbers.

**Frontier comparator:** Mean Uncheatable BPB `0.9798883332146539` from the validated cap-4 conservative branch (`val_01m0xnbbg6jc6cj6awyfgrq2d1`).

## Hypothesis queue

1. `active` A proportional prefix remains recoverable: a phase-1 branch beats its tied continuation.
2. `active` Phase-1 optimization from proportional can reach or beat `0.979888333` Uncheatable BPB.
3. `active` A low-complexity tangent-space response model can improve acquisition over outcome-blind coverage after Wave 1.
4. `blocked` Final cross-prefix certification waits for a predicted candidate; proportional prefix seeds 0, 1, and 2 are already available.

## Frozen design intent

- Wave 1 uses 80 fit branches, 8 sealed geometry referees, and 14 controls/repeats.
- Fit coverage is outcome-blind and full-rank in the 39-bucket simplex tangent space.
- The exact validated frontier continuation is a fit anchor and has four fresh repeats; tied continuations have five observations in each of prefix seeds 0 and 1.
- Seed repeats do not consume fit budget.
- The acquisition target is scalar Uncheatable BPB. Atomic components and GitHub C++ are diagnostics unless a model change is separately registered before outcomes are inspected.
- Wave 2 is frozen before outcomes as 40 model-guided and 40 outcome-blind fit rows.
- Final validation crosses saved proportional prefix seeds and continuation data seeds.
- Wave 1 is explicitly a cross-hardware discovery panel: v5p-8 prefix states continued on v6e-8. A canonical frontier claim requires same-hardware confirmation.

## Checkpoint provenance

- Seed 0: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/prefix_proportional_control_seed0-105401/checkpoints/step-2399`
- Seed 1: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/prefix_proportional_control_seed1-4d8226/checkpoints/step-2399`
- Seed 2: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/prefix_proportional_control_seed2-0bd984/checkpoints/step-2399`

## Log

### 2026-08-25

- Recovered all three exact proportional phase-boundary checkpoints and verified their frozen provenance hashes.
- Chose a dedicated single-prefix experiment rather than reusing the hard-coded four-prefix common launcher.
- Began freezing Wave 1 from the audited harsh-cap branch design: full-rank square-root-simplex coverage, sealed referees, common-random tied comparison, and fresh tied controls.
- Bound each source checkpoint to a successful GCS executor record that pins v5p-8/east5a, exact seeds, and tensor parallelism.
- Added the exact validated cap-4 frontier mixture and four fresh repeats without using it as a geometry repeller; the 80-row fit panel remains full rank.
- Froze the proportional Wave-2 allocation and model/acquisition policy before opening any Wave-1 endpoint.
