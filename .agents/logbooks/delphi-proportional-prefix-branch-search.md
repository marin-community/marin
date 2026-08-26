# Delphi proportional-prefix branch search

## Current status

**TL;DR:** Wave 1 is running as 102 v6e-8 continuations in `us-east5-b` from exact proportional phase-boundary checkpoints. The model and Wave-2 acquisition contracts were frozen before endpoint outcomes were available.

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
- Submitted `/calvinxu/dm-delphi-3e18-phase1-proportional-prefix-wave1-v6e8-east5b-20260826` at interactive priority from commit `af65494a1ea74d84e03f7cfba4d3e525112279bd` after a successful 102-row dry run and east5 launch-safety validation.
- CC's read-only review found five launch blockers in the first draft. Repairs added the exact frontier anchor, repeat groups, rank-preserving geometry, cross-hardware provenance, and a sealed Wave-2 contract; the follow-up verdict was launch-ready after runtime-path tests passed.
- A second CC review of the adaptive implementation found three Wave-2 blockers: an unpaired data seed, exploitation overflow when more than two ridge settings survived, and disagreement across ridge settings rather than coordinate model classes. The corrected design reuses data seed `970000`, selects one alpha representative per eligible direct/square-root model, caps exploitation at 16, and uses fold-ensemble spread when only one coordinate model survives.
- The second correction happened after 13 Wave-1 endpoints were visible. The contract records this deviation explicitly; the endpoint values and ordering did not enter any correction. The revised code also fails closed when no model passes the gain-sign gate, enforces contract provenance and bucket order, and refuses to overwrite a differing frozen Wave-2 artifact.
- CC's follow-up found no Wave-2 launch blocker. Before submission, the generic launcher now requires an explicit branch run-ID base, the combined-wave materializer rejects run-ID overlap, the validated frontier BPB is a required fit argument, and tests assert the generated `970000` seed plus the 16/12/12 adaptive composition.
