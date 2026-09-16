# Delphi TPP40 multiregion final launch review

This is a durable summary of the subscription-safe Claude Code review, not a verbatim transcript.

## Verdict

**GO.** No launch blockers were found for the exact East5 and Europe production commands reviewed in `delphi_tpp40_multiregion_final_launch_cc_review_20260831.md`.

The review used `claude-opus-5` with maximum reasoning in read-only mode. The subscription account was preflighted as `plambdafour@proton.me` with `stripe_subscription`; `ANTHROPIC_API_KEY` was absent.

## Load-bearing findings

- The frozen assignment is structurally disjoint: 29 completed coordinates, 125 assigned to East5, and 126 assigned to Europe. Run order 30 is the only resumable East5 coordinate and is forced to East5.
- Both commands pin assignment semantic SHA-256 `8074b0d3a92e5e002336389849f33bbd630d9be2ea1580ccf436dfb2b40ea836`; no stale-v1 assignment path or digest remains in either reviewed command.
- Parent, training, Table-9, checkpoint, state, and cache placements are internally region-local for each command.
- The strict preregistered v4 numerical acceptance remains failed. Production is authorized only by the separately recorded, user-approved one-pair operational screen; the strict result must not be relabeled.

## Required post-launch analysis

- Estimate a region fixed effect and a region-by-mixture interaction before pooling endpoint results across accelerators and regions.
- Re-evaluate a common checkpoint set in one region when feasible to separate evaluation-region effects from training-region effects.
- Exclude the Europe-only `ngd3dm2_stratified_300m_6b` row from the fixed-effect estimate, or model its unmatched status explicitly.

## Non-blocking observations

- The assignment's `run_order_args.east5` field omits completed rows, while the production command correctly consumes the assignment file and intentionally replays completed East5 rows for missing evaluation work.
- The launcher still supports explicit `--run-orders`, but the exact production commands use the frozen assignment contract and expected semantic digest.
- The legacy East5 freeze state is recorded in the assignment and Fieldbook; it was also verified live before submission.
