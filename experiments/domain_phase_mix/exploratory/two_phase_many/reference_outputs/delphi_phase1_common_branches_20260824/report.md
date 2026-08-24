# Delphi phase-1 common continuation design

Round 1 crosses the same 50 fit-budget continuations with every selected
prefix. Ten continuations are selected in each of five maximum phase-1 exposure strata. Within each
exposure stratum, frozen total-variation quotas preserve local, intermediate, and global coverage before
deterministic maximin selection over unit `sqrt(c1 * w1)` directions. The pool combines all 280
historical phase-1 coordinates with deterministic proportional-centered Dirichlet draws, after exact
`1/2048` runtime materialization.

Three controls do not consume fit budget: proportional, UniMax-8, and the historical continuation paired
with the observed cap-safe prefix incumbent `run_00125`. Prefix-specific tied controls and
branch-noise sentinels are added by the launcher and likewise remain outside the fit budget.

Every row stays within the observed canonical panel's scalar phase-1 maximum-exposure envelope
(62.281654 epochs) and scalar total maximum-exposure envelope
(255.824635 epochs) for every frozen candidate prefix. These are coarse
non-extrapolation guardrails, not per-bucket support guarantees; the total-exposure envelope does not bind
the current candidate pool.

Fit exposure-bin counts: `{"[0,5)": 10, "[15,30)": 10, "[30,45)": 10, "[45,62.2817)": 10, "[5,15)": 10}`

Fit TV-bin counts: `{"[0,0.25)": 6, "[0.25,0.5)": 14, "[0.5,0.75)": 15, "[0.75,1)": 15}`

Minimum pairwise fit-direction distance: 0.2558

Continuation weights SHA-256: `3b3fba7f7905ba9788cd9a94936eb4f86561b80cb64cd8234a08d2acd70f126e`
