# Delphi phase-1 common continuation design

Round 1 crosses the same 50 fit-budget continuations with every selected
prefix. Ten continuations are selected in each of five maximum phase-1 exposure strata. Within each
exposure stratum, frozen total-variation quotas preserve local, intermediate, and global coverage before
deterministic maximin selection over unit `sqrt(c1 * w1)` directions. The pool combines all 280
historical phase-1 coordinates with deterministic proportional-centered Dirichlet draws, after exact
`1/2048` runtime materialization.

Three controls do not consume fit budget: proportional, UniMax-8, and the historical continuation paired
with the observed cap-safe prefix incumbent `run_00125`. The launcher also adds prefix-specific
tied controls and three prefix-seed stability sentinels per selected prefix; these remain outside fit budget.

Every row stays within each bucket's observed canonical-panel phase-1 and total materialized-exposure
envelopes for every frozen candidate prefix. The largest bucket-wise caps are
62.281654 phase-1 epochs and
255.824635 total epochs. These remain coordinate-wise support guardrails,
not a claim that every joint mixture is in-distribution.

Fit exposure-bin counts: `{"[0,5)": 10, "[15,25)": 10, "[25,35)": 10, "[35,62.2817)": 10, "[5,15)": 10}`

Fit TV-bin counts: `{"[0,0.25)": 11, "[0.25,0.5)": 21, "[0.5,0.75)": 18, "[0.75,1)": 0}`

Minimum pairwise fit-direction distance: 0.3219

Continuation weights SHA-256: `9547515728e3c85ed564066cf8cfa36eefa80a7241cde3b32e351004f9afc883`
