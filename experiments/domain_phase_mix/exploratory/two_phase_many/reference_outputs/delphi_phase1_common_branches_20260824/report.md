# Delphi phase-1 common continuation design

Round 1 crosses the same 50 fit-budget continuations with every selected
prefix. Ten continuations are selected in each of five maximum phase-1 exposure strata. Within each
exposure stratum, frozen total-variation quotas preserve local, intermediate, and global coverage before
deterministic maximin selection over unit `sqrt(c1 * w1)` directions. The pool combines all 280
historical phase-1 coordinates with deterministic proportional-centered Dirichlet draws, after exact
`1/2048` runtime materialization.

Three controls do not consume fit budget: proportional, UniMax-8, and the historical continuation paired
with the observed cap-safe prefix incumbent `run_00125`. The incumbent is outcome-selected, so it
is not used as a maximin repeller for the fit panel. The launcher also adds prefix-specific tied controls,
three prefix-seed stability sentinels per selected prefix, and four same-checkpoint phase-1 data-seed
replicates; these remain outside fit budget.

Every row stays within each bucket's observed canonical-panel phase-1 and total materialized-exposure
envelopes for every frozen candidate prefix. The largest bucket-wise caps are
62.281654 phase-1 epochs and
255.824635 total epochs. These remain coordinate-wise support guardrails,
not a claim that every joint mixture is in-distribution.

Historical support caps are measured before runtime lattice materialization, while every candidate is
checked after exact materialization. This conservative asymmetry accounts for the rejected Dirichlet draws.
The total-exposure cap is a verification guardrail and did not reject a candidate in this frozen pool.

Fit exposure-bin counts: `{"[0,5)": 10, "[15,25)": 10, "[25,35)": 10, "[35,62.2817)": 10, "[5,15)": 10}`

Fit TV-bin counts: `{"[0,0.05)": 1, "[0.05,0.15)": 3, "[0.15,0.25)": 7, "[0.25,0.5)": 21, "[0.5,0.75)": 18, "[0.75,1)": 0}`

Minimum pairwise fit-direction distance: 0.3017

Continuation weights SHA-256: `9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355`
