# Review package: Delphi prefix candidates and conditional branch design

## Decision being reviewed

We have a budget of 600 fitted phase-1 continuations after selecting phase-0 prefixes. Prefix confirmations,
tied controls, and fresh endpoint confirmations are scientifically necessary controls and do not count toward
that fit budget. At most two adaptive rounds are allowed because wall-clock latency matters.

The primary target is endpoint Uncheatable BPB. Exact-boundary GitHub C++ BPB is a diagnostic component of
Uncheatable, not an independent guardrail. Historical one-phase panels at 60M, 300M, and Delphi are the
non-selecting transfer check. Table-9 is not a selection target in this round.

## Prefix stage

The canonical 280-row Delphi phase-0 replay is at:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_3e18_phase0_prefix_replay_20260820/materialized_boundary_metrics/prefix_boundary_fit_matrix.csv`

Every prefix model uses all 280 exact-boundary rows. Candidate optimization imposes the preregistered hard
constraint `max_b c0_b w0_b <= 10` materialized phase-0 epochs and the simplex constraints. Semantic family
partitions and equal-count exposure strata are prohibited. Same-domain high/low quality pairs may be used only
as coefficient shrinkage, not as a partition.

The initial compact model was shared-shape canonical DSP:

`y0 = intercept - sum_b a_b (1-exp(-rho E0_b)) + sum_b p_b softplus(log(1+E0_b)-tau)^2`

with `E0_b=c0_b w0_b`, nonnegative bucket amplitudes, one global `rho`, one global `tau`, and optional
same-domain quality-pair shrinkage. Narrow preregistered alternatives are a shared offset-power benefit and
damage-free versions. Model selection is nested mixture-blocked OOF. The exact implementation and constrained
runtime-lattice optimizer are:

- `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_phase0_prefix_surrogates_20260824.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/select_delphi_phase0_prefix_candidates_20260824.py`

One preregistered same-complexity diagnostic replaces canonical DSP's unbounded quadratic damage with a bounded
one-parameter damage curve. It improves outer-OOF Uncheatable RMSE from 0.01566 to 0.01471, Spearman from
0.9297 to 0.9366, and admissible regret@1 from 0.00403 to zero; GitHub C++ Spearman improves from 0.9373 to
0.9511 with equal regret. However, the direct historical transfer comparison is mixed: bounded-shape improves
Delphi one-phase Uncheatable RMSE and regret, while shared-shape is better in several 60M and 300M cells.
Neither head is therefore treated as a uniquely identified deployment model.

An equal ensemble of shared-shape and bounded-shape DSP is optimized at three forward-KL penalties, 0.05, 0.20,
and 0.50 BPB per nat, away from the runtime-materialized best observed cap-admissible prefix. This preserves
model uncertainty without spending candidates on near-duplicate head-specific optima. The runtime-materialized
observed incumbent and proportional mixture are controls, for five candidates total.
The unregularized model optimum is excluded because it lies TV 0.585 from every observed row and the fitted
models disagree materially there.
All five candidates are retrained at the same three frozen seed pairs. Candidate identities are fixed before
fresh outcomes: the observed cap-safe incumbent and all three ensemble-KL challengers enter the branch panel if
each challenger has paired mean-plus-SEM boundary regression no worse than 0.01 BPB. Proportional and GitHub C++
are diagnostic only. Primary seed 0 supplies the crossed branch state; seed 1 supplies a trajectory-seed
sensitivity panel. The challengers' nearest Hellinger separations are 0.136 and 0.053; both exceed the relevant
maximum partition-refit movement (0.101, 0.045, and 0.025 by KL level).

## Round-1 branch design

The fit budget is exactly 200 endpoints: the same 50 absolute phase-1 mixtures crossed with all four selected
prefixes. Tied continuations, branch-noise repeats, and selected-policy confirmations are separate scientific
controls and do not count toward the 600-row fit budget. Round 1 does not spend fit rows on a provisional lead.

All 50 fit mixtures are exact 2,048-example runtime lattice points. The candidate pool combines the canonical
280 historical phase-1 coordinates with deterministic proportional-centered Dirichlet draws at concentrations
500, 100, 20, 5, and 1. Ten points are selected from each maximum phase-1 materialized-exposure stratum:
`[0,5)`, `[5,15)`, `[15,30)`, `[30,45)`, and `[45,62.281654]`. Inside every exposure stratum, frozen radial
quotas cover TV bands `[0,.25)`, `[.25,.5)`, `[.5,.75)`, and `[.75,1]`; deterministic round-robin maximin then
spreads unit `sqrt(c1 * w1)` directions within each cell. The first three exposure strata use quotas 2/2/3/3;
the two highest use 0/4/3/3 because no TV<.25 candidates exist there. The resulting panel has 20 points within
TV 0.5 and 35 within TV 0.75, rather than concentrating at simplex vertices.

Three common controls are outside the fit budget: proportional, UniMax-8, and the original phase-1 continuation
paired with observed prefix incumbent `run_00125`. Each prefix also receives its tied continuation. Seed-1
trajectory-sensitivity sentinels repeat proportional, the incumbent continuation, and the highest-exposure fit
point. Two additional proportional continuations per seed-0 prefix vary only the branch data seed, giving three
same-checkpoint draws including the primary cross row. These controls do not count toward the fit budget.

Every continuation stays inside two scalar historical envelopes: at most 62.281654 materialized phase-1 epochs
in any bucket, and at most 255.824635 total materialized epochs for every candidate prefix. These are coarse
non-extrapolation guardrails, not per-bucket support guarantees, mechanistic claims, or estimated optima. The
total-exposure ceiling is diagnostic and does not bind the current candidate pool. The frozen design is:

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_common_branches_20260824.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_common_branches_20260824/`

The resulting fit panel has exactly ten points per exposure stratum, spans 1.526 to 59.301 maximum phase-1
epochs, and has minimum pairwise exposure-direction distance 0.256. The launch graph has 236 rows: 200 fitted
common continuations and 36 controls/sentinels. It is deliberately simpler than Bayesian
optimization: fixed exposure quotas, exact lattice realization, fixed controls, and maximin coverage in a
measured exposure geometry. Its novelty claim is the conditional shared-prefix experiment with common-action
contrasts, not a new black-box optimizer.

Round 1 is analyzed only after the complete crossed block is available. It jointly identifies prefix main
effects, continuation main effects, and the low-rank prefix-by-continuation interaction needed to decide whether
one prefix can safely receive all of Round 2.

## Round 2

The remaining 400 fit rows are allocated only after Round 1. If one prefix is clearly superior on the common
block, all 400 go to that prefix. If uncertainty remains, allocate about 300 to the leader and 100 to the nearest
competitor. Within a prefix, use about 70% acquisition-driven refinement and 30% Hellinger maximin coverage.
Fresh selected-policy confirmations remain outside the fit budget.

## Review questions

1. Does the prefix benchmark leak outer-fold outcomes through shape, shrinkage, cap filtering, or candidate
   selection? Is the equal shared/bounded ensemble at KL coefficients 0.05, 0.20, and 0.50 a defensible
   deployment candidate set?
2. Is three common-seed boundary validation sufficient to select a provisional prefix lead without using
   endpoint branch outcomes prematurely?
3. Does the fully crossed 50-by-4 allocation preserve the right inferential contrast and use the first 200 fit
   rows better than a provisional-lead allocation?
4. Is exposure-by-TV stratification followed by maximin selection over unit `sqrt(c1 * w1)` directions a
   defensible simple design in 39 dimensions? Does it omit any label-blind action geometry needed for
   branch-surrogate identification?
5. Are the scalar observed phase-1 and total-exposure envelopes adequate coarse non-extrapolation guardrails,
   given that they do not certify per-bucket support?
6. Do the seed-1 trajectory-sensitivity sentinels and same-checkpoint branch-draw repeats adequately separate
   seed sensitivity from branch sampling noise without spending fit budget?
7. List exact launch/manifest assertions needed to guarantee bitwise-compatible prefix replay and idempotent
stateful continuation from the post-update-2400 checkpoint `step-2399` through update 3007.
