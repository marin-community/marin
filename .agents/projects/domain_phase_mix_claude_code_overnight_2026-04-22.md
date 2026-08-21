# Claude Code Overnight Brief: Direct Backbone Search

## Working Context

Work directly in the repo:

- `/Users/calvinxu/Projects/Work/Marin/marin`

Do not prepare a new packet.

Keep the research log updated as you go:

- `/Users/calvinxu/Projects/Work/Marin/marin/.agents/logbooks/swarm-transfer-calibration.md`

## Mission

Search for better **direct backbones** for the corrected two-phase data-mixing
problem.

The goal is not to force a specific formula. The goal is to find a form that is:

- fully end-to-end trainable
- elegant and structurally motivated
- competitive on the corrected benchmark basis
- plausible under the continuous free-mixture geometry solve

## High-Level Constraints

Serious candidates should satisfy all of the following:

- no frozen GRP-derived feature map in the final model
- use a real optimizer for nonlinear structure
- avoid generic high-capacity residual regressors unless there is a very clear
  structural reason
- evaluate on the corrected trustworthy `520M` basis
- use the standard continuous free-mixture candidate geometry solve, not an
  observed-mixture proxy
- do not treat imported packet results or stale external summaries as final
  evidence; a candidate only counts after a local apples-to-apples rerun on the
  corrected basis with the standard validation stack
- do not use toy optimizer budgets for final validation; final promoted runs
  should use a real nonlinear fit budget, not one- to three-iteration search
  probes

  The hybrid framing still matters, but the direct backbone is the current
  bottleneck, so focus there first.

## What To Optimize For

Use these as the **primary selection metrics**, in this order:

1. grouped `fold_mean_regret_at_1`
2. `lower_tail_optimism`
3. `low_tail_rmse`
4. overall `rmse`

Use these as **secondary metrics / tie-breakers**:

- overall `spearman`
- fixed-`520M` `rmse`
- fixed-`520M` `spearman`
- predicted/actual spread calibration

## Hard Acceptance Gates

Do not promote a model as a serious candidate unless all of these are true:

- it is fully end to end, with no frozen feature body
- it clearly beats the corrected frozen-basis baselines on the 4 primary
  metrics above, or else has a compelling tradeoff that is clearly explained
- it does not fail badly on corrected fixed-`520M` ranking
- it passes a continuous candidate-geometry sanity check

Geometry should be treated as a gate, not the primary objective. But geometry
failures still disqualify benchmark-only wins.

## Current Validated Context

The benchmark basis was corrected again after recovering exact-step final evals.

Canonical current basis:

- strong-tier perplexity-ready rows: `94`
- trustworthy fixed-`520M` rows: `16`
- all fixed-`520M` rows are qsplit rows
- the active benchmark split now has:
  - train rows: `274`
  - holdout rows: `49`
  - fixed-`520M` rows: `16`
  - random supplement rows: `33`

These newly added `520M` rows were sanity-checked against the matching
`130M -> 300M -> 520M` chains and are monotonically improving.

Do not rely on any stale `4`-row or `7`-row `520M` conclusions.

## Current Models To Beat Or Explain

Corrected frozen-basis baselines on the current basis:

- `direct_shared_score_tilt_poly4`
  - no longer the meaningful frontier
- `two_stage_quality_split_signed`
  - no longer the meaningful frontier

The refreshed local rerun changed the ranking substantially.

Current live lead:

- `gated_ci_scale_n_rpm200`
  - overall `RMSE 0.01603`, `Spearman 0.97403`
  - `fold_mean_regret@1 0.00000`
  - `lower_tail_optimism 0.0`
  - `low_tail_rmse 0.02840`
  - fixed-`520M` `RMSE 0.02339`, `Spearman 0.96618`
  - seed-7 geometry:
    - mean nearest-TV `0.3656`
    - max nearest-TV `0.3748`
    - min phase-1 support `5.81`
    - max phase-1 weight `0.294`

Strong simple baseline:

- `gated_centered_interactions`
  - `99` params
  - overall `RMSE 0.01840`, `Spearman 0.96293`
  - `fold_mean_regret@1 0.00000`
  - fixed-`520M` `RMSE 0.02451`, `Spearman 0.96103`
  - geometry:
    - mean nearest-TV `0.3688`
    - max nearest-TV `0.3752`
    - min phase-1 support `6.11`
    - max phase-1 weight `0.294`

Important correction:

- `a012_p003_lb22` is **not** the live lead anymore on the refreshed basis.
  - overall `RMSE 0.01904`
  - fixed-`520M` `Spearman 0.75846`
- `e2e_full_qrho_centered` is now clearly behind as well.

The live frontier is now the gated retained-residual family, not the old latent
`a012` family.

## What The Last Autonomous Run Actually Exhausted

The most recent autonomous search established a useful boundary:

- the **trainable GRP body + linear floor-log head + small centered interaction**
  family looks locally saturated
- within that family, three things do **not** unlock a ship-level win:
  - feature enrichment alone
  - more Powell or simple training-procedure tweaks
  - larger regularization sweeps

Concretely, the last run tested:

- feature enrichments such as:
  - signal × scale
  - penalty × scale
  - entropy terms
  - quality decompositions
- optimizer / training variants:
  - Powell 120 / 200 iterations
  - multistart Powell
  - scale-balanced weighting
- larger regularization grids over:
  - `alpha`
  - body prior
  - interaction penalty multiplier

The finding was not "the search space is exhausted." The finding was:

- **that specific architecture family is near its frontier**
- it trades 520M rank against geometry in a mostly monotone way

Treat that as a boundary condition, not a final conclusion.

## What Does Not Count As A New Direction

The following do **not** count as materially new directions on their own:

- adding a few more linear interaction features to the current centered q+rho
  head
- rerunning the same model with more Powell iterations
- larger regularization sweeps around the same linear-head family
- simple reweighting tricks such as naive scale-balanced loss

You may revisit any of these only if they are paired with a genuinely new
ingredient such as:

- a different model class
- a different optimizer class
- explicit geometry constraints
- a different training objective
- a nonlinear head

## What We Think Is Promising

The refreshed rerun points in a simpler direction than before.

The evidence now favors:

- gated retained-residual body
- centered family-share interactions
- `uN`-only scale-adaptive residuals
- parameter-efficient variants of that family
- optimizer / plausibility improvements on top of that body

The evidence now suggests:

- `quality_tilt` is not the main donor on the refreshed basis
- the old latent-channel `a012` branch is no longer the best use of effort
- the remaining bottleneck is mixture plausibility at `300M+`, not fixed-`520M`
  ranking

Priority directions now include:

- simplify the current gated winner without losing its corrected `520M` gains
  - lower-rank or family-shared `uN` residuals
  - grouped/domain-block residual compression
  - variants between `gated_centered_interactions` and
    `gated_ci_scale_n_rpm200`
- explicit optimizer / plausibility control
  - constrained mixture search
  - manifold or prototype search priors
  - geometry-aware candidate optimization
- small, elegant corrections only if they materially help the gated family
  - compact hi/lo quality gaps
  - compact low-quality discount terms

The evidence currently does **not** favor:

- reviving the old latent `a012` branch as the default search direction
- adding `quality_tilt` by default to the current gated winner
- broad unconstrained domain-scale interaction growth
- larger model classes before exhausting the simple gated family
- geometry checks that only use observed-mixture proxies

## Search Guidance

Good directions:

- parameter-efficient simplifications around the gated retained-residual winner
- constrained or manifold-aware mixture optimization on top of the gated body
- forms that improve plausibility at `60M` and `300M` without losing the new
  corrected `520M` ranking gains
- small structured corrections that can be defended clearly
- explicit ablations that answer whether each extra block is really earning its
  keep

Bad directions:

- freezing the body and only retuning the head
- adding arbitrary capacity without structural motivation
- using a model that wins RMSE by collapsing candidate mixtures to corners
- defaulting back to the stale-packet latent frontier

## Reporting Requirements

For each serious family you try, report:

1. exact form
2. parameter count
3. whether any part is frozen
4. optimizer used
5. corrected benchmark metrics
6. fixed-`520M` metrics
7. candidate geometry
8. whether it is:
   - serious improvement
   - benchmark-only improvement
   - negative result

   Be explicit about negative results. Those are valuable.

## Stopping Rule

Do **not** stop for:

- a small local win
- a near-tie
- a benchmark-only win
- exhaustion of one architecture family

Only stop early if you find a **ship-level serious candidate**. For this brief,
that means a model that is fully end to end and clears all of:

- validated locally on the refreshed `16`-row trustworthy `520M` basis with the
  standard continuous geometry solve
- beats `gated_ci_scale_n_rpm200` on the primary lexicographic stack, or has a
  clearly better simplicity/performance tradeoff that is explicitly defended
- overall `rmse <= 0.0160`, or else is materially simpler and nearly tied
- fixed-`520M` `rmse <= 0.0234`
- mean fixed-`520M` `spearman >= 0.96`
- minimum fixed-`520M` `spearman >= 0.94`
- geometry no worse than the current serious candidate in a meaningful way:
  - mean nearest-TV `<= 0.366`
  - max nearest-TV `<= 0.38`
  - min phase-1 effective support `>= 5.0`
  - max phase-1 weight `<= 0.32`
- qualitative optima at `60M` and `300M` are at least as plausible as the
  current leader; do not stop on a model that buys metrics by becoming even
  more phase-1-tech-heavy or more collapsed

If you do **not** find something that good, keep searching.

Finding a modest improvement, near-tie, or benchmark-only win is **not** a
reason to stop. Those should be logged as interim results and the search should
continue.

Benchmark-only wins do not count as a stopping condition.

A claim that the space is exhausted must be scoped narrowly:

- allowed: "the current linear-head GRP-body family looks saturated"
- not allowed: "the direct-law search is exhausted"

Unless a ship-level serious candidate is found, keep searching until the human
explicitly stops the run.

Do not stop just because:

- several directions failed
- one architecture family looks saturated
- the next step is harder or more speculative

If you are making progress, finding informative negatives, or opening a new
family with a coherent rationale, continue rather than stopping.

If you do choose to pause because you are blocked on something external rather
than because the search is exhausted, the logbook must say exactly what the
blocker is and what the next runnable direction should be.

## Suggested Opening Move

Start from the refreshed local lead as a reference:

- `gated_ci_scale_n_rpm200`

And the strong simple baseline:

- `gated_centered_interactions`

Suggested first move:

1. reproduce the refreshed holdout counts to confirm the environment
2. reproduce `gated_centered_interactions` and `gated_ci_scale_n_rpm200`
3. test the smallest simplification or optimizer/plausibility improvement you
   can justify
4. only then widen the search if the simple branch is clearly saturated

The important thing is to preserve the high-level properties:

- end to end
- small and motivated
- refreshed-basis valid
- geometry-aware
- simplicity-biased unless added complexity clearly earns itself
