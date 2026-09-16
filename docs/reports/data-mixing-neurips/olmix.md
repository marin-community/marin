# Olmix Implementation Review And Subset-Fit Findings

## TL;DR

- The released Olmix source code confirms that the exact convex part is the proposer, not the log-linear fit.
- We implemented a paper-faithful two-phase exact proposer in Marin and reran the local Olmix subset benchmark.
- It does not materially change the subset-fit optima on our current packet.
- The current bad Olmix subset validations are explained by fit instability and basin selection, not by our previous softmax-plus-L-BFGS-B solve.

## Status

Stable enough for paper background. This is strong enough to support how we describe Olmix in a paper or slide deck. It is not, by itself, a reason to rerun the full Olmix subset validation sweep.

## Question

We had two competing explanations for the bad Olmix subset validations:

1. our two-phase adaptation was too far from the released Olmix method, especially in the optimizer, or
2. the fitted surrogate itself was unstable, and the final optimizer was not the main problem.

The released source code lets us separate those two.

## What The Released Olmix Code Actually Does

Repository:

- `/Users/calvinxu/Projects/Work/Marin/data-mixture/olmix`

Key files:

- `olmix/fit/law.py`
- `olmix/fit/utils.py`

What matters:

- The log-linear fit is not closed-form. It uses multistart `LBFGS` with Huber loss in the original target space.
- The proposer for the optimized mixture is exact and convex. It optimizes directly over simplex weights with `cvxpy`.
- KL regularization is applied directly to the simplex weights, not through a softmax reparameterization.

So the released implementation matches the paper on the proposer side, but not in the sense of having a closed-form regression fit.

## What We Changed In Marin

We added a paper-faithful two-phase exact proposer to:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/two_phase_many_olmix_loglinear_sl_verb.py`

The exact solver:

- optimizes directly on the two phase simplices,
- uses the same fitted surrogate we already had,
- and applies weighted KL to the natural prior in each phase.

We then reran the subset benchmark with the exact proposer using:

```bash
uv run --with cvxpy --with torch python \
  /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_olmix_loglinear_subset_optima.py \
  --solver cvxpy
```

Outputs:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_olmix_loglinear_subset_cvxpy_curve_points.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_olmix_loglinear_subset_cvxpy_summary.json`

## Main Result

The exact proposer and the existing softmax-plus-L-BFGS-B proposer produce essentially the same subset-fit optima on the current packet.

Across all 9 subset sizes:

- chosen observed run changed on `0/9`
- mean absolute change in predicted optimum value: `0.000060`
- max absolute change in predicted optimum value: `0.000138`
- mean phase-TV distance between the two deployment schedules: `0.001350`

Representative rows:

| Subset size | Old predicted optimum | Exact predicted optimum | Validated BPB | Takeaway |
| --- | ---: | ---: | ---: | --- |
| `20` | `0.267129` | `0.267170` | `1.396656` | Still pathological |
| `140` | `1.079932` | `1.079929` | `1.067286` | Still one of the good subset sizes |
| `180` | `1.078177` | `1.078180` | `1.067173` | Still one of the good subset sizes |
| `242` | `0.807187` | `0.807228` | `2.534581` | Still pathological |

## Interpretation

This rules out the strongest optimizer-mismatch explanation.

The current evidence says:

- our previous proposer was not the main reason Olmix failed on the pathological subset sizes,
- matching the released exact proposer does not rescue those cases,
- and the instability is upstream in the surrogate fit or basin selection.

That also means we should be careful how we describe the result:

- it is fair to say our earlier Marin implementation diverged from the released Olmix proposer,
- but it is no longer fair to say that divergence explains the bad subset-fit Olmix validations.

## Important Caveat About The Historical `1.069` Olmix Number

The `1.069` uncheatable Olmix number used in presentation tables is likely a real historical validated metric, but its currently exported phase weights appear to be recomputed from current local code rather than pulled from the original historical fit artifact.

That means:

- the metric itself is probably usable,
- the currently exported weights for that historical row should not be treated as provenance-clean,
- and direct weight comparisons between that row and the subset-fit reruns are still suspect until provenance is fixed.

See the chronological debug log for details:

- [Olmix subset debug log](../../debug-log-olmix-subset-fit.md)

## What This Means For Paper Writing

The safe paper-facing claim is:

> We reviewed the released Olmix code and implemented a matching exact proposer for our two-phase setting. On our packet, this does not materially change the subset-fit Olmix optima. The remaining failure mode is instability in the fitted surrogate, not in the final optimization step.

What I would avoid claiming:

- that our original bad Olmix results were caused mainly by the wrong proposer,
- or that Olmix as a method is refuted by these runs alone.

## Next Steps

If we want to improve the Olmix comparison further, the next useful work is:

1. multi-seed surrogate fitting with explicit seed selection criteria,
2. rejection of collapsed solutions using support or nearest-TV guards,
3. and provenance cleanup for the historical validated Olmix row.
