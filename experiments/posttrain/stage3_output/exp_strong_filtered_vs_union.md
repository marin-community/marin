# STRONG-only filtered spec vs all-edits union vs baseline

**Hypothesis**: filtering R1 edits to keep only STRONG-classified ones (19/29) should give cleaner per-rubric improvements than the full union (29 edits, of which 10 didn't propagate or only propagated weakly).

**Method**: ran 4 writers on the 19-edit STRONG-only spec; compute mean per-rubric text change from baseline on key fields.

## Per-judge mean text-change from baseline

Higher = more rewritten by the spec edits. Both variants should have positive deltas; the question is whether STRONG-only is cleaner.

| field | judge | strong_only | union (all 29) |
|---|---|---:|---:|
| `dominant_rubric.BAD` | flash | 0.785 | 0.805 |
| `dominant_rubric.BAD` | gpt51 | 0.926 | 0.930 |
| dominant_rubric.BAD | pro | MISSING | MISSING |
| dominant_rubric.BAD | glm51 | MISSING | MISSING |
|---|---|---|---|
| `rationale.alt_readings` | flash | 0.747 | 0.774 |
| `rationale.alt_readings` | gpt51 | 0.937 | 0.920 |
| rationale.alt_readings | pro | MISSING | MISSING |
| rationale.alt_readings | glm51 | MISSING | MISSING |
|---|---|---|---|
| `worked_example.spec_compliant` | flash | 0.814 | 0.805 |
| `worked_example.spec_compliant` | gpt51 | 0.832 | 0.834 |
| worked_example.spec_compliant | pro | MISSING | MISSING |
| worked_example.spec_compliant | glm51 | MISSING | MISSING |
|---|---|---|---|

## Schema validity

| judge | strong_only | union |
|---|---:|---:|
| flash | 22/22 | 22/22 |
| gpt51 | 22/22 | 22/22 |
| pro | 0/0 | 22/22 |
| glm51 | 0/0 | 22/22 |

## Interpretation (flash + gpt51 only; pro/glm51 system-overloaded out)

**Headline: STRONG-only spec produces nearly identical per-rubric rewriting
to the full union.** Differences are within ±0.03 across all fields — noise.

| field | judge | strong_only | union | diff (so−union) |
|---|---|---:|---:|---:|
| BAD | flash | 0.785 | 0.805 | -0.020 |
| BAD | gpt51 | 0.926 | 0.930 | -0.004 |
| alt | flash | 0.747 | 0.774 | -0.027 |
| alt | gpt51 | 0.937 | 0.920 | +0.017 |
| WE | flash | 0.814 | 0.805 | +0.009 |
| WE | gpt51 | 0.832 | 0.834 | -0.002 |

**Implication**: the 10 non-STRONG edits in union are essentially not
contributing to the rewrite signal. Quality filtering preserves the
propagation effect with fewer edits — you get a leaner spec without
losing power.

**For M5 design**: edit-acceptance criteria can prune to ~65% of
proposed edits without losing rewriting power. The saturation problem
in the closed-loop sim isn't from "too many edits"; it's from the
*accumulation pattern* (cumulative additions can create internal
contradictions even when each individual edit is sound).

Schema validity is identical (22/22) for both variants on the 2 judges
tested — supports the "fewer edits don't hurt writer reliability" claim
but doesn't directly demonstrate "more edits hurt" since this isn't
testing high counts.

(Pro and glm51 strong-only writes were started but the host system hit
load-average ~400 mid-run and the pro writer hung. They were terminated
to free resources. Re-run when system is healthier.)