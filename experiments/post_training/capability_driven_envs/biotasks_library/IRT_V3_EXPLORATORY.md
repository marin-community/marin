# Exploratory Bayesian 2PL on frozen BioTasks v3

**Diagnostic only, 2026-11-07.** No new model calls; this analysis reads the
private, immutable v3 OpenRouter response snapshot (SHA-256
`acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda`).
The tasks remain provisionally referenced, ShellSim/provider format failures
are non-ignorable, and these results are **not** calibrated item parameters
or scientific comparisons of biological competence.

## Model and reproducibility

`irt_v3.stan` fits an identifiable, prior-anchored, hierarchical 2PL:

```
P(correct | scored, model m, task i) = logistic(a_i * (theta_m - b_i))
a_i = exp(log_a_i) > 0
```

Model abilities `theta` are centered to zero; their scale is anchored by
`theta_raw ~ Normal(0,1)` and the slope priors, **not** identified from an
external population. Difficulties are partially pooled with mean prior
`Normal(0,1.5)` and half-normal SD prior `Normal+(0,0.8)`; log slopes with
mean prior `Normal(0,0.35)` and half-normal SD prior `Normal+(0,0.35)`.
There is no guess/lapse parameter. Fitting uses four CmdStan 2.38.0 NUTS
chains, 600 warmup and 600 retained iterations each, `adapt_delta=0.95`,
seed 324. **Changing the discrimination priors is a necessary sensitivity
check before trusting individual slopes.** The respondent sample is only 12
models, and the posterior is conditional on the current tasks, routes, and
scoring policy.

Primary analysis: retain 1,495 correct and **all 855 scored-zero** responses,
including **396 failed final submissions** (malformed JSON, etc.). Omit 360
ungraded and 50 infrastructure-missing **pairs**; no missing scores are
imputed. The secondary *graded-only* fit omits those 396 zero-valued
submission failures too (1,954 observations total) to expose sensitivity to
submission policy. This is not an alternative official score. Neither fit
models the missing/submission process, so neither establishes causal model
ability or missing-at-random.

To reproduce on a trusted machine with access to the private snapshot, first
verify its SHA-256 and save it locally (do not check it into git); then install
CmdStan and run:

```bash
uv run --with cmdstanpy python experiments/post_training/capability_driven_envs/biotasks_library/fit_irt_v3.py \
  --input /tmp/biotasks-v3-responses-private.jsonl \
  --output /tmp/biotasks-irt-v3-all-scored --mode all_scored
uv run --with cmdstanpy python experiments/post_training/capability_driven_envs/biotasks_library/fit_irt_v3.py \
  --input /tmp/biotasks-v3-responses-private.jsonl \
  --output /tmp/biotasks-irt-v3-graded-only --mode graded_only
```

The script checks matrix uniqueness, all score/status combinations, and the
snapshot hash; writes per-model summaries, **private per-item posteriors**, and
raw Stan draws only under the requested local output directory. The snapshots
and numeric item parameters are not committed.

## What the posterior says

| Fit | Scored pairs | Divergences | Max parameter R-hat | Min bulk ESS | Median item log-slope posterior SD | Median 95% slope interval width |
|---|---:|---:|---:|---:|---:|---:|
| All scored, **primary** | 2,350 | 0 | 1.009 | 1,066 | **0.474** | **5.64** |
| Graded-only, sensitivity | 1,954 | 0 | 1.008 | 819 | 0.237 | 2.02 |

The primary median item slope is 2.42, but its typical **95% interval spans
5.64 slope units**. That is broad: slopes are not individually precise even
though the sampler converged. Thirteen tasks are all correct and 23 all wrong
among their scored respondents; 194 have both outcomes. Individual tasks have
just **5–12 scored models** (median 10). The graded-only fit's narrower slope
intervals must **not** be read as more item information: the posterior
hierarchical slope SD collapses toward ~0.11 (primary ~0.48), pooling item
slopes together when failures are dropped. This is a vivid prior/selection
sensitivity, not evidence that all items have the same true discrimination.

The primary posterior places Nemotron and Kimi close together at the top
(about 50% and 45% posterior probability of ranking first, respectively), but
these are **scored-outcome rankings under one provider contract**, not a
leaderboard of biological reasoning. Removing submission failures changes
positions substantially (e.g. DeepSeek moves from third to the top group;
Grok from fifth to eighth). Llama has no scored correct responses and heavy
format/tool problems; its fitted ability is principally a measurement of the
current interface mismatch. Scores also depend on references not independently
validated. Model rank differences here are not scientific claims.

## How many more models?

For a **next diagnostic checkpoint**, target **30–50 total independently
routed respondents**, i.e. roughly **18–38 more models** spanning low,
mid, and high scores under the **same frozen contract**. This is a ballpark,
not a power calculation: ignoring pooling and selection, shrinking the median
log-slope posterior SD from 0.47 to ~0.30 suggests roughly
`12 * (0.47/0.30)^2 ≈ 30` models total *if* new respondents provide similar
information. That extrapolation is optimistic because current posteriors are
prior-dependent, 36 items are at score extremes, failures are model-specific,
and the current 12 routes were not a random sample.

For **stable individual item discrimination** rather than a diagnostic,
expect **many dozens to hundreds** of suitably diverse respondents; even
**100+ total models is a checkpoint, not a guarantee**. Collecting more
near-identical strong models will not identify slopes on universally easy
items, nor weak models on universally hard items. Before paying for a large
panel: independently check references/instructions, resolve ShellSim/provider
format failures, use a small diverse new-model tranche, refit with weaker
slope priors and a submission-process sensitivity, and stop only when the
**item-level slope intervals and posterior predictive checks** meet a
predeclared precision target. The 58+17 quarantined tasks stay excluded.
