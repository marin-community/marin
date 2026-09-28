# Debugging log for GRP MMLU choice_prob_norm fit

## Initial status

GRP no-L2 fits MMLU BPB poorly on `300m_6b` (`qsplit240_core`), with OOF Spearman around `0.30-0.35`
after optimizer debugging. `choice_prob_norm` is a better accuracy-like target, but the first fit still
showed two concerns:

- default-family standard MMLU `choice_prob_norm` logit OOF Spearman was only `0.578`;
- raw simplex optima extrapolated to implausibly optimistic corner mixtures.

## Hypothesis 1: the default family partition is wrong for benchmark choice metrics

The default GRP partition treats several synthetic/QA-like domains as broad text. For MMLU-style choice
metrics, synthetic and QA domains may behave more like tech/reasoning signal.

### Result

Family scheme sweep on standard MMLU `choice_prob_norm` logit:

- `synthetic_tech`: OOF Spearman `0.598`, OOF RMSE `0.003467`
- `qa_reasoning`: OOF Spearman `0.591`, OOF RMSE `0.003478`
- `default`: OOF Spearman `0.578`, OOF RMSE `0.003501`
- `synthetic_reasoning`: OOF Spearman `0.548`, OOF RMSE `0.003560`

Conclusion: family assignment matters modestly. The best tested setting is `synthetic_tech`, but this
only improves rank fit by about `+0.020` Spearman.

## Hypothesis 2: the probability transform is the main bottleneck

Tested raw probability, logit, probit, arcsin-sqrt, and rank-normal diagnostics under `synthetic_tech`.

### Result

Probability-space transforms are close:

- logit: OOF Spearman `0.598`, OOF RMSE `0.003467`
- probit: OOF Spearman `0.593`, OOF RMSE `0.003484`
- raw: OOF Spearman `0.591`, OOF RMSE `0.003474`
- arcsin-sqrt: OOF Spearman `0.566`, OOF RMSE `0.003587`

Conclusion: logit is the best of the tested transforms, but target transform is not the main bottleneck.

## Hypothesis 3: NNLS monotonicity is too restrictive

Compared the existing NNLS head against unconstrained OLS and RidgeCV using the same `synthetic_tech`
nonlinear features.

### Result

- NNLS: OOF Spearman `0.598`, OOF RMSE `0.003467`
- RidgeCV: OOF Spearman `0.605`, OOF RMSE `0.003441`
- OLS: OOF Spearman `0.572`, OOF RMSE `0.003921`

Conclusion: NNLS is not the main problem. Ridge regularization gives a small improvement, but fully
unconstrained OLS is worse.

## Hypothesis 4: GRP feature blocks are mismatched

Tested feature-block ablations with fixed `synthetic_tech` nonlinear parameters.

### Result

- full: OOF Spearman `0.598`, OOF RMSE `0.003467`
- no pairs: OOF Spearman `0.605`, OOF RMSE `0.003419`
- signals only: OOF Spearman `0.539`, OOF RMSE `0.003604`
- family only: OOF Spearman `-0.116`, OOF RMSE `0.004645`

After re-optimizing nonlinear params for `no_pairs`, the result fell to OOF Spearman `0.585`, so pair
removal is not a reliable fix.

## Results

Best current GRP choice-probability diagnostic:

- Target: standard MMLU `choice_prob_norm`
- Transform: logit probability
- Family scheme: `synthetic_tech`
- Block variant: full GRP
- OOF RMSE: `0.003467`
- OOF Spearman: `0.598493`

Main remaining issue:

- raw simplex optimum still extrapolates to a corner-like mixture and predicts unrealistically high
  `choice_prob_norm` (`0.341` vs observed max `0.275`).

## Hypothesis 5: choice-probability needs constrained deployment, not raw simplex deployment

`choice_prob_norm` differs from BPB in two ways that matter for optimization:

- it is bounded and higher-is-better, so small target-space errors can create large apparent gains near
  the upper tail;
- in this 300M panel its observed range is narrow (`0.248` to `0.275`), so raw simplex optimization can
  exploit extrapolation directions that are not validated by any nearby observed mixture.

Target-geometry diagnostics on the same 240-row panel:

- uncheatable BPB range: `0.098574`;
- MMLU BPB range: `0.347474`;
- standard MMLU `choice_prob_norm` range: `0.026722`;
- SL-Verb MMLU `choice_prob_norm` range: `0.002742`;
- standard MMLU accuracy vs standard `choice_prob_norm` Spearman: `0.866`;
- standard MMLU accuracy vs standard `choice_logprob_norm` Spearman: `-0.026`;
- standard MMLU BPB vs standard `choice_logprob_norm` Spearman: `-0.970`;
- standard MMLU BPB vs standard accuracy Spearman: `-0.069`.

This explains the earlier behavior: MMLU BPB and choice log-probability mostly measure likelihood of the
full prompt/answers, while `choice_prob_norm` tracks the normalized mass on the correct choice and is much
closer to accuracy.

Added top-actual hull and trust-blend diagnostics to the fitting script.

### Result

For the current best setting (`synthetic_tech`, logit target, full GRP body):

- best observed standard-MMLU `choice_prob_norm`: `0.274843`;
- raw optimum predicted metric: `0.340919`, optimism `+0.066076`, nearest observed TV `0.748`;
- top-8-actual hull predicted metric: `0.268796`, nearest observed TV `0.002`;
- trust-blend cap `0.05` predicted metric: `0.274535`, optimism `-0.000307`, nearest observed TV `0.049`;
- trust-blend cap `0.10` predicted metric: `0.280172`, optimism `+0.005330`, nearest observed TV `0.100`.

Conclusion: raw optimization is not reliable for this bounded target. A small trust-region move from the
top-actual hull captures most of the predicted gain without the extreme phase-1 tech corner. If we use
`choice_prob_norm` as an optimization target, the default deployment rule should be a constrained hull or
trust-blend rule until a raw benchmark optimum has been validated.

## Future Work

- [x] Add constrained/trust-region optima for benchmark choice-probability objectives.
- [ ] Add a benchmark-specific regularized head if this target becomes central.
- [ ] Fix `components()` for configurations with family penalties but no family group inputs, or explicitly
      mark those block variants unsupported.
