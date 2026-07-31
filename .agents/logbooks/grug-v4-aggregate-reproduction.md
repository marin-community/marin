# Grug v4 Aggregate Reproduction: Research Logbook

## Scope
- Goal: Reproduce the collaborator's aggregate-metric and mixture-generation
  notebook that may have produced the Grug-MoE v4 mixture.
- Primary artifacts:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/`.
- Source gist:
  https://gist.github.com/Helw150/de1cd95cf1617d5d812fd6dad6d17d17.

## Experiment Log

### 2026-05-25 - Reproduce collaborator aggregate from sent matrix zip
- Hypothesis:
  The v4 mixture may be recoverable by running the collaborator's Marimo
  notebook against the exact raw metric matrix packet sent to them.
- Input:
  `/Users/calvinxu/Downloads/raw_metric_matrix_300m.zip`. The zip contains
  `raw_metric_matrix_300m/raw_metric_matrix_300m.csv`,
  `raw_metric_matrix_300m/noise_baseline_run00097_300m.csv`, fixed/variable
  subset noise files, README, and summary JSON.
- Command:
  `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/reproduce_collaborator_grug_v4_aggregate.py --dataset-name sent_raw_metric_matrix_300m_zip --raw-csv experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_zip_input/raw_metric_matrix_300m/raw_metric_matrix_300m.csv --noise-csv experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/sent_zip_input/raw_metric_matrix_300m/noise_baseline_run00097_300m.csv --download-gist`
- Method:
  Ported the gist's pipeline from Marimo/polars into
  `reproduce_collaborator_grug_v4_aggregate.py`: select task columns with the
  gist's rules, standardize oriented metrics, compute the 5-factor varimax
  aggregate, fit the 235-parameter per-phase/domain epoch-loss form with
  5-fold CV, then run the bootstrap LCB optimizer and Pareto-blend scan.
- Result:
  The exact sent zip has 242 completed rows, 1209 columns, and all 41
  gist-selected task columns complete. The aggregate target reproduced cleanly:
  `corr(mean, factor)=0.797`, `corr(mean, pc1)=0.958`, `corr(pc1, factor)=0.685`.
  The 235-parameter epoch-loss fit achieved `R2_cv=0.5987`.
- Mixture result:
  The raw LCB optimum emphasizes `dolmino_synth_math`, `dolma3_arxiv`,
  `dolmino_stem_heavy_crawl`, `dolmino_synth_instruction`,
  `dolma3_wikipedia`, and `dolma3_finemath_3plus` by total epochs. This does
  not match the dashboard v4 weights: flattened phase-weight correlation with
  v4 is `0.454`, with mean per-phase TV `0.470`.
- Pareto-blend result:
  The strict no-regression Pareto scan selects `alpha*=0.0`, i.e.
  proportional. The resulting proportional-like blend is closer to v4
  (`corr=0.788`, mean per-phase TV `0.322`) but is not v4.
- Interpretation:
  The aggregate construction and model fit are reproducible from the sent data,
  but the notebook as written does not recreate dashboard v4. v4 likely used an
  additional decision step not represented in the gist output we reproduced:
  manual/consensus editing, a different alpha/guardrail rule, different
  notebook state, or a different optimizer output than the raw LCB/Pareto-safe
  blend.
- Current-registry caveat:
  Running the gist's selector on today's local
  `metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv` selects
  41 task columns but only 16 are complete; dropping incomplete columns gives a
  much weaker fit (`R2_cv=0.2102`). The current registry matrix is therefore not
  a drop-in replacement for the exact packet the collaborator used.

### 2026-05-25 - Test simple postprocessing from LCB optimum to v4
- Hypothesis:
  The dashboard v4 mixture may have been produced by applying a final softmax,
  temperature, blend, floor, cap, or similar postprocessing step to the raw LCB
  optimum from the collaborator notebook.
- Command:
  `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/test_grug_v4_lcb_postprocessing.py`
- Method:
  Compared dashboard v4 against transformed versions of the reproduced LCB
  optimum over the 39 training domains only. Tested arithmetic interpolation
  with proportional, phase-specific arithmetic interpolation, geometric/log
  softmax interpolation with proportional, softmax/power transforms of LCB
  weights, additive floors, clipping/capping, uniform interpolation, and
  epoch-space interpolation.
- Result:
  The best simple family is arithmetic interpolation between proportional and
  raw LCB: shared `alpha=0.326` gives mean per-phase TV `0.2305`, flattened
  phase-weight correlation `0.7889`, and top-8 overlap `6/8`; phase-specific
  `alpha0=0.32`, `alpha1=0.34` is only marginally better at TV `0.2303`.
  Geometric/log-softmax interpolation is worse: best TV `0.2744`, correlation
  `0.7324`, and top-8 overlap `4/8`. LCB-only softmax/floor/power transforms
  are much worse, with TV around `0.33-0.35` or higher.
- Interpretation:
  This does not support a final softmax/temperature explanation. The only
  plausible simple postprocessing is roughly one-third arithmetic movement from
  proportional toward LCB, but even that leaves large domain residuals: it
  underweights `dolma3_cc/food_and_dining_high` by about `0.123` total weight
  and overweights `dolma3_cc/entertainment_high` by about `0.088`. That pattern
  suggests v4 involved additional manual/consensus edits or a different
  objective/state, not just a deterministic softmax of the reproduced LCB
  weights.
- Artifacts:
  `reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/postprocess_tests/`.

### 2026-05-25 - Fit canonical DSP to collaborator aggregate
- Hypothesis:
  The collaborator factor aggregate may be better modeled by our canonical DSP
  surrogate than by the gist's 235-parameter per-phase epoch-loss model; the
  raw DSP optimum may clarify what the aggregate wants before any guardrails or
  human postprocessing.
- Command:
  `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/fit_grug_v4_aggregate_canonical_dsp.py`
- Method:
  Reused the exact sent `raw_metric_matrix_300m.zip` extraction and the gist's
  aggregate target construction. Fit target is higher-is-better `y_factor`, so
  the loss-like DSP formula was fit to `-y_factor`. The default variant was
  `dsp_effective_exposure_penalty_nnls`, matching our promoted canonical DSP
  convention.
- Result:
  Effective-exposure DSP fit 242 rows, 41 task columns, and 39 domains with
  158 parameters. Train RMSE/R2/Spearman was `0.1422 / 0.8961 / 0.9497`; OOF
  RMSE/R2/Spearman was `0.1968 / 0.8010 / 0.8961`. This is much stronger than
  the reproduced collaborator epoch-loss model's `R2_cv=0.5987` on the same
  aggregate.
- Raw optimum:
  The unconstrained raw optimum predicts `y_factor=3.0944`, a `+2.6126` gain
  over proportional in model space. It is far from the observed manifold:
  nearest observed row is `run_00183` at average phase TV `0.6875`; TV vs
  proportional is `0.7523`, vs dashboard v4 is `0.7456`, and vs reproduced LCB
  is `0.6961`. The materialized optimum is sparse (`13` phase-0 and `7`
  phase-1 domains above `1e-3`, with phase-1 max weight about `0.80`).
- Stability check:
  Re-running raw optimization with 128 and 512 random starts reached the same
  predicted value but different sparse materialized optima. The raw optimum is
  therefore best interpreted as a saturated/extrapolative plateau, not a stable
  deployment candidate.
- Naming disambiguation:
  Also ran the older code-key `canonical` variant
  (`dsp_phase_benefit_penalty_nnls`). It fit worse OOF (`RMSE=0.2285`,
  `R2=0.7318`, `Spearman=0.8726`) and produced an even more extreme raw optimum
  (`y_factor=7.2502`, phase-0 effectively all `dolmino_synth_math`, TV vs
  proportional `0.9315`). Effective-exposure remains the better canonical
  result here.
- Artifacts:
  `reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/canonical_dsp_sent_zip/`
  and
  `reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/phase_benefit_dsp_sent_zip/`.
