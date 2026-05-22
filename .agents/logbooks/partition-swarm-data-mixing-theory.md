# Partition and Swarm-Based Data Mixing: Research Logbook

## Scope
- Goal: formalize data mixing under coarse partitions, swarm designs, local interventions, and scale-transfer surrogates.
- Current focus: connect partition expressivity, repetition-aware scaling, DSP/MCT-style mixture surrogates, SNR, and projected controllability.
- Primary empirical panels: 300M/6B raw metric matrix, issue #5416 aggregate, ND scaling packet, perturbation/intervention panels, and MoE/path-test validation tracks.

## Experiment Log

### 2026-05-21 - Review arXiv 2404.07177 and test repetition-aware forms
- Reference: Goyal et al., "Scaling Laws for Data Filtering: Data Curation cannot be Compute Agnostic", arXiv:2404.07177 v1.
- Source reviewed: `/Users/calvinxu/Zotero/storage/F7S8L7S6/arXiv-2404.07177v1.tar.gz`.
- Hypothesis from paper: limited high-quality data has higher initial utility but loses marginal utility under repetitions, so optimal filtering/mixture policy should depend on compute.
- Main fixed-size law:
  - Baseline loss curve: \(y = a n^b + d\), with \(b < 0\).
  - Instantaneous utility: \(\frac{dy}{dn} = \frac{y}{n}b\).
  - Repeated-sample utility: \(b_{k+1} = b 2^{-k/\tau} = b\delta^k\).
  - Epoch-wise repeated-data loss: \(y_k = a n_1^{b_1}\prod_{j=2}^{k}(n_j/n_{j-1})^{b_j}+d\).
  - For \(p\) equal pools, their mixture theorem uses \(\hat{\tau}_i=p\tau_i\) and \(b_{\mathrm{eff}}^{(k)} = p^{-1}\sum_i b_i\hat{\delta}_i^k\).
- Evidence in the paper:
  - DataComp medium pool, ViT-B/32 CLIP, 128M image-caption pairs.
  - T-MARS and CLIP-score partitions into quality buckets: top 10%, 10-20%, 20-30%, 30-40%.
  - Individual bucket scaling curves fit over 2-10 epochs; fitted \(b\) varies by quality and \(\tau\) varies by bucket.
  - High-quality bucket marginal utility falls below lower-quality buckets after enough repetitions.
  - LAION-style top-10% filtering is better early but becomes worse than unfiltered data after enough samples seen.
  - Mixture/filtering curves are predicted from individual bucket parameters without fitting on mixture-combination runs; scatter points are held-out mixture checks.
  - A second external check uses Cherti et al. CLIP-family models across architectures, pool sizes, and 3B-34B sample budgets.
- Caveats for our setting:
  - Their cleanest theory assumes a ranked quality ladder with equal-size buckets; Marin has 39 heterogeneous domains and two phases.
  - Their contrastive \(N^2\) comparison argument does not transfer directly to autoregressive language-model pretraining.
  - Their no-negative-penalty framing models diminishing positive utility; our DSP evidence often needs explicit harmful/overexposure penalty features.
  - Their mixture theorem assumes a simple interaction of half-lives under bucket merging; our partition basis has non-equal domains, repeated spans/tokens, and phase effects.

- Local 300M fixed-scale check:
  - Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_apple_repetition_variants_300m.py`
  - Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_apple_repetition_variants_300m_20260514/`.
  - Current checkout loaded 100 complete rows for the legacy uncheatable-BPB fit frame, not the full 242-row raw matrix.
  - Apple shared-\(r_1\) DSP: CV RMSE 0.040506, OOF Spearman 0.507135.
  - Apple per-domain-\(r_1\) DSP: CV RMSE 0.012125, OOF Spearman 0.925173.
  - Baseline old GRP no-L2: CV RMSE 0.011141, OOF Spearman 0.804042.
  - Interpretation: per-domain repetition half-life can rank well, but only with 39 extra nonlinear parameters and a far/sparse optimum; shared half-life is not enough.

- Clean 300M fixed-scale RAML check on the current raw matrix:
  - Command: `uv run --with numpy --with pandas --with scipy --with scikit-learn --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/fit_repetition_aware_data_filtering_law_300m.py`
  - Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/repetition_aware_data_filtering_law_300m_20260521/`.
  - Source rows: 242 signal rows.
  - Complete target rows: 100 for uncheatable BPB, 93 for issue #5416 aggregate. The current raw matrix is not complete enough for a 242-row aggregate fit.
  - Uncheatable BPB: constant fixed-scale baseline RMSE 0.055459, Spearman -0.033070; uniform-domain RAML RMSE 0.054776, Spearman 0.173165; positive per-domain value RAML RMSE 0.051973, Spearman 0.367417.
  - Issue #5416 loss: constant baseline RMSE 0.448819, Spearman -0.082002; uniform-domain RAML RMSE 0.449192, Spearman -0.030005; positive per-domain value RAML RMSE 0.463667, Spearman 0.252753; per-domain \(r_1\) signed-head variant Spearman 0.494032 but RMSE 0.788401 and optimizer did not converge cleanly.
  - Interpretation: on fixed 300M, the literal paper-style repetition form is weak; domain-value parameters create some rank signal, but this is still far below DSP-style rank fit and not a credible standalone optimizer.

- Existing issue #5416 aggregate Apple comparison:
  - Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_apple_repetition_issue5416_300m_20260514/`.
  - Rows: 242 signal rows, 10 variable-subset noise rows for projection, 26 selected aggregate items, 5 factors.
  - Split benefit/saturation/penalty DSP was best in that report: score CV RMSE 0.148561, OOF Spearman 0.918927.
  - Apple shared-\(r_1\): score CV RMSE 0.184027, OOF Spearman 0.877191.
  - Apple per-domain-\(r_1\): score CV RMSE 0.194039, OOF Spearman 0.862864.
  - Interpretation: for the aggregate target, explicit Apple-style repetition did not beat the split DSP form.

- Direct ND scaling check of the repetition-aware variable-size law:
  - Command: `uv run --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with kaleido --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/fit_repetition_aware_variable_size_law_nd.py`
  - Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/repetition_aware_variable_size_law_nd_20260514/`.
  - Grouped OOF rows: 641.
  - Literal scale-only baseline: RMSE 0.033012, Spearman 0.808008.
  - Uniform-domain repetition-aware law: RMSE 0.028837, Spearman 0.863451.
  - Positive per-domain value weights: RMSE 0.026197, Spearman 0.891984, optimizer abnormal termination.
  - Per-domain value + signed linear mix + per-domain \(r_1\): RMSE 0.025702, Spearman 0.895723, optimizer hit evaluation limit.
  - Comparison to frozen variable-scale DSP report: best DSP variants have similar RMSE around 0.0263-0.0268 but better OOF Spearman around 0.927-0.930.
  - Interpretation: repetition-aware scaling is a useful backbone and improves over scale-only, but by itself it does not replace DSP/MCT-style mixture geometry.

- Next actions:
  - Add a clean 242-row fixed-300M RAML-style fit on `raw_metric_matrix_300m.csv` for both uncheatable BPB and issue #5416 aggregate, separate from DSP feature variants.
  - Keep Apple/RAML ideas as scale/repetition structure candidates inside a richer mixture model, not as standalone mixture optimizers.
  - For theory writing, frame their result as evidence that optimal curation is compute-conditioned under repeated limited data, then distinguish that from our partition-polytope expressivity and actuation questions.

### 2026-05-21 - RAML-DSP hybrid and joint-scaling checks
- Hypothesis: the paper's repetition-aware curve may improve DSP if used as a per-domain repeated-exposure mechanism, either by replacing the saturation feature or by discounting physical exposure before the existing DSP saturation.
- Code changes:
  - Added RAML-phi variants to `experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_300m.py`.
  - Added focused comparison script `experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_raml_variants_300m.py`.
  - Generalized `experiments/domain_phase_mix/exploratory/two_phase_many/fit_variable_scale_dsp_nd.py` so the frozen DSP geometry can be supplied with `MARIN_FROZEN_DSP_MODEL_PATH` and outputs with `MARIN_VARIABLE_SCALE_DSP_OUTPUT_DIR`.
- Fixed-300M command:
  - `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_raml_variants_300m.py`
- Fixed-300M output:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_raml_variants_300m_20260521/`.
- Fixed-300M result:
  - Uncheatable BPB, 100 complete rows: Apple per-domain-\(r_1\) DSP was best, CV RMSE 0.012125 and OOF Spearman 0.925173. Current canonical DSP was CV RMSE 0.039204 and Spearman 0.715008. RAML-phi per-domain-\(r_1\) was second among repetition variants but worse than Apple-discounted exponential saturation, CV RMSE 0.022129 and Spearman 0.853561.
  - Issue #5416 loss, 93 complete rows: Apple per-domain-\(r_1\) DSP was best, CV RMSE 0.200074 and OOF Spearman 0.887321. Tied effective-exposure DSP had CV RMSE 0.447095 and Spearman 0.655158 on this complete-row subset. RAML-phi variants did not beat the Apple-discounted exponential saturation form.
- Fixed-300M interpretation:
  - The useful adaptation is not replacing \(1-\exp(-\rho_i z_i)\) with the RAML effective-repeat curve directly.
  - The useful adaptation is discounting physical repeated exposure before DSP's existing exponential saturation, with a per-domain \(r_{1,i}\).
  - This is a strong fixed-scale fit but increases domain-dependent parameters to 5 per domain, outside the earlier canonical DSP budget.
- Erratum after comparing against the stable `dsp_exact.py`/canonical outputs:
  - The large fixed-300M "improvements" above compare against a later focused script run whose canonical/effective-exposure baselines overfit badly: e.g. uncheatable train RMSE 0.000692 but CV RMSE 0.039204 for `dsp_phase_benefit_penalty_nnls`, and effective-exposure CV RMSE 0.114172.
  - The stable canonical DSP sweep in `reference_outputs/dsp_canonical_variants_300m_20260510/summary.csv` has much better uncheatable fits: `dsp_effective_exposure_penalty_nnls` CV RMSE 0.007106 and OOF Spearman 0.919645; `dsp_saturation_penalty_split_nnls` CV RMSE 0.006407 and OOF Spearman 0.929312.
  - Against that stable baseline, Apple per-domain-\(r_1\) is not a meaningful uncheatable-BPB improvement: CV RMSE is worse at 0.012125, OOF Spearman is only slightly above effective-exposure (0.925173 vs 0.919645), and it is below split DSP's Spearman 0.929312.
  - The existing 242-row issue #5416 aggregate check also does not support promotion: `dsp_phase_benefit_saturation_penalty_nnls` CV RMSE 0.148561 and OOF Spearman 0.918927; Apple per-domain-\(r_1\) CV RMSE 0.194039 and OOF Spearman 0.862864.
  - Corrected conclusion: per-domain repetition half-life is a useful diagnostic/high-capacity comparator, but it has not beaten the stable canonical/split DSP implementations.
- Joint ND commands:
  - Effective-exposure frozen geometry baseline rerun: `MARIN_VARIABLE_SCALE_DSP_OUTPUT_DIR=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/variable_scale_dsp_effective_geometry_nd_20260521 uv run --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with kaleido python experiments/domain_phase_mix/exploratory/two_phase_many/fit_variable_scale_dsp_nd.py`
  - Apple per-domain-\(r_1\) frozen geometry: `MARIN_FROZEN_DSP_MODEL_PATH=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_apple_repetition_variants_300m_20260514/dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls/model.json MARIN_VARIABLE_SCALE_DSP_OUTPUT_DIR=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/variable_scale_dsp_apple_per_domain_nd_20260521 uv run --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with kaleido python experiments/domain_phase_mix/exploratory/two_phase_many/fit_variable_scale_dsp_nd.py`
- Joint ND result:
  - Effective-exposure frozen geometry remains better: best grouped OOF RMSE 0.026334, Spearman 0.927391 for split amplitude, and exposure-scaled Spearman 0.930388.
  - Apple per-domain-\(r_1\) frozen geometry is worse in the same scaffold: best grouped OOF RMSE about 0.030166, best Spearman about 0.910978.
  - Apple geometry improves regret for the exposure-scaled variant to 0.0 but gives lower RMSE/rank and worse leave-scale behavior on important scales.
- Joint ND interpretation:
  - Per-domain repetition half-life is valuable for fixed 300M interpolation, but freezing that high-capacity geometry and transferring it across scale is not a joint-modeling improvement.
  - The current best practical joint path remains MCT or effective-exposure variable-scale DSP; Apple-style repetition should be revisited only with proper cross-scale retuning/regularization, not as a 300M-frozen geometry.
- Next action:
  - Do not promote RAML-phi as canonical.
  - Treat Apple per-domain-\(r_1\) as a high-capacity fixed-scale comparator and a candidate regularized term for future joint models.
