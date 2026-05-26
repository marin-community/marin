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

### 2026-05-25 - ChatGPT Pro factor-DSP dashboard packet
- Goal: prepare a self-contained packet for parallel external analysis sessions before implementing the factor-DSP constrained optimization dashboard.
- Packet:
  - Directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/chatgpt_pro_factor_dsp_packet_20260525/`
  - Zip: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/chatgpt_pro_factor_dsp_packet_20260525.zip`
  - Builder: `experiments/domain_phase_mix/exploratory/two_phase_many/build_chatgpt_pro_factor_dsp_packet.py`
- Included data:
  - 300M raw metric matrix snapshots and fixed/variable/proportional noise panels.
  - Clean-slate aggregate/factor outputs, per-metric effective-exposure DSP controllability diagnostics, and reactive metric tables.
  - Canonical/split/effective DSP summaries, perturbation agreement tables, MoE v4 path-test data, Grug v4 aggregate reproduction artifacts, and current theory notes.
  - Packet-local self-contained starter script `code/analysis_starter.py` with PEP 723 dependencies.
- Validation:
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/build_chatgpt_pro_factor_dsp_packet.py`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/build_chatgpt_pro_factor_dsp_packet.py`
  - `unzip -t experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/chatgpt_pro_factor_dsp_packet_20260525.zip`
  - `uv run --script experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/chatgpt_pro_factor_dsp_packet_20260525/code/analysis_starter.py --packet-root experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/chatgpt_pro_factor_dsp_packet_20260525 --output-dir /tmp/chatgpt_pro_factor_dsp_packet_smoke`
- Smoke result:
  - Packet directory size: 21M.
  - Zip size: 6.6M.
  - Manifest file count: 89.
  - Starter script loaded `raw_metric_matrix_300m.csv` with shape `(242, 1445)` and found 423 numeric metric-like columns with at least 50% finite coverage.
- Interpretation:
  - This packet is meant to ask for statistical critique of factor aggregation, uncertainty-aware constrained optimization, DSP surrogate assumptions, low-SNR/low-controllability metric handling, and the theory framing.
  - It intentionally marks Grug-MoE path analysis as interim because path eval coverage is still being filled.

### 2026-05-26 - Initial factor-DSP constraint dashboard
- Goal: create an interactive Marimo dashboard for choosing candidate mixtures under factor/DSP targets, guardrails, and trust-region constraints.
- Code:
  - Notebook: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - Helpers: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py`
  - Tests: `tests/test_factor_dsp_constraint_dashboard.py`
- Dashboard scope:
  - Loads existing 300M signal matrix, clean-slate aggregate/factor scores, aggregate effective-exposure DSP raw optima, issue #5416 Pareto candidates, factor loadings, metric controllability, and Pareto item/group deltas.
  - Provides sliders for target gain, nearest-observed TV, max phase weight, minimum group guardrail delta, and minimum item guardrail delta.
  - Visualizes candidate frontier, selected mixture weights vs proportional, factor loading interpretation, and metric SNR/controllability diagnostics.
  - Writes local dashboard snapshots under `reference_outputs/factor_dsp_constraint_dashboard_20260526/`.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`
- Caveat:
  - This first pass consumes existing fit artifacts. It does not refit per-task DSP models inside the UI or solve arbitrary nonlinear constrained optimization from scratch. It ranks and filters the existing observed rows, aggregate raw DSP optima, and issue #5416 Pareto candidates.

### 2026-05-26 - Per-task sliders added to factor-DSP dashboard
- Issue: the initial dashboard only exposed an aggregate/factor target-gain slider and global item/group guardrail thresholds; it did not expose per-task target sliders.
- Change:
  - Added observed per-item deltas from `target_scores.csv` against `baseline_proportional`.
  - Added per-task threshold sliders for the 26 issue #5416 selected items.
  - Sliders are in variable-noise standard deviation units; the inactive floor is `-20`, positive values require improvement relative to proportional.
  - Filtering now requires candidates to satisfy all active task thresholds using observed signal-row deltas and precomputed issue #5416 Pareto candidate item deltas.
- Caveat:
  - Aggregate raw DSP optima still do not have per-task predictions in the dashboard. They are filtered out by active per-task constraints unless the "keep candidates with no active per-task predictions" checkbox is enabled.
- Validation:
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py -q`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`

### 2026-05-26 - y_factor candidate-library cache for factor-DSP dashboard
- Goal: make the constraint dashboard a fast trust-region recommender by precomputing many candidate mixtures and canonical DSP scores offline, rather than sampling/scoring inside Marimo.
- Code:
  - Builder: `experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py`
  - Tests: `tests/test_factor_dsp_candidate_library.py`
- Cache:
  - Directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/`
  - Files: `candidate_summary.parquet`, `candidate_summary.csv`, `candidate_weights_wide.parquet`, `candidate_top_domain_deltas.parquet`, `observed_runs.csv`, `summary.json`.
  - Candidate count: 524,564 total = 524,288 Sobol-logit trust-region candidates, 242 observed signal rows, 4 canonical DSP named mixtures, and 30 interpolation-path candidates to named DSP mixtures.
- Sampling/scoring:
  - Sobol candidates are generated as two-phase logit tilts around proportional.
  - Directions are centered and normalized in local \(L^2(p)\)/Fisher coordinates, \(\sum_i p_i v_i=0\) and \(\sum_i p_i v_i^2=1\).
  - Radius uses \(\alpha \in [0, 0.45]\), sampled by a scrambled Sobol dimension.
  - The score is current `y_factor` only, using the canonical `dsp_effective_exposure_penalty_nnls` model.
- Diagnostics:
  - The builder stores predicted `y_factor` gain vs proportional, one-RMSE lower confidence gain, average phase TV to proportional, nearest observed TV, max phase weight, support, entropy/effective support, and a basic deployability gate.
  - It validates the model formula by reproducing cached canonical-DSP predictions for `proportional`, `dashboard_v4`, `collaborator_lcb`, and `raw_dsp_optimum`.
- Run result:
  - Full build elapsed time: about 10.1 seconds on the local M3 Max.
  - Cache sizes: 59M `candidate_summary.parquet`, 206M `candidate_weights_wide.parquet`, 1.7M `candidate_top_domain_deltas.parquet`.
  - Canonical DSP fit metadata: OOF RMSE 0.1968, OOF Spearman 0.8961.
  - Sobol-trust candidates are close to observed manifold by construction: nearest-observed TV median 0.1268, q95 0.1781, q99 0.1885.
  - The best unconstrained rows remain the known extrapolative DSP endpoints/interpolations; the cached summary therefore keeps nearest-TV and max-weight diagnostics first-class.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
- Caveat:
  - This cache only predicts the current factor aggregate. It does not solve per-task constrained optimization yet because we still need per-task/factor uncertainty surfaces or a factor-space task surrogate for reliable slider back-propagation.

### 2026-05-26 - Endpoint discovery cache for extrapolative DSP directions
- Goal: augment the proportional-centered candidate library with explicit DSP endpoint discovery, so the dashboard can show far extrapolative directions as hypotheses rather than hoping local Sobol-logit sampling finds them.
- Code:
  - Added endpoint-discovery mode to `experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py`.
  - Added tests for softmax logits, KL-to-proportional, and endpoint optimization in `tests/test_factor_dsp_candidate_library.py`.
- Method:
  - Optimize the canonical `dsp_effective_exposure_penalty_nnls` `y_factor` surrogate directly over two phase logits.
  - Objective: predicted `y_factor` minus \(\lambda\,\mathrm{KL}(w\|p)\) minus a squared excess penalty above max phase weight 0.50.
  - Uses analytic DSP gradients through the softmax logits rather than finite-difference gradients.
  - Runs a KL regularization path with \(\lambda \in \{0,0.02,0.05,0.1,0.2,0.5,1,2,5,10\}\), 21 deterministic starts, and 16 Sobol-logit random starts.
- Cache:
  - Directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/endpoint_discovery/`
  - Files: `endpoint_summary.parquet`, `endpoint_weights_wide.parquet`, `endpoint_path_summary.parquet`, `endpoint_path_weights_wide.parquet`, `endpoint_top_domain_deltas.parquet`, `summary.json`.
  - Endpoint count: 10.
  - Interpolation path count: 110, using \(t \in \{0,0.05,0.1,0.2,0.25,0.35,0.5,0.65,0.75,0.9,1\}\).
- Run result:
  - Full endpoint build elapsed time: about 13.9 seconds.
  - Unregularized endpoint predicted gain vs proportional: 2.6126, matching the earlier raw DSP optimum scale.
  - Heavily regularized endpoint at \(\lambda=10\): predicted gain 0.8835, nearest-observed TV 0.0905, max phase weight 0.1861.
  - The \(t=1\) endpoint at \(\lambda=0.5\) passes the basic dashboard gate with predicted gain 2.5213, nearest-observed TV 0.4448, and max phase weight 0.2745.
  - The \(t=1\) endpoint at \(\lambda=2\) is much safer: predicted gain 2.1311, nearest-observed TV 0.3051, and max phase weight 0.1731.
- Caveat:
  - Two middle KL values hit the 1000-iteration cap; convergence status and optimizer messages are preserved in `endpoint_summary.parquet`. These are still useful as candidate directions, but the dashboard should surface optimizer status rather than treating every endpoint as equally finalized.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`

### 2026-05-26 - Factor-DSP dashboard wired to candidate caches
- Goal: make the Marimo dashboard use the precomputed y_factor candidate caches instead of only observed rows and named DSP mixtures.
- Code:
  - Dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - Helpers: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py`
  - Tests: `tests/test_factor_dsp_constraint_dashboard.py`
- Changes:
  - Loaded `candidate_summary.parquet` and `endpoint_discovery/endpoint_path_summary.parquet` into the candidate frontier.
  - Added candidate-source filtering for observed rows, canonical DSP named mixtures, Sobol-logit trust candidates, canonical DSP paths, and endpoint-discovery paths.
  - Bounded the Plotly frontier to 20,000 plotted points while preserving the top 2,000 target-gain rows.
  - Bounded the candidate selector to the top 2,000 filtered rows.
  - Lazy-loads the selected candidate weights from `candidate_weights_wide.parquet` or `endpoint_path_weights_wide.parquet`; observed/named weights remain eager.
  - Default selected candidate is the safer endpoint path `path_endpoint_kl2p0_mw2p0_t1p0`.
  - Snapshot export now writes only the top 50,000 filtered candidates to avoid large incidental CSVs.
- Render result:
  - Exported dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html`
  - Default filtered snapshot has 50,000 rows after cap; top rows are endpoint-discovery paths.
  - Default selected weights are for `path_endpoint_kl2p0_mw2p0_t1p0`.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with pyarrow marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`
- Caveat:
  - Per-task sliders still only have empirical observed-row deltas. Cached model-only candidates can be kept or excluded under active per-task constraints, but they do not yet have per-task surrogate predictions.

### 2026-05-26 - Factor-DSP dashboard task-slider UX correction
- Goal: make the per-task target controls match the intended semantics: every task slider starts at `0 = proportional`, with visible tick stops, and task constraints are activated explicitly rather than via an artificial inactive sentinel.
- Code:
  - Dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - Helpers/tests: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py`, `tests/test_factor_dsp_constraint_dashboard.py`
- Changes:
  - Removed the old `-20` inactive-floor slider behavior.
  - Added symmetric task-slider steps from `-10` to `10` in `0.5` standardized-delta units, centered at `0`.
  - Added an explicit `lock` checkbox per task. Locked tasks become candidate constraints at the current slider value; unlocked sliders are informational.
  - Reworked the task-control layout into aligned `Lock? / Task / Target delta vs proportional` rows.
  - Updated the active-constraint table to record locked constraints directly.
- Render result:
  - Re-exported `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html`.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with pyarrow marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`
- Caveat:
  - The deeper requested UX, where moving one task slider recomputes a mixture and moves all other task sliders to predicted outcomes, is still not implemented. That requires per-task/factor predictive surfaces for cached candidates; the current dashboard only filters by observed empirical task deltas or keeps/excludes model-only candidates when those deltas are missing.

### 2026-05-26 - Factor-DSP dashboard reactive task surrogate and epoch diagnostics
- Goal: implement the next dashboard interaction step: locked task sliders should select feasible candidates from the cached library, and the UI should show the candidate-implied values for all task sliders plus materialized epochs.
- Code:
  - Dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - Helpers/tests: `experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py`, `tests/test_factor_dsp_constraint_dashboard.py`
  - Task cache builder: `experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_task_prediction_cache.py`
- Task-response method:
  - Built `task_prediction_wide.parquet` for 524,674 dashboard candidates and 41 selected tasks.
  - Prediction source is explicitly labeled `local_weight_ridge`.
  - The local surrogate maps centered two-phase weights to standardized oriented task deltas; observed rows are overwritten with empirical deltas when candidate names match.
  - This is a dashboard responsiveness approximation, not a replacement for canonical DSP or a claim that low-SNR tasks are reliably controllable.
- UI changes:
  - Locked task sliders now filter candidates using the task-prediction cache when available.
  - The selected/recommended candidate displays `Candidate-implied task slider positions`, colored by locked-pass, locked-fail, or unlocked.
  - Added `Task-response surrogate quality` with per-task train RMSE/Pearson to flag tasks not captured well by this approximation.
  - Added `Materialized epochs` summary and plot for the selected candidate, using DSP epoch scales recovered from the canonical mixture metadata.
- Cache files:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/task_prediction_wide.parquet`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/task_prediction_metrics.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/task_prediction_metrics.parquet`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_candidate_library_y_factor_20260526/task_prediction_summary.json`
- Render result:
  - Re-exported `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html`.
- Validation:
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_task_prediction_cache.py`
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_task_prediction_cache.py`
  - `uv run --with marimo marimo check experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with pyarrow marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`
- Caveat:
  - Marimo cannot programmatically change the slider widget values themselves from another cell; the dashboard instead shows a candidate-implied task-position plot/table. This preserves the distinction between slider targets/constraints and predicted candidate outcomes.

### 2026-05-26 - Factor-DSP dashboard recommendation reactivity fix
- Issue: dragging a task slider appeared to do nothing because only checked `lock` boxes activated constraints, and the manual candidate dropdown could continue masking the best feasible recommendation.
- Fix:
  - Moving any task slider away from `0` now activates it as a candidate filter.
  - Checking `lock` now mainly means "enforce this task even at exactly `0`", i.e. no predicted regression versus proportional.
  - Added an `Automatically inspect best feasible recommendation` checkbox, enabled by default. In this mode, the inspected candidate follows the current best feasible candidate as sliders change.
  - The manual dropdown remains available when automatic recommendation following is disabled.
  - Added a visible `Currently inspecting` status block so it is obvious when the recommendation changed.
- Validation:
  - `uv run pytest tests/test_factor_dsp_constraint_dashboard.py tests/test_factor_dsp_candidate_library.py -q`
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard_helpers.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_candidate_library.py experiments/domain_phase_mix/exploratory/two_phase_many/build_factor_dsp_task_prediction_cache.py`
  - `uv run --with marimo marimo check experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py`
  - `uv run --with marimo --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with pyarrow marimo export html experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_constraint_dashboard.py -o experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/factor_dsp_constraint_dashboard_20260526/factor_dsp_constraint_dashboard.html --no-include-code -f`
