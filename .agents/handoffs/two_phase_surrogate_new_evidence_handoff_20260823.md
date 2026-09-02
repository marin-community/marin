# Handoff: next two-phase surrogate round after the gradient and state audits

## Objective

Develop a surrogate that can select genuinely strong two-phase data mixtures, not merely fit endpoint levels.
The ultimate target is reliable optimum prediction for the 39-bucket swarms at 300M and 3e18, for
Uncheatable and Table-9. The canonical deployment budget remains 280 training rows. Historical data
may be used to discover model form, but any final claim must state exactly which rows and features are
available at deployment time and must be evaluated without test-set selection.

Do not assume that RPL, HPR, or the current general surrogate is the final architecture. Preserve useful
parts when evidence supports them, but be willing to replace the model family. We want a small number of
falsifiable structural hypotheses, not another broad coefficient sweep.

The modeling ladder must remain explicit:

| Repetition | Buckets | Objective | Role |
|---|---:|---|---|
| none | 2 | one atomic metric | simplest representability and optimum-placement test |
| varied | 2 | one atomic metric | isolate repetition, horizon, and schedule effects |
| varied | 2 | multiple atomic metrics | test target sharing and target-specific policy value |
| low or varied | 39 | one atomic metric | test dimensional scaling without macro averaging |
| low or varied | 39 | multiple atomic metrics | primary 300M development problem |
| low-TPP validation | 39 | multiple atomic metrics | final 3e18 frontier test, not the first model-development cell |

Success in a two-bucket cell is necessary but not sufficient. The model must state how its parameter
count and identification requirements scale from one contrast dimension to 38.

## Scope decision for this round

Work only on the **open-loop endpoint surrogate**. The prefix-search panel has not produced the crossed
prefix/continuation outcomes needed to identify a transferable state-action critic. Therefore every
active candidate in this round must use inputs available before training:

- phase mixtures `w0` and `w1`;
- phase lengths and optimizer/LR schedule;
- model size, token horizon, TPP, and other fixed training configuration;
- static bucket metadata, pool size, family, and content geometry; and
- deterministic exposure, repetition, availability, or latent-trajectory features computed from the
  policy and configuration.

Measured phase-boundary losses, gradients, optimizer updates, hidden states, and CAP summaries are not
admissible active inputs. Existing measurements may constrain the functional form or falsify a mechanism,
but they may not be smuggled into an apples-to-apples 280-row predictor.

Do not fit a measured-prefix continuation critic, a CAP state model, or a prefix-to-state transition
model in this round. At most, preserve a short feature schema and frozen evaluation protocol for later.
Reopen those routes only after crossed-prefix results exist.

## Start here

Read these in order before proposing or fitting a candidate:

1. `.agents/projects/two_phase_surrogate_active_registry.csv`
   - This is the authoritative candidate and correction ledger. Register every new route before fitting.
   - In particular, read ATOM-019 through ATOM-031. Several intuitive state-feature routes have already
     failed adversarial or within-cell checks.
2. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_mechanism_contextual_plots_v19_20260823/index.html`
   - Self-contained presentation of the gradient, target-utility, endpoint, and TPP results.
3. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_fixed_n_tpp_gradient_probe_results_20260823/report.md`
   - Exact fixed-N TPP result, measurement caveats, and the reason TPP is not yet an identified change-point controller.
4. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/trajectory_conditioned_surrogate_review_20260820/report.md`
   - Context only for this round: it explains why state-conditioned routes are deferred until crossed-prefix data exists.
5. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/wsd80_selected_policy_noninferiority_gate_20260815/report.md`
   - Corrected WSD80 gate. Coordinate distance is descriptive, not an acceptance criterion.
6. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_dense_support_calibration_results_20260813/report.md`
   - Heteroskedasticity, selection bias, and failure of global quartic surfaces.
7. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_dense_horizon_replay_confirmation_scaling_20260811/report.md`
   - Dense horizon-by-replay surfaces and fresh confirmations.
8. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_full_pool_atomic_surface_explorer_20260811/report.md`
   - No-repetition atomic surfaces. Repetition is not necessary for an interior or untied optimum.
9. `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_3e18_fixed_aggregate_phase_snr_20260724/report.md`
   - Why the 3e18 phase-order channel is a low-SNR development target even though large harmful asymmetries are visible.
10. `/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper/theory.md`
    - Use its aggregate/contrast and state/action notation. Preserve the distinction between open-loop
      policy optimization and a measured-state feedback policy.

Also inspect the current implementation and drivers:

- `experiments/domain_phase_mix/exploratory/two_phase_many/general_mixture_surrogate_20260809.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/range_selected_surrogate_20260810.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/fit_frontier_wsd80_20260812.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/fit_swarm39_split_damage_20260817.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/test_general_mixture_surrogate.py`

Treat `.agents/handoffs/two_phase_surrogate_promotion_case_20260809.md` as withdrawn and
`.agents/handoffs/general_surrogate_followup_20260809.md` as void. Do not revive the old 43/44 headline:
it mixed estimators and is not a valid single acceptance score.

## Current frontier and remaining failure

The current general surrogate is best described as a label-blind hierarchical retained-power-law
model with WSD80 mechanism terms:

- retained/offset power-law responses at near and final horizons;
- HPR-style family pooling with shrunk per-bucket amplitudes;
- an exponential boundary kernel;
- bounded repetition damage;
- a rank-truncated solve; and
- range-penalized selection.

On the 39-bucket 300M Table-9 split, the current `split` variant fits 280 rows and evaluates on 1,957:

| model | RMSE | R2 | Spearman | best actual among predicted top 20 |
|---|---:|---:|---:|---:|
| blended | 0.07412 | 0.276 | 0.865 | 1.05950 |
| physical | 0.07071 | 0.341 | 0.840 | 1.05950 |
| split | **0.06049** | **0.518** | **0.885** | 1.05950 |

The panel best is 1.05331 and the median is 1.09684. The split model ranks grossly bad policies well,
but its RMSE is about 17.7 times the run-noise scale and about 23 times the roughly 0.0026 BPB effect
we want to resolve. Its optimized phase-0 mixture is TV 0.649 from the best one-phase mixture and TV
0.911 from proportional, while good observed endpoints cluster near the one-phase optimum. It is not a
usable prefix optimizer.

On WSD80, GEN-039 passes 33/33 corrected diagnostics: 11 RMSE, 11 observed-row Regret@1, and 11 gain-error
checks. This is not a promotion. Its continuous selected coordinate was absent from the historical
panel, so the only direct gate, fresh same-seed selected-policy non-inferiority, has not run.

The present gap is therefore optimum placement and support-aware selection, not basic expressiveness
or gross ranking alone.

## Evidence that must change the next model

### 1. Global source-source conflict is not the main control variable

Across four fixed-N WSD80 rungs at total TPP 4.77, 9.14, 18.67, and 35.27, the finite-batch raw
StarCoder-Nemotron gradient cosine falls late. However:

- source gradient norms fall 39% to 53%, so finite-batch cosine attenuation remains viable;
- the 32-block precision control confirms a late raw-cosine change at the highest rung but does not
  identify its cause;
- projected optimizer-update cosine does not decline and instead becomes slightly more aligned; and
- the experiment does not identify a TPP-controlled onset because the common time grid has only one
  point after 0.80T and LR decay, normalized progress, and decay fraction are collinear.

Do not use TPP as a standalone phase-change switch. Do not make global StarCoder-Nemotron cosine the
main phase-control feature.

### 2. Target-conditioned utility remains state dependent

Both source updates become less aligned with evaluation targets late even while they remain mutually
aligned. The action-relevant contrast is not the pairwise source cosine. For target `y`, use quantities
such as:

```text
X_y(s) = -g_y(s)^T [u_StarCoder(s) - u_Nemotron(s)]
       = U_y(StarCoder | s) - U_y(Nemotron | s)

A_y(s) = X_y(s) / (||g_y(s)|| ||u_StarCoder(s) - u_Nemotron(s)||)
```

where `u_i` is the optimizer-aware local update induced by source `i`. The frozen probe rows already
contain the ingredients for `X_y` and `A_y`. Use them as mechanistic evidence about which source should
be preferred in which regime, not as inputs to the active open-loop surrogate. They are unavailable for
arbitrary candidate policies before training.

The 512-step H4 result is important but limited: post-hoc local optimizer-aware utility predicts short
rollout outcomes well (R2 0.825, Spearman 0.93), yet its selection is degenerate and its RMSE is 0.0381,
about 29 times the H5 endpoint effect. Local predictive association is not yet a policy optimizer.

### 3. The stable-LR phase has an approximate local-utility plateau, not proven constant value

From 0.55T to 0.80T across 32 target/source/rung cells, optimizer utility cosine changes by a median
-1.6%, unnormalized utility by -2.2%, and source update norm by -0.6%. This supports a local plateau
during the latter constant-LR regime. It does not show that data has no diminishing returns, that utility
is state independent, or that the same ordering holds earlier, after decay, under repetition, or in a
39-bucket model.

A promising model may use a piecewise-stationary or slowly varying control regime, but it must permit
late target-conditioned revaluation.

### 4. Repetition matters, but cannot be the only source of two-phase gain

Dense WSD80 replay/horizon panels show that repetition changes the surface, optimum, and noise. Fresh
five-seed confirmations support positive gains in the largest-token cells. But no-repetition, full-pool
atomic surfaces can still have interior tied or untied optima. A repetition penalty is therefore a
mechanism term, not a complete explanation.

### 5. Endpoint state helps level prediction more readily than continuation ranking

The 3e18 phase-boundary readout correlates strongly with endpoint level across untied policies, but its
median within-aggregate Spearman is negative. A trajectory head improves some endpoint RMSE, and a
shrunk boundary-state column can improve within-cell rank, but the gains reverse under adversarial
splits. Naive global checkpoint summaries likewise improve development level fit while collapsing
adversarial ranking and calibration.

Consequences:

- an additive boundary feature shared by every continuation of one prefix cannot rank those continuations;
- state-action interaction is required for a continuation critic;
- measured state may still rank prefixes or calibrate endpoint level;
- global summaries should not be added to the headline surrogate without incremental adversarial evidence;
- CAP-style hidden-state features remain untested only in a bucket-conditioned form with loss-matched,
  split-shard, and shuffled-bucket controls.

These are reasons to defer measured-state modeling, not invitations to run it now.

### 6. More endpoint rows alone have not solved the 39-bucket problem

Separately fitted 60M, 300M, and 3e18 39-bucket optima disagree by roughly random-simplex distances.
The 3e18 archive has the most rows but the least stable fitted argmin. The problem is not only sample
count. Coverage is concentrated, phase benefit is small near strong aggregates, and aggregate and
phase-order effects are poorly separated.

At 3e18, whole-swarm signal is high because aggregate changes and large harmful asymmetries are easy
to detect, while aggregate-matched phase effects are near the noise scale. This makes 3e18 appropriate
for final validation and falsification, not the primary place to discover a high-dimensional temporal
mechanism. Prioritize the higher-TPP 300M swarm for 39-bucket model development.

### 7. Static content geometry is useful but insufficient

Hellinger action geometry improves some gross ranking, support, and optimism metrics, but it does not
consistently improve RMSE, within-cell phase rank, or Table-9 Regret@20. Retain it as an action or support
embedding and baseline. Do not treat static geometry as the missing temporal state.

## Active and deferred deployment modes

The active problem is:

```text
f_y(w0, w1, training_config, static_data_metadata) -> endpoint objective y
```

This is the apples-to-apples replacement for current Observatory models. A model may internally simulate
or integrate a latent trajectory, but every latent quantity must be deterministically predicted from the
policy and pre-training configuration. It may not depend on a measured checkpoint.

Two modes are explicitly deferred:

1. A measured-prefix continuation critic `Q_y(z_tau, w1)`, which requires crossed prefixes and common
   continuations to identify transferable state-action interactions.
2. A joint optimizer that first learns `z_tau = T(w0, schedule, scale, exposure)` and then composes `T`
   with the critic. This is premature until the measured-state critic succeeds under leave-prefix-out tests.

The handoff records these deferred modes so they are not confused with the active claim. Do not spend
candidate-search or fitting effort on them now.

## Modeling directions to test

These are prompts, not required architectures. Propose alternatives if the evidence supports them.

### A. Target-conditioned inventory-control model

Treat the stable-LR interval as an approximate local-utility plateau, then model late revaluation and
finite bucket inventory through policy-derived features. A separable or convex baseline could integrate
target-specific source value over exposure, with bucket availability, repetition, and reserve constraints.
Any temporal value curve must be shared or inferred from the training configuration, not measured from
the candidate run. The model must allow an untied optimum without requiring pairwise source conflict.

Falsifier: it fits levels but cannot place WSD80 optima or rank aggregate-matched 39-bucket phase contrasts.

### B. Low-rank aggregate/contrast factorization

Parameterize the policy by aggregate mixture and phase contrast, then use low-rank bucket and target
factors for the contrast response. Condition those factors on policy-derived exposure, repetition,
availability, and schedule features. The factor rank and shrinkage must be selected inside training folds
and must scale plausibly from one contrast dimension to 38.

Falsifier: it improves global RMSE or Spearman but not held-out within-aggregate phase rank, Regret@1,
or selected-policy performance.

### C. Open-loop mechanistic trajectory integral

Predict endpoint change by integrating a policy-derived utility proxy along a coarse latent trajectory
rather than using an unconstrained direct surface. Permit piecewise regimes around optimizer schedule
transitions and exposure-dependent repetition damage. The trajectory state must be computed from
`(w0,w1)`, phase lengths, static bucket metadata, and training configuration. Share latent dynamics across
atomic targets only as an ablation and retain target-specific heads.

Falsifier: the additional trajectory structure does not improve held-out endpoint regret or selected-policy
performance over an equal-capacity static-coordinate control.

### D. Atomic-objective factorization

Model individual evaluation components first, then aggregate. Share low-rank state and action factors
only where cross-target evidence supports it. Prior attempts to force all targets through one parameter
set did not work; target sharing must be an ablation, not a premise.

Falsifier: component gains vanish or reverse on the macro, or target sharing worsens worst-target regret.

### E. Support-constrained uncertainty-aware optimization

Separate response modeling from policy selection. Use heteroskedastic uncertainty, support distance,
range penalties, or conservative lower-confidence objectives so a good ranker does not extrapolate into
unsupported corners. Quantile, pairwise-rank, or regret-weighted losses are admissible, but evaluate the
selected policy, not only the training loss.

Falsifier: optimism or support distance improves while actual Regret@1 and paired validation do not.

## Evaluation protocol

Freeze the candidate, hyperparameter search, data splits, and gates before inspecting the relevant heldout
outcomes. Use nested selection where a model family is chosen. Re-select all hyperparameters inside each
bootstrap; fixed-parameter bootstraps have been severely under-covered in this project.

### WSD80

- Report interior/optimum-region RMSE and calibration separately from noisy boundary behavior.
- Report observed-row Regret@1/@3 and continuous selected-policy coordinates, but do not conflate them.
- Coordinate distance is descriptive only.
- Promotion requires a fresh same-seed paired selected-policy comparison. The one-sided 95% upper CI on
  candidate-minus-reference BPB must be at most +0.002.
- Run the same model across multiple atomic metrics to test whether the structure transfers beyond
  Programming Languages BPB.

### 39 buckets

- Use 300M as the primary development panel and 3e18 as validation.
- Report Regret@1/@3/@20, observed-on-predicted calibration, optimism count, support distance, and
  within-aggregate phase ranking. RMSE and Spearman are secondary.
- Report individual components before macro aggregation.
- Compare against HPR, RPL, the current general/split model, static Hellinger geometry, nearest-neighbor,
  and strong tied-policy baselines.
- Report results for the canonical 280-row fit budget. If tied or aggregate panels are added, show the
  exact row accounting and a 280-row apples-to-apples ablation.
- A fitted argmin is not a frontier claim. Materialize and validate selected policies at 3e18.

Do not collapse incomparable OOF diagnostics, full-fit optimization results, and fresh-policy tests into
one gate tally.

## Routes not to repeat without new evidence

- Adding more unconstrained RPL coefficients without a new identified mechanism.
- One-standard-error selection as a general remedy.
- Unpooled per-bucket ordering columns under the current row budget.
- Treating phase machinery as simply unidentifiable in 39 buckets; some structure is identifiable, but
  optimum placement remains weak.
- Using a phase-boundary readout as a free additive continuation score.
- Using global checkpoint summaries without a state-action interaction and adversarial split.
- Treating static Hellinger distance as a complete temporal model.
- Treating raw source-source gradient cosine or TPP alone as the phase switch.
- Fitting measured-prefix, CAP, or prefix-transition candidates before crossed-prefix outcomes exist.
- Selecting on the current 3e18 heldout panel and then reporting that panel as confirmation.
- Reviving a coordinate-distance acceptance gate.

## Requested work product

Return a compact research packet containing:

1. A corrected summary of the current frontier and the strongest negative evidence.
2. Three to five preregistered **open-loop endpoint** hypotheses. For each, state:
   - mechanism;
   - exact pre-training features and how they are computed;
   - parameter scaling from 2 to 39 buckets;
   - predictions across the repetition/bucket/objective ladder;
   - decisive falsifier.
3. A model and ablation matrix, registered in the active registry before fitting.
4. Results under the frozen protocols above, with uncertainty and support diagnostics.
5. One recommended candidate, or an explicit no-promotion verdict.
6. If no candidate transfers to 39 buckets, identify whether the blocker is representability,
   action-space coverage, target aggregation, or selected-policy noise. State whether the negative result
   is strong enough to justify reopening state-conditioned modeling after prefix-search results arrive.

The north star is a new validated 39-bucket two-phase frontier, not a lower average RMSE. Favor a model
whose selected policy survives fresh validation, even if a more flexible model has better in-panel fit.
