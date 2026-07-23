# Prompt To ChatGPT Pro

## Current Task Statement

You are leading a sustained scientific model-discovery effort on the attached self-contained packet. Find a simple, principled, mechanistically motivated parametric surrogate for predicting smooth evaluation BPB from one-phase and two-phase language-model data-mixture policies. The surrogate must explain useful training dynamics rather than merely interpolate these datasets, must support optimization into a plausible mixture, and must have a clean fitted single-phase restriction. Do not assume that a superior model exists. A rigorous negative result, with every investigated family blocked for an explicit reason, is preferable to selecting an unsupported winner.

The practical unresolved question is whether a fixed checkpoint budget can support a two-phase surrogate whose selected policy reliably improves over the corresponding single-phase solution. Dense two-domain StarCoder surfaces and several controlled 3e18 interventions show that phase order can matter, but our high-dimensional 39-bucket surrogates still fail to identify a globally reliable two-phase optimum. They often rank ordinary mixtures well yet become severely optimistic out of support, compress the response range among frontier candidates, or select mixtures that do not validate better than the one-phase frontier.

Work directly in the extracted packet. Start by reading `README.md`, `docs/SCIENTIFIC_BRIEF.md`, `docs/DATA_DICTIONARY.md`, `docs/MODELS.md`, `docs/CHATGPT_PRO_PROTOCOL.md`, and `evidence/mechanistic_surrogate_discovery/final_synthesis/final_report.md`. Run `uv run --no-project --script standalone_code/inspect_packet.py` and `uv run --no-project --script standalone_code/test_packet.py` before changing equations. Reproduce at least one shipped baseline with `standalone_code/reproduce_fit.py`.

## Mathematical Setup

The corpus is partitioned into (m) buckets with token counts (n_i). A (K)-phase policy is (w=(w^{(0)},\ldots,w^{(K-1)})), where each (w^{(k)}\in\Delta^{m-1}), and phase fractions are (\alpha_k). At token budget (D), bucket (i) receives simulated exposure

$$
e_i^{(k)}=\frac{\alpha_kD}{n_i}w_i^{(k)}=c_i^{(k)}w_i^{(k)}.
$$

The target (L(w)) is a smooth evaluation loss measured in bits per byte; lower is better. The one-phase policy class is the tied restriction (w^{(0)}=w^{(1)}). The two-phase class contains it. Aggregate exposure alone does not identify ordering: a useful two-phase model must separate aggregate learning from phase contrast through an explicit state, transition, or response mechanism.

The model may receive mixture weights, phase fractions, bucket sizes, realized or simulated exposures, and optionally the packet's predeclared bucket families. Parameters may be target-specific, but the functional form should transfer across swarms. Deployment regularization such as KL penalties may constrain a proposed policy, but it is not evidence that the raw surrogate surface is correct.

## Evidence In The Packet

The core fit and falsification panels are:

- `300m_two_phase_fit` and `300m_one_phase_fit`: matched 280-row, 39-bucket Dolma 3 plus Dolmino swarms with Uncheatable and OLMoBaseEval Table-9 macro BPB.
- `delphi_3e18_two_phase_fit` and `delphi_3e18_one_phase_fit`: scale-matched 280-row, 39-bucket swarms with both targets.
- `delphi_3e18_heldouts`: 1,533 complete checkpoint observations over 1,487 unique policy coordinates as of packet creation. Of these, 1,521 observations are coordinate-disjoint from the two-phase fit panel. This archive is intervention-designed development evidence, not an IID test set.
- `production_two_phase_fit`: an 840-row, 168-bucket Grug-MoE production swarm with Uncheatable BPB.
- `starcoder_cosine_50_50` and `starcoder_wsd80`: dense two-domain surfaces with different phase and learning-rate schedules.
- `legacy_60m_uncheatable`: an additional 39-bucket cross-scale panel.

The two primary targets are `uncheatable_bpb` and `table9_macro_bpb`. Every 3e18 heldout row has both targets. Treat target-matched evaluation as primary and cross-target evaluation as secondary transfer evidence. Always stratify heldout results by policy class, proposal target, training series, candidate kind, and support distance. Do not report only a pooled number.

The strongest existing model families and exact historical sources are included: canonical and effective-exposure DSP, separate heads, Compact retained state, bucket-resolved family GRP, hierarchical phase replay, OLMix log-linear, linear baselines, and later counterfactual/deficit/output-link routes. `standalone_code/reference_models.py` provides a clean common API; `exact_source/` preserves historical implementations and search details.

The current Compact retained-state raw-optimum result is an especially important falsifier. Its 280-row Uncheatable optimum predicted 0.953118 BPB and observed 0.990255; its 280-row Table-9 optimum predicted 0.951460 and observed 1.071885. More fit rows made the proposed policies stable without repairing deployment calibration. The full result is under `evidence/compact_raw_optimum_validation/`. Treat policy convergence as necessary but not sufficient.

The previous two large local drives investigated 99 registered mechanistic routes and found no model that passed the frozen gate. Their registry and blocking evidence are included. Do not repeat a rejected route under a new name. Reopen one only when you introduce a materially new latent state, invariant, transition law, response mechanism, or identification argument.

## Data-Use Boundary

All outcomes in `delphi_3e18_heldouts` have been observed and are development/falsification evidence. Never fit model parameters, output calibration, feature definitions, or hyperparameters directly to those target values and then describe performance on the same rows as validation. Work in batches: derive and freeze a mechanistically diverse candidate batch using algebraic audits, StarCoder surfaces, grouped CV, fit-complement checks, and cross-swarm evidence; record the freeze; evaluate the frozen batch on heldouts once; append the exposure to your own data-use ledger; and only then synthesize.

The packet deliberately excludes results from a currently running Compact sub-280 raw-optimum panel. Treat it as sealed future evidence. Do not search for, infer, request, or use those outcomes. No result produced from this packet is finally confirmed. Any recommended model must later be tested on a new untouched panel whose generation rule and acceptance criteria are frozen before outcomes are inspected.

Do not submit training jobs. This task is local modeling, fitting, optimization, and falsification only. Public search may be used for ordinary scientific background and mechanistic inspiration, but not to locate leaked outcomes or solutions to this exact packet.

## Non-Negotiable Model Qualities

A serious candidate must have a concise governing equation and a mechanistic interpretation for every term; specify its latent state, state transition, response link, and limiting cases; reduce naturally to a phase-tied restriction; distinguish the algebraically tied restriction of a two-phase fit from the same restricted form independently fitted on one-phase data; state units or dimensionless quantities and identify parameter symmetries; represent diminishing benefit, shortage, repetition harm, retention, forgetting, recency, competition, or interaction only through explicitly justified mechanisms; report nominal parameter count, effective degrees of freedom, identifiability, and bootstrap stability; explain why its mechanism should transfer across scales, schedules, and swarms; and produce a plausible raw optimum without relying entirely on a strong trust region.

Do not use ensembles, nearest-neighbor corrections, lookup tables, candidate-series indicators, arbitrary residual features, unconstrained output calibration, or post-hoc transformations of heldout predictions as the headline model. Model disagreement and support distance may be uncertainty or abstention diagnostics, but they are not substitutes for a mechanistic surrogate. Keep deployment regularization separate from evidence about model form.

## Search Strategy

Use parallel agents or independent workstreams aggressively if your interface supports them, but manage them dynamically rather than assigning a fixed number to each favored idea. Begin with a genuinely diverse portfolio of mechanisms. Preserve independence during first derivations so that the current GRP, DSP, Compact, or hierarchical replay interpretations do not dictate every proposal. Explore substantially different latent-state dynamics, coverage and overload laws, learning-and-forgetting systems, survival or hazard models, bottleneck production functions, constrained competition, phase-specific response laws, identifiable aggregate-plus-contrast decompositions, and other mechanisms supported by the inputs. These are starting points, not a request to repeat prior implementations.

Maintain an explicit approach registry. For each family record its relationship to prior work, materially new mechanism, governing equations, latent state and transition, additional degrees of freedom, expected signature on both StarCoder surfaces, expected resolution of catastrophic optimism, expected resolution of adversarial response compression, expected scale-transfer behavior, cheapest falsification test, status, and exact evidence for that status. Group routes by their actual mechanism, not superficial algebra.

Do not allow an approach to dominate merely because it improves one scalar metric or gives an elegant reparameterization. Mark a route blocked when a new coefficient collapses to zero or a boundary, parameters are unstable or unidentified across folds, the model restates the target through calibration, the missing mechanism is effectively the residual, fit-panel RMSE improves while optimum-region optimism or adversarial calibration worsens, complexity grows without improvement on at least two independent panels, or the route works only after tuning against exposed heldout outcomes. Reopen a blocked route only for a materially new state variable, invariant, transition law, response mechanism, or identification strategy.

Keep several incompatible mechanisms alive through multiple rounds. Cross-pollinate only after their independent derivations expose real strengths and failure modes. Require concrete equations, limiting-case calculations, fits, residual tables, counterexamples, or optimized policies. Reject vague progress reports and claims that a missing compatibility or identifiability argument is routine.

Use adversarial review throughout. For each promoted model, assign independent mechanistic and statistical audits without telling reviewers which candidate is favored. Require concrete counterexamples, leakage findings, unidentified parameters, invalid units, bad limiting cases, pathological optima, missing ablations, or proposal-series failures rather than general impressions.

Repeatedly synthesize, challenge, redirect, and launch a genuinely new round when prior routes fail. Do not stop after the first unsuccessful wave. Spend at least eight hours on this before considering a final answer. This is a direction for sustained investigation, not permission to pad the report or retune exposed families indefinitely.

## Falsification Ladder

Evaluate every serious candidate in this order.

1. Algebraic audit: check units, limiting cases, phase-tied reduction, monotonicity where justified, parameter symmetries, boundedness, and whether the raw model can create corner or unbounded optima.
2. Two-domain shape audit: fit both StarCoder surfaces; report surface RMSE, leave-region-out prediction, Nike-swoosh geometry, predicted optimum location, and whether phase-order effects change sensibly between cosine 50/50 and WSD 80/20.
3. Multi-swarm grouped CV: report grouped OOF RMSE, Spearman, Regret@1/@3/@5, low-tail RMSE, lower-tail optimism, and parameter stability for every applicable swarm and target. Respect paired interventions and repeated coordinates in fold construction.
4. Single-phase restriction audit: evaluate both the two-phase fit restricted algebraically to tied inputs and the identical restricted form fitted directly to one-phase rows. Explain any gap.
5. Cross-scale matched-policy audit: compare matched 300M and Delphi 3e18 policies separately for one- and two-phase classes. Report BPB and rank correlations, calibration, parameter transfer, and whether the model explains weaker two-phase transfer.
6. Frozen 3e18 development heldouts: after freezing the batch, report policy-matched RMSE, bias, observed-on-predicted calibration slope, calibration bins, optimism greater than 0.05 BPB, worst optimism, selected optimism, Regret@1/@3/@5, and support-stratified behavior. Inspect the ten worst mixtures directly and identify which exposure pattern is mispriced.
7. Adversarial strata: report target-matched and cross-target results separately by proposal target, policy class, selection stratum, candidate series, and proposer where available. Test response compression and within-frontier rank, not just catastrophic errors.
8. Optimization audit: optimize the unregularized surrogate before any deployment penalty. Report predicted BPB, max bucket weight, max simulated epochs, aggregate exposure, phase divergence, empirical and convex-support distance, and bootstrap stability. Then, separately, show a minimal preregistered deployment regularizer if needed.
9. Nested ablation: remove every retained mechanism and repeat the relevant audits. Retain a term only when its contribution survives on at least two independent panels or schedules.

Treat residual-versus-observed slope as descriptive only because the observed value appears on both axes. Use observed-on-predicted calibration, heldout bias, binned calibration, explicit optimism counts, tail errors, and post-selection regret for decisions.

## Frozen Acceptance Standard

Reproduce the shipped Pareto baseline and metric definitions before candidate generation. Do not select one incumbent solely by RMSE. Freeze your numerical gate and bootstrap procedure before inspecting a new candidate batch; do not move it afterward. At minimum, a headline candidate must materially improve at least one primary heldout failure diagnostic beyond paired-bootstrap uncertainty; improve both targets, or improve at least two primary diagnostics on one target without a material regression on the other; avoid worsening any core grouped-OOF RMSE by more than 5 percent; avoid worsening policy-matched heldout Regret@1 by more than 0.002 BPB; reduce or preserve optimism counts above 0.05 on both targets; move heldout calibration toward slope one without a free calibrator; improve or preserve calibration within major proposal strata; survive full two-phase, algebraically tied, and independently fitted one-phase comparisons; transfer across at least two independent swarms or both StarCoder schedules; retain stable parameter signs and comparable dimensionless values; and produce a plausible bootstrap-stable raw optimum.

If no candidate clears the gate, say so. The correct conclusion may be that the available designs do not identify a trustworthy global two-phase optimum, and that a specific new intervention panel is necessary. Make that negative result precise enough to guide the next data collection.

## Required Deliverables

Return a complete self-contained zip named `chatgpt_pro_two_phase_surrogate_solution.zip`. It must run after extraction without the Marin repository, W&B, GCS, Iris, Fieldbook, private credentials, or undocumented local files. Include:

- `README.md` with a one-command or short-command reproduction path.
- `REPORT.md` with the verdict, exact evidence, limitations, and strongest argument that any apparent winner is spurious.
- `APPROACH_REGISTRY.csv` covering every route considered and linking each status to evidence.
- `DATA_USE_LEDGER.csv` recording every exposure to development heldouts and what was frozen beforehand.
- `ACCEPTANCE_GATE.json` frozen before candidate evaluation.
- `src/` with the complete model, fitting, grouped-CV, metric, optimization, and plotting implementation.
- `tests/` with algebraic, limiting-case, tied-restriction, and deterministic smoke tests.
- `results/` with comparable metrics across all applicable panels, target-matched and cross-target heldout tables, proposal-stratified diagnostics, all heldout predictions and residuals, parameter/identifiability tables, nested ablations, raw and regularized optima, and bootstrap summaries.
- `figures/` with calibration, residual, StarCoder surface, cross-scale transfer, worst-mixture exposure, and optimum visualizations.
- `PROPOSED_CONFIRMATION_PANEL.md` and a machine-readable manifest describing the smallest untouched validation panel that could confirm or falsify the recommendation, without launching it.
- Exact commands and dependency declarations needed to reproduce every headline table.

If a model survives, state its equation, interpretation, latent state, transition law, response link, single-phase restriction, independent one-phase fitting procedure, optimization procedure, deployment regularization policy, parameter count, remaining identifiability risks, and exact future confirmation criterion. Describe it as provisional until new untouched training validates it.

If no model survives, still return the zip with the full registry, reproducible negative evidence, strongest remaining hypotheses, and the exact minimum experiment needed to distinguish model misspecification from non-identification. Do not return only a narrative answer, a notebook with hidden state, a reduction to an unspecified future model, or a prediction file without executable code.

Return only after the best surviving conclusion has passed adversarial audit and the zip has been rebuilt and tested from a clean extraction.
