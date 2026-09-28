# Review brief: three data-mixing workstreams

Please review three linked proposals. Treat all existing outcomes as exploratory unless a fresh-seed gate is explicitly described. Focus on scientific validity, identifiability, and the smallest defensible model or experiment. Do not edit files.

## 1. StarCoder coupled-onset refinement

Fieldbook: `exp_01m19sep9vs99vb9wxq6e14vvy`

Relevant files:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/results.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/observations.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/selected_policies.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/design_starcoder_wsd80_matched_nd_stage2_20260801.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/design_starcoder_wsd80_matched_nd_confirmation_20260801.py`

All 375 discovery endpoints use one common reference seed. Selected Programming Languages BPB gains of untied over tied are 0.003673 at 0.60T, 0.006674 at 0.80T, and 0.006154 at 0.90T. The paired noise anchor is 0.001182 BPB. Selection is asymmetric: 94 eligible untied rows versus 26 tied rows per arm, and selected minima have small runner-up margins. Selected C4 gains are +0.002596, -0.004406, and -0.120262 BPB, respectively.

Proposed protocol:

1. Fit a fixed Gaussian-process committee separately to tied and eligible-untied Programming BPB within each arm.
2. Add a small deterministic acquisition batch inside the observed convex hull/trust region, with distance exclusion. Acquisition rows use the existing discovery seed and remain discovery-only.
3. After acquisition outcomes complete, freeze one tied and one untied candidate per arm.
4. Compare each pair on the eight fresh seeds reserved before discovery, for 48 fixed-policy runs total. Programming BPB is primary; C4 and a C4-noninferior candidate screen are secondary. No adaptive minimum is itself used as evidence.

Questions:

1. Does this actually distinguish a smaller 0.60T gain from the 0.80T and 0.90T gains, or should the confirmation target direct gain contrasts across arms in another way?
2. What is the smallest defensible acquisition batch per arm and split between tied/untied surfaces?
3. Should the confirmed untied candidate be the raw Programming optimum, a C4-noninferior Programming optimum, or should both be frozen and tested with an explicit multiplicity cost?
4. Identify any leakage, post-selection, support, or common-random-number failure.

## 2. Michael Ryan OLMix swarm benchmark

New Fieldbook: `exp_01m1e2aceqr63b6z9a1vh6epp1`

Live small-data inventory:

- `gs://marin-us-east5/metadata/olmix/dclm_10k/swarm_s42_K363.json`: 118 buckets, 363 designed mixtures, 96 evaluated endpoints currently available.
- `gs://marin-us-east5/metadata/olmix/high_quality_10k/swarm_s42_K363.json`: 118 buckets, 363 designed mixtures, 226 evaluated endpoints currently available.
- FineWeb-CC has 10 evaluated endpoints; FineWeb-Edu has zero, so neither is proposed for model comparison yet.
- Exact endpoint payloads are under `gs://marin-us-east5/metadata/olmix_swarm_bpb/`.

The exact OLMix law fits each evaluation task as

\[
f_j(p)=c_j+\exp(t_j^\top p)
\]

with many random restarts and a summed Huber objective. The proposed benchmark uses identical mixture-blocked outer folds for every model and reports aggregate 42-task BPB RMSE, Spearman correlation, calibration slope, and held-fold selection regret.

Candidate label-blind challengers, in increasing complexity:

1. aggregate linear exposure baseline;
2. shared-shape DSP with one global response-rate parameter and nonnegative per-bucket amplitudes;
3. task-joint shared-shape DSP, sharing the nonlinear exposure shape across tasks while shrinking task-specific amplitudes toward a common bucket direction.

No semantic family partition is allowed. Quality levels may be used only as an explicitly evaluated shrinkage relation, not as a predictive label. Hyperparameters and shape selection must occur inside each outer fold.

Questions:

1. Is this matched enough to claim a model improvement over OLMix on these swarms?
2. Is fitting with 96/363 and 226/363 evaluated rows valid if the missingness is scheduler/evaluation progress rather than outcome-selected? What diagnostics are required?
3. Which challenger is the minimum sufficient model, and what ablation would distinguish a real exposure mechanism from generic regularization?
4. Should the primary target be direct 42-task macro BPB or a task-joint likelihood whose predictions are then macro-averaged?

## 3. State-dependent data-exhaustion model

Fieldbook: `exp_01m11xf71q364xt8dx4jrwhs68`

Relevant files in the sibling worktree `/Users/calvinxu/Projects/Work/Marin/marin-delphi-y0-y1-surrogate-20260827`:

- `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_crossed_prefix_memory_dsp_20260827.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_crossed_prefix_memory_dsp_20260827/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_crossed_prefix_memory_dsp_20260827/mechanism_gate.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_crossed_prefix_panel_v3_20260827/`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_crossed_prefix_results_20260827/crossed_results.csv`

The prior scalar-memory model used

\[
B_i(E_i^{(0)},E_i^{(1)})=e^{-\rho\lambda E_i^{(0)}}g_\rho(E_i^{(1)})
\]

plus incremental cumulative repetition damage. It reduced point RMSE from 0.0403 to 0.0178 BPB, but failed the corrected gate: memory strength was unstable, it did not significantly beat the cumulative endpoint, and its tied-decision regret did not beat always-tied abstention.

Proposed successor: use measured bucket inventory rather than a free scalar memory. For each prefix state and branch action, decompose phase-1 materialized exposure into

\[
F_i=\min(E_i^{(1)}, [1-E_i^{(0)}]_+),\qquad
R_i=[E_i^{(1)}-[1-E_i^{(0)}]_+]_+,
\]

where exposure is measured in materialized epochs of the bucket. Fit shared-shape nonnegative fresh-benefit and repeated-damage amplitudes, with state intercepts. Compare to action-only additive, total cumulative exposure, and the previous scalar-memory model. Keep action-blocked repeated outer CV, leave-one-prefix-out transfer, tied-abstention regret, and data-seed/hardware-code confound scales. This is exploratory because the mechanism was proposed after seeing the coupled-onset and crossed-prefix results.

Questions:

1. Is the fresh/repeated split actually identified by the 50-actions-by-9-prefix crossing, or is it still collinear with cumulative exposure and state intercepts?
2. What exact minimal feature map and parameter sharing make the hypothesis falsifiable rather than merely more flexible?
3. Which evaluation split should be primary for the intended claim: unseen actions at known prefixes, unseen prefixes with a tied boundary readout, or both?
4. What result would count as evidence for state-dependent data exhaustion, and what result should cause us to reject this mechanism?

Please return: blockers first, then a concrete minimal protocol for each workstream, then optional improvements. Distinguish claims you verified from inferences.
