# StarCoder WSD80 LR-onset dense-surface design review

## Decision to review

Design a central1 experiment that tests whether beginning cosine learning-rate decay earlier, and therefore using a longer decay phase, changes the advantage of an untied two-phase StarCoder/Nemotron mixture over the best tied mixture.

Do not review this as another gradient-probe experiment. The outcome is endpoint Paloma Programming Languages BPB over a dense two-dimensional policy surface. The phase boundary remains fixed at 0.80T. Lower BPB is better.

## Existing evidence and code

- `experiments/domain_phase_mix/launch_starcoder_wsd80_lr_onset_intervention.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_lr_onset_gradient_probe_results_20260823/report.md`
- `experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py`
- `experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_dense_horizon_replay_confirmation_scaling_20260811/report.md`
- Fieldbook LR-onset experiment: `exp_01m0qrwq4dfc3bs0wd3yw97me4`
- Fieldbook dense horizon-by-replay experiment: `exp_01kzgj5v870a2s1tn0db1ayx9y`
- New design experiment: `exp_01m0xqtjba8rr95gxdme079rwz`

The LR-onset intervention used a 210M-parameter model, 3,820 updates, 1.001B materialized tokens, a tied 35% StarCoder policy, full physical StarCoder support, and eight paired seeds. Its four audited arms were cosine decay beginning at 0.60T, 0.80T, or 0.90T, plus no decay. It established that raw source-gradient geometry follows LR decay, but it deliberately read no endpoint metric and cannot answer whether the mixture optimum changes.

The dense support panel used the same 210M model and a common 125-coordinate two-phase grid at four token horizons and seven support regimes. Fresh five-seed confirmation found no positive mean selected-policy gain in any full-pool block. At 7.41B tokens, the 1x, 2x, and 4x finite-replay blocks had confirmed gains of +0.007576, +0.010487, and +0.013843 BPB. Raw smoothed-surface optima were unreliable; raw-grid discovery followed by fresh paired confirmation is the accepted procedure.

## Candidate design

The default proposal is deliberately simple and end-to-end rather than a new shared-prefix execution path:

1. Freeze the four already-audited schedule arms: onset 0.60T, 0.80T, 0.90T, and no decay.
2. Hold model, total updates, peak LR, warmup, phase boundary, datasets, reference holdout, optimizer, initialization seed, data seed, and the 125 policy coordinates fixed across arms.
3. Use one common discovery seed per schedule-coordinate pair. Preserve coverage rather than spend discovery compute on repeats.
4. Select the lowest observed tied and untied coordinate independently within each schedule arm.
5. Confirm those frozen selected policies with five fresh paired seeds per arm. Discovery outcomes are not pooled into confirmation.
6. Treat coordinate distance or `|w1-w0|` as descriptive only. The primary per-arm quantity is confirmed BPB gain, best tied minus best untied; positive favors two-phase.
7. Test the schedule interaction with seed-blocked contrasts of confirmed gain across arms. The main directional hypothesis is that earlier onset has larger gain. Report every arm and do not replace a non-monotone pattern with a fitted trend.

The practical schedule intervention holds peak LR and total steps fixed, so earlier onset also lowers cumulative LR. This is intentional for the first endpoint experiment: it asks whether changing the actual training schedule changes two-phase gain. It does not by itself identify onset timing separately from cumulative optimizer distance. An area-matched schedule would change the pre-onset trajectory or terminal LR and is not proposed as part of the primary dense panel.

## Competing choices

### A. Exact causal-panel cell

- 1.001B tokens, full StarCoder pool, same holdout and seeds as the LR-onset probe.
- Four by 125 = 500 logical discovery rows; the four tied 35% rows for one seed may be exact aliases if configuration fingerprints prove identity.
- Cheapest and cleanest bridge from the gradient result.
- Risk: existing fresh confirmation found no positive full-pool gain and the likely schedule effect may be below run noise.

### B. High-SNR endpoint cell

- 7.408B tokens with 1x finite StarCoder replay, where historical 0.80T decay has a confirmed +0.007576 BPB selected-policy gain.
- Four by 125 = 500 discovery rows, substantially more compute.
- Most likely to produce an interpretable schedule-by-policy interaction whichever direction it goes.
- It no longer exactly matches the LR-onset causal panel, and the existing probe checkpoints are not reusable.

### C. Staged combination

- Run A as the dense mechanistic bridge and B either concurrently or only if A is underpowered.
- Strongest scope, but a sequential B decision based on A can turn the second block into an outcome-contingent follow-up unless its trigger is frozen now.

## Questions for adversarial review

1. Which of A, B, or C is the smallest design that can give a determinate answer to the stated endpoint question? Do not recommend the cheap panel merely because it is cheap.
2. Is confirmed tied-minus-untied BPB gain the right operational definition of `two-phaseness`? Identify a better primary estimand if needed, but do not use coordinate distance alone.
3. Is one discovery seed over 125 common coordinates plus five fresh paired confirmation seeds statistically defensible under the observed ~0.007 BPB seed noise and selection bias?
4. Should selected tied and untied policies be chosen independently per schedule, or should the same coordinate pair be compared across schedules to isolate response-surface deformation? Specify primary and secondary contrasts.
5. Does holding peak LR fixed while cumulative LR changes invalidate the requested claim, or is careful claim wording sufficient? Is there a cheap control that actually identifies onset duration without introducing a worse confound?
6. Can any completed LR-onset or dense-support rows be reused exactly? Require bitwise-equivalent training data, holdout, seed, optimizer, and evaluation contracts, not merely similar hyperparameters.
7. What launcher and manifest assertions are needed to prevent schedule, coordinate, seed, support, and central1 locality drift?
8. Identify any scientific or implementation blocker that should be resolved before code freeze. The full job must not be submitted as part of this review.

Return a concrete recommended design, explicit primary/secondary estimands, confirmation plan, reusable-row verdict, and implementation blockers. Be adversarial; an inconclusive design is worse than a larger one.
