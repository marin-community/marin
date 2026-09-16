# Delphi offline selection benchmark, 2026-09-06

Use the canonical 280-run Delphi single-phase panel for every final fit. Read
rounds 1–6 through their synthesis and reports, the September 6 pilot handoff,
and the Observatory protocol before choosing this matrix. These and all bank
results are development evidence. No training, evaluation, or infrastructure
jobs may be submitted. Do not inspect the WSPU cross-scale ladder.

1. Freeze the canonical panel, its 58 atomic outcomes, inventory, evaluator
   weights, and original five blocked outer / three inner partitions. Restrict
   the development bank to coordinate IDs and source memberships present in
   the round-3 canonical prediction artifact; use its frozen aggregate labels.
   Store hashes, features, labels, and split assignments separately. Repeat
   seeds and pool variants must not cross any fitted split.
2. Refit WSPU, canonical per-bucket DSP, and repository taskwise OLMix with the
   original starts, grids, folds, and deterministic component seeds. Compare
   newly computed predictions against original shards and frozen bank scores.
   Keep algorithmic convergence claims separate from adapter success flags.
3. Run matched alternatives with the same outer/inner partitions: WSPU fitted
   directly to the aggregate; signed ridge on shares, square-root shares, and
   log(1 + materialized epochs); shared versus per-task regularization and
   rank-three response pooling; a Matérn kernel in share versus square-root
   share geometry; training-only residual variance weighting; and hyperparameter
   selection by inner regret versus RMSE. Report effective degrees of freedom.
   The regularization grids and kernel length scales are frozen in code before
   the bank scores are inspected. Do not expand them in response to bank scores.
4. Compare point selection with a fixed one-refit-standard-deviation penalty and a
   label-blind diverse shortlist. Report regret@1, best-of-5/10 regret, selected
   measured rank, observed-minus-predicted optimism, RMSE, and Spearman. Score
   every outer fold, the pooled out-of-fold panel, the frozen external bank,
   model-proposed versus intervention strata, and individual source families.
   Add leave-one-source-out method selection using development sources only,
   purging all overlapping source memberships from its training candidates.
5. Diagnose extrapolation with nearest-panel distance and constrained convex
   hull projection; distinguish joint mixture support from per-bucket ranges.
   Audit the pilot's local versus global result and repeat-noise covariance.
   Use paired source resampling for descriptive uncertainty; do not treat
   correlated folds or previously used development sources as prospective data.
6. Publish reproducible local artifacts, a short diagnosis, and either one
   fully specified frozen candidate with a prospective paired-seed plan or an
   explicit null. Any candidate is unconfirmed. Preserve the existing dirty
   checkout and record completed artifacts in the benchmark Fieldbook ledger.

Implementation will reuse the existing panel loader, model registry, evaluator
weights, fold generator, and atomic shard writer. New files own this benchmark;
the incumbent registry and launchers remain outside the implementation scope.

Completed: all three baselines reproduced, 12 bounded alternatives and the
direct-macro WSPU ablation evaluated, and fixed dispersion/diversity policies
scored. A linear kernel control was included before bank scoring to isolate
the Matérn nonlinearity from feature scaling. The screen did not support
promotion of a replacement. Final fits use only the canonical 280; source and
pilot analyses remain development evidence. Results and exact commands are in
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_offline_selection_20260906/`.
