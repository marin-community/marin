# Frozen-procedure validation at 3e18 (submitted 2026-09-07 11:50 UTC; measured 18:34 UTC, gate passed)

Iris parent `/calvinxu/dm-delphi-3e18-lwspu-frozen-v6e8-20260908` (parent us-east5-a, children v6e-8 in
us-east5-b, `MARIN_PREFIX=gs://marin-us-east5`), launcher
`experiments/domain_phase_mix/launch_delphi_frozen_procedure_validation_3e18.py`, candidate table
`delphi_corrected_screen_20260908/materialized_flat15_nocap/runtime_materialization/candidate_weights.csv`
(sha256 e3a90c96...; numerically identical to `data/reference_policies.csv` of the standalone implementation,
mixture-selection commit 14fda28), run ids 7,402,000+ (Uncheatable, data seed 666200) and 7,402,100+ (Table 9,
data seed 662009). Exact command, safety check and launcher tests in `submission/`.

| candidate | target | cap | predicted | cap status | nearest validated mixture (TV) |
|---|---|---|---|---|---|
| lwspu_u_snc_cap06 | Uncheatable | 6 | 0.9810 | interior at 5.25 epochs; caps 6-32 identical | flat-validation optimum 0.9832 (0.038) |
| lwspu_t9_snc_cap06 | Table 9 | 6 | 1.0638 | binds on two buckets | flat-validation cap-6 optimum 1.0680 (0.025) |
| lwspu_t9_snc_cap08 | Table 9 | 8 | 1.0631 | interior at 7.50 epochs; caps 8-32 identical | flat-validation cap-8 optimum 1.0685 (0.027) |

Purpose: these are the proposals of the procedure to be frozen (flat-profile kappa-floor link without the
prediction cap, fitted with the proportional run pinned to the training side of every fold). The mixtures
validated before the protocol fix were proposed by capped, unpinned fits, so the paper's reported numbers must
come from these runs. Same 3e18 recipe, seeds and evaluations as the earlier link validations. Fieldbook experiment `exp_01m1xv9n8j3v1e9175xf5te225`.

Measured (`measured_results.csv`, `review.md`): Uncheatable 0.9814 (predicted 0.9810), Table 9 cap 6 1.0642 (1.0638), cap 8 1.0682 (1.0631). Every row below its incumbent, the additive WSPU and Olmix; gate passed; fairness round submitted.
