# Flat-profile kappa-floor link validation at 3e18 (submitted 2026-09-07 03:41 UTC)

Iris parent `/calvinxu/dm-delphi-3e18-lwspu-kappafloor-flat-v6e8-20260907` (parent us-east5-a, children v6e-8 in
us-east5-b, `MARIN_PREFIX=gs://marin-us-east5`), launcher
`experiments/domain_phase_mix/launch_delphi_kappa_floor_flat_validation_3e18.py`, candidate table
`runtime_materialization/candidate_weights.csv` (sha256 d99a0f9c...), run ids 7,400,000+ (Uncheatable, data seed
666200) and 7,400,100+ (Table 9, data seed 662009).

| candidate | target | cap | predicted | nearest measured mixture (TV) |
|---|---|---|---|---|
| lwspu_u_kff_cap06 | Uncheatable | 6 | 0.9807 | bounded-link optimum 0.9820 (0.14); unbounded kappa-floor optimum 0.9890 (0.215) |
| lwspu_t9_kff_cap06 | Table 9 | 6 | 1.0635 | unbounded kappa-floor cap-6 optimum 1.0613 (0.011) |
| lwspu_t9_kff_cap08 | Table 9 | 8 | 1.0626 | unbounded kappa-floor cap-8 optimum 1.0672 (0.012) |

Purpose: the flat-profile rule (`@kappa_floor_link_flat15`, kappa in [1, 6], 1.5 for monotone profiles) is the
candidate final procedure; both targets must be measured under it. The two Table-9 rows are near-replicates of
the measured unbounded optima and double as replication at the optimum. Same 3e18 recipe, seeds and evaluations
as the earlier link validations; collect with
`collect_delphi_3e18_validation_results_20260906.py --launch kappa_floor_flat`.
