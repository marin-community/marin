# Kappa-floor link validation at 3e18 (KL 0)

Submitted 2026-09-07 as `/calvinxu/dm-delphi-3e18-lwspu-kappafloor-v6e8-20260907` (parent us-east5-a, v6e-8 children in us-east5-b, max_concurrent 3),
under Calvin's instruction to validate the successor's optima first ("materializing the epoch sweep and submitting
candidates").

Surrogate: `weibull_softplus_unscaled@kappa_floor_link` (one log-space head; floor = proportional − κ (proportional −
swarm minimum), κ per task by inner CV up to 100; fits from `delphi_single_head_selection_20260907`). Policies:
Uncheatable cap 6 (data seed 666200) and Table-9 caps 6 and 8 (data seed 662009), trainer seed 0, objective the
predicted aggregate, five starts plus a polish pass, runtime rounding
to the 2048 grid.

| candidate | target | cap | predicted | predicted for the WSPU κ-0 policy | TV to it | effective buckets | max epochs |
|---|---|---:|---:|---:|---:|---:|---:|
| `lwspu_u_kf_cap06` | uncheatable | 6 | 0.9711 | 0.9749 | 0.174 | 8.0 | 5.86 |
| `lwspu_t9_kf_cap06` | table9 | 6 | 1.0631 | 1.0647 | 0.072 | 13.6 | 6.00 |
| `lwspu_t9_kf_cap08` | table9 | 8 | 1.0623 | 1.0646 | 0.088 | 12.5 | 7.50 |

Comparators: `wspu_uncheatable_cap06` 0.9834, `wspu_table9_cap06` 1.0722, cap-8 control 1.0736, the bounded link's
optima 0.9820 / 1.0651 / 1.0636, the 26-run centre 1.0639. Table `runtime_materialization/candidate_weights.csv`
sha256 `87072f50ac02d1ec27f7c08104b2c94bc9a726428f3da0d15babd854aba10b7f`; launcher `experiments/domain_phase_mix/launch_delphi_kappa_floor_validation_3e18.py`
(run ids 7,398,000+ / 7,398,100+); `submission/` holds the launch command, the passed
`east5_launch_safety --expected-child-zone us-east5-b` log, four passing launcher tests, and the redacted submit log.
Collect with `collect_delphi_3e18_validation_results_20260906.py` once the launch is registered there.
