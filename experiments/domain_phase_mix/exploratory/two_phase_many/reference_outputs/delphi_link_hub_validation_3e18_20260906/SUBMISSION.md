# Delphi link-plus-hub Table-9 validation at 3e18

Submitted on 2026-09-06 at 13:59 UTC as `/calvinxu/dm-delphi-3e18-lwspu-linkhub-v6e8-20260906`
([Iris parent](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fdm-delphi-3e18-lwspu-linkhub-v6e8-20260906)),
under the same overnight authorization as the bounded-link validation.

| Optimization target | Epoch cap | Surrogate | Data seed | Candidate |
| --- | ---: | --- | ---: | --- |
| Table 9 | 6 | WSPU, bounded log-deficit link + total-hub interactions | 662009 | `lwspu_t9_bh_cap06` |
| Table 9 | 8 | WSPU, bounded log-deficit link + total-hub interactions | 662009 | `lwspu_t9_bh_cap08` |

KL coefficient zero, trainer seed zero, the existing 3e18 recipe, both suites evaluated; CPU parent in
us-east5-a, v6e-8 children in us-east5-b, all paths on gs://marin-us-east5; max_concurrent 2.

The surrogate is `weibull_softplus_unscaled@log_deficit_bounded_link_total_hub` (registry entry added
2026-09-06: WSPU's 78 columns plus signed Scheffé products of the total benefit signal with each bucket's,
under the bounded log-deficit link), fitted on the canonical 280 rows in
`delphi_link_variants_selection_20260906` (fold -1). On the frozen archive it is the best Table-9 selector
seen: regret@1 0.0140, rank 9/157, best-of-10 0.0084, optimism −0.008, RMSE 0.023, with a within-source-block
regret contrast against WSPU of −0.0005 [−0.0017, +0.0001] (the bounded link alone: +0.0036). Heads were
reconstructed from the saved shapes and ridges (parity < 1e-15), minimized under each cap from the five
standard starts, polished with a coarser finite-difference step because SLSQP's default step reports
"positive directional derivative" on this objective, and the lowest feasible endpoint was taken (all
polished restarts converged; endpoint spreads 0.0011 and 0.0025 BPB, so the surface is multimodal at that
level). Runtime rounding and one-count exchange as before; runtime penalty below 1e-3 BPB.

Predicted values (hub surrogate) at the runtime policies: cap 6 1.0747, cap 8 1.0737; it predicts the κ-0
WSPU comparator policies at 1.0813 / 1.0804, so it expects gains of 0.0066 / 0.0067 BPB. WSPU predicts the
hub optima at 1.0196 / 1.0156. TV from the WSPU κ-0 policies 0.116 / 0.128; hull distance from the panel
0.33 / 0.34; effective buckets 14.1 / 12.7. Controls: `wspu_table9_cap06` 1.0722 measured, the cap-8
control from the same sweep, the bounded-link runs `lwspu_t9_bl_cap06/08` (predicted by the link at 1.0841 /
1.0839), and the coupling runs.

Evidence: `offline_materialization/` (restart diagnostics with the polish flags, continuous weights,
`policies.json`), `runtime_materialization/candidate_weights.csv` (sha256
`ff7d002fa65182d4fe503214ac2cb15c9fe8bda3b1809a9d2e40a99843d456ba`), `launch_dry_run/`,
`submission/` (launch command, safety log with `--expected-child-zone us-east5-b`, dry-run log, eight
launcher tests, lint, submit log).
