# Matched Olmix validation at 3e18 (submitted 2026-09-08 14:13 PDT)

Launcher `experiments/domain_phase_mix/launch_delphi_matched_olmix_3e18.py` (11 v6e-8 runs, run ids 7,430,000+ Uncheatable and
7,430,100+ suite), Iris parent `/calvinxu/dm-delphi-3e18-matched-olmix-v6e8-20260908`, Fieldbook experiment id in
`submission/fieldbook_experiment_id.txt`. Collect with `collect_delphi_3e18_validation_results_20260906.py --launch matched_olmix`.

Why: the paper's trained Olmix comparators were fitted in June/July on the earlier Llama 200M/6B data, before the Qwen3
3e18 swarm existed; the Uncheatable ones used one aggregate head on tied-phase two-phase runs. These policies fit the
reference per-task log-linear law (Huber 0.01, 48 starts, seed 0) on the frozen 280-run Qwen swarm
(`delphi_offline_selection_20260906/inputs/panel.npz`) and solve Olmix's exact capped KL proposer (cap 4) with cvxpy
(`materialize_delphi_matched_olmix_20260908.py`), then round to the 1/2048 runtime grid with the remainder assigned to the
largest buckets that still have cap slack, so every candidate honestly carries `cap04`.

Design: Uncheatable KL 0.05 (ladder coefficient) and 0.1 (development-sweep winner) at data seed 666200, trainer seeds 0/1/2;
suite KL 0.005 at data seed 662009, trainer seeds 0/1/2; unpenalized KL 0 policies once each at trainer seed 0. The seeds are
those of the seed-matched repeats, so paired differences against MARINER and the native Olmix repeats are available.

| candidate | target | KL | predicted (runtime) | max epochs | active | at cap | TV to native | predicted at native | measured native |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `olmixq_u_kl0p05_cap04` | uncheatable | 0.05 | 0.9939 | 3.18 | 35 | 0 | 0.313 | 1.0089 | 1.0039 |
| `olmixq_u_kl0p1_cap04` | uncheatable | 0.1 | 1.0039 | 2.65 | 37 | 0 | 0.191 | 1.0101 | 1.0022 |
| `olmixq_u_kl0_cap04` | uncheatable | 0.0 | 0.9736 | 4.00 | 7 | 6 | 0.696 | 1.0113 | 1.0146 |
| `olmixq_t9_kl0p005_cap04` | table9 | 0.005 | 1.0953 | 4.00 | 27 | 5 | 0.293 | 1.1003 | 1.0769 |
| `olmixq_t9_kl0_cap04` | table9 | 0.0 | 1.0944 | 4.00 | 12 | 7 | 0.330 | 1.1008 | 1.0851 |

`solutions.csv` holds continuous, runtime and native weights per bucket; `laws_*.json` the fitted laws; `materialize.log` the run.
