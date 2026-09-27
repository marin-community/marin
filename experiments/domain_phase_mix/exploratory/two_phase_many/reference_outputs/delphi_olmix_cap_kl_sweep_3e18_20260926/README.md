# Matched-Olmix epoch-cap x KL sweep at 3e18 (2026-09-26)

Calvin's design: the matched Olmix policy (Qwen-fitted per-task log-linear laws from
`../delphi_matched_olmix_3e18_20260908/laws_*.json`, exact KL-regularized proposer) over epoch caps
{1, 4, 8, 12, uncapped} and the KL axis of the paper's MARINER KL table (lambda in {0, 0.005, 0.01, 0.025, 0.05,
0.075, 0.1, 0.2, 0.5}), for Uncheatable and OlmoBaseEval Easy: 90 cells. One run per distinct mixture, no repeats.

- `materialize_delphi_olmix_cap_kl_sweep_20260926.py` writes `candidate_weights.csv` (39 runs: 18 Uncheatable, 21
  OlmoBaseEval Easy; sha256 c138bfe2...), `grid_map.csv` (every cell, the run that measures it, and the distance to
  that run) and `summary.json`. Cells within total variation 0.01 of an earlier cell reuse its run (the cap stops
  binding at larger lambda); cap 1 runs only at lambda 0. Uncapped runs carry the nominal cap ceil(max epochs).
- Launcher `experiments/domain_phase_mix/launch_delphi_olmix_cap_kl_sweep_3e18.py`: v5p-8 in us-east5-a (the swarm's
  training hardware), data seeds 666200 (Uncheatable) and 662009 (suite), trainer seed 0, run ids 7,500,000+ and
  7,500,100+; OlmoBaseEval Easy evaluations on the swarm's v6e-8 evaluation resources.
- Canary: `olmixq_u_kl0p05_cap04` (the current Uncheatable policy; 1.0022 as a 3-seed mean on v6e-8), job
  `/calvinxu/dm-delphi-3e18-olmix-cap-kl-canary2-v5p8-20260926` (the first submission failed on a launcher count
  check before any training). Its training and evaluation steps are identical to the full launch's, so the full
  launch reuses it. `submission/early_release.py` submits `submission/launch_full_command.sh` once the canary has
  committed a checkpoint.
- Fieldbook experiment `exp_01m3ggw616j8406rh5n1pe3z50`.
- Appendix table (planned): all 90 cells filled; cells measured by another cell's run in italics.
