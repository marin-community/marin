# Handoff: abrupt-switch repeats for the ramp comparison (2026-10-07)

## Goal

Retrain the abrupt switch of the 4x (m400) schedule on the four seeds it lacks in the current training stack, so that
every ramp run pairs with an abrupt run from the same software. Calvin asked for these four runs.

Why: the paper's Appendix E.3 (Figure 25B) plots each ramp's gain in final programming-languages BPB over the abrupt
switch. Only seed 20260821 has an abrupt run in the October stack (`tramp_r3d28260_m400_abrupt_s0821`, the drift
canary). The other four seeds compare against the August runs (JAX 0.10.1), and the between-launch noise widens those
intervals to about ±1.7 to ±3.3 ×10⁻³ BPB, against ±0.5 to ±1.0 for comparisons within the October runs. Even the
5,632-step ramp's cost then spans zero against the abrupt switch (−2.8 [−6.1, +0.5] ×10⁻³) while it is clear against
the 512-step ramp (−3.0 [−4.0, −2.0], 5/5 seeds).

## Runs

Four runs, arm `abrupt`, seeds 20260822, 20260823, 20260824, 20260825, identical to the seed-0821 canary: 28,260
steps, switch and decay onset at step 22,608, StarCoder 0 → 0.28 on the 4x subset, the frozen confirmation schedule
with its realized sequence counts. Run ids `tramp_r3d28260_m400_abrupt_s0822` to `_s0825`, same W&B group
(`pinlin_calvin_xu/data_mixture/starcoder_wsd80_transition_ramp_20261006`).

Estimated cost: about 9.3×10¹⁸ FLOPs per run (3.7×10¹⁹ in all), about 3.6 hours each on v5p-8 in us-central1-a, so
about 4 hours with all four running. If the cost or placement differs materially, check with Calvin first.

## How

- Launcher: `experiments/domain_phase_mix/launch_starcoder_wsd80_transition_ramp.py` (`--design main|quick`). Add a
  variant with no ramp widths and the abrupt arm on the four seeds, write its frozen design JSON beside the others
  (e.g. `starcoder_wsd80_transition_ramp_abrupt_design_20261007.json`), and run `--audit-manifest`,
  `--audit-runtime-identities`, and `--dry-run` before submitting. `build_design` already checks the abrupt counts
  against the frozen confirmation.
- Submission as for the ramp launches (see the memory note `curriculum-transition-switch-transient.md`): through
  `marin.run.iris_run`, `--memory 3GB --disk 8GB`, excluding `exploratory/two_phase_many/` from the bundle;
  `max_concurrent` 4. Detach the launch (nohup, own log, Monitor) and log done/total with an ETA in Pacific time.
- Evaluation: evaluate the four final checkpoints offline with the same code as the other 26 finals (as in
  `/calvinxu/dm-switch-window-eval-finals-20261006`). Training-time PL values are not comparable (offset 0.0133 BPB
  since August), so use only the offline finals.

## Analysis and deliverable

Extend `collect_starcoder_wsd80_transition_ramp_20261006.py` and update `ramp_endpoints.csv` and `report.md` in
`reference_outputs/starcoder_wsd80_transition_transient_20261006/`:

- Per ramp width (128, 256, 512, 2,048, 5,632): gain over the October abrupt switch = abrupt PL − ramp PL at the same
  seed (positive means the ramp is better), mean with 95% paired-t interval over the five seeds, and the number of
  seeds in which the ramp is worse.
- Drift check: October abrupt minus August abrupt per seed, mean and interval.
- The two-phase gain of each arm against the August tied runs, as now.
- One table with those numbers at the top of a new report section, and C4 BPB of the same comparison for the widest
  ramp (the paper says the widest ramp lowered C4 slightly).

Do not edit the paper; Calvin brings the numbers into Figure 25B and the E.3 text in the paper session.

## Rules

Read `AGENTS.md` and the `use-iris` skill. Never stop or restart an Iris cluster. Keep compute and data in
us-central1. Use paired seeds as the inferential unit. Report times in Pacific. Commit only when asked, with no
attribution lines (`reference_outputs` needs `git add -f`).
