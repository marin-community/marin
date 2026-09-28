# Seed-matched fairness repeats at 3e18 (submitted 2026-09-07 11:35 PDT)

Launcher `experiments/domain_phase_mix/launch_delphi_fairness_repeats_3e18.py` (12 v6e-8 runs, run ids 7,403,000+),
command in `submission/launch_command.sh` (east5 safety check passed, dry-run manifests in `launch_dry_run/`).
Conditional on the frozen-procedure validation passing its gate (freeze handoff Section 13).

| candidate | policy | trainer seeds | data seed |
|---|---|---|---|
| olmix_u_kl0p1_cap04 | Olmix best Uncheatable (KL 0.1, cap 4; measured 1.0022 on v5p-8 at data seed 662005) | 0, 1, 2 | 666200 |
| lwspu_u_snc_cap06 | frozen Uncheatable proposal (KL-0 control at trainer seed 0 = validation run) | 1, 2 | 666200 |
| olmix_t9_kl0p005_cap05 | Olmix best Table 9 (KL 0.005, cap 4; measured 1.0769 on v5p-8 at data seed 662009) | 0, 1, 2 | 662009 |
| lwspu_t9_snc_cap06 / cap08 | frozen Table-9 proposals (KL-0 controls at trainer seed 0 = validation runs) | 1, 2 | 662009 |

The Olmix rows are the exact runtime mixtures of the trained runs (`olmix_quantization.md`). Every comparison is
then three (data seed, trainer seed) pairs per objective on the same hardware. Collect with a group-aware
extension of `collect_delphi_3e18_validation_results_20260906.py` (candidate ids repeat across trainer seeds).

Iris parent `/calvinxu/dm-delphi-3e18-fairness-repeats-v6e8-20260908`, Fieldbook `exp_01m1yjft203bdg75hg0rdv4b5g`; submitted after the frozen-procedure validation passed its gate (review in `delphi_frozen_procedure_validation_3e18_20260908/review.md`).
