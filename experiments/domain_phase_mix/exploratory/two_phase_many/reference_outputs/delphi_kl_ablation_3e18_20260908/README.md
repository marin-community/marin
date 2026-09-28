# KL ablation of the frozen procedure at 3e18 (submitted 2026-09-07 11:35 PDT)

Launcher `experiments/domain_phase_mix/launch_delphi_kl_ablation_3e18.py` (16 v6e-8 runs, run ids 7,404,000+),
command in `submission/launch_command.sh` (east5 safety check passed, dry-run manifests in `launch_dry_run/`).
Conditional on the frozen-procedure validation passing its gate (freeze handoff Section 13).

Policies: the frozen fits (`delphi_frozen_procedure_validation_3e18_20260908/fits/`) minimized with
`kl x KL(w || proportional)` added to the aggregate, kl in {0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5}
(`mixture_selection.py optimize --fit <fit> --cap 6|8 --kl <kl>`, commit 14fda28); Uncheatable under cap 6 and
Table 9 under cap 8, both inactive. `policy_summary.csv` gives the penalized objective per policy; the raw surrogate
prediction is reported separately at collection (KL 0.005 to 0.5: Uncheatable 0.9817 to 1.0103, Table 9 1.0633 to
1.1163). The KL-0 controls are the frozen-procedure validation runs at the same seeds. Collect with
`collect_delphi_3e18_validation_results_20260906.py --launch kl_ablation`.

Iris parent `/calvinxu/dm-delphi-3e18-lwspu-kl-ablation-v6e8-20260908`, Fieldbook `exp_01m1yjg60x0tz6tjrzj80pxm6e`; submitted after the frozen-procedure validation passed its gate.
