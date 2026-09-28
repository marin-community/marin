# Floor replicates at 3e18: prepared 2026-09-07, NOT submitted

Status: launcher, tables, tests, dry run and the east5 safety check are done; the submission waits for Calvin's
decision, because 23 children of the coupling, link, hub and factorial launches were still queued behind v6e-8
capacity when this was prepared (only two coupling children had started).

Why: the bank's Table-9 floor is a 0.023 BPB band of competing optima measured once each (run SD 0.0038), and no
panel-fitted surrogate orders that band (`delphi_top_band_ordering_20260906/`). The best-measured coordinate
(1.0579, one run) is 0.006 below the 26-run centre (1.0639); whether any single-run coordinate truly beats the
centre is unknown at that noise. Two more runs per candidate (data seed 662009, trainer seeds 0 and 1) give each
a three-run mean (SE 0.0022) and a difference from the centre with SE about 0.0024 BPB.

Rule (fixed before looking at anything but the measured values): the five best optima-stratum coordinates with at
most two existing runs. Weights are the bank's recorded targets rounded to the 2048-count runtime grid (four of
the five were recorded as continuous targets; rounding moves at most 0.003 TV). The cap suffix `cap17` is the
loader's requirement; no cap binds.

| candidate | source | runs | measured | TV to centre | TV moved by grid | max epochs | kernel forecast (TV 0.05) |
|---|---|---:|---:|---:|---:|---|---:|
| `floor_1_e8df5e6e_cap17` | delphi-corrective-hpr-280-tied-controls-3e18 | 1 | 1.0579 | 0.136 | 0.0023 | 16.0 (dolma3_wikipedia) | 1.0647 |
| `floor_2_d3ffb3bd_cap17` | hpr_300m_to_3e18_optimum_validation_panel_20260720 | 1 | 1.0660 | 0.120 | 0.0028 | 16.0 (dolma3_wikipedia) | 1.0660 |
| `floor_3_a446352f_cap17` | delphi_decoupled_phase_information_validation_3e18_2026071 | 2 | 1.0663 | 0.077 | 0.0025 | 13.5 (dolma3_wikipedia) | 1.0662 |
| `floor_4_5969f449_cap17` | aggregate_v_epoch_cap | 1 | 1.0664 | 0.332 | 0.0000 | 8.0 (dolma3_finemath_3plus) | 1.0725 |
| `floor_5_c8a7abc2_cap17` | delphi_symmetric_sepheads_geometry_frontier_3e18_20260711 | 1 | 1.0678 | 0.128 | 0.0025 | 13.0 (dolmino_stem_heavy_crawl) | 1.0759 |

Ten runs, about 3.3e19 FLOPs. Launcher `experiments/domain_phase_mix/launch_delphi_floor_replicates_3e18.py`
(run ids 7,395,000+ and 7,395,100+; W&B Table-9 group
`olmo_base_eval_table9_delphi_3e18_one_phase_floor_replicates`); tables `candidate_weights_t0.csv` and
`candidate_weights_t1.csv` (identical content, sha256
`f4913b5f093f083115b08449289631c368291fd87d35601be8af6e9b4929e23b`); tests
`tests/test_launch_delphi_floor_replicates_3e18.py` (3 passed); dry run under `launch_dry_run/`; the launch
command in `submission/launch_command.sh` passed
`east5_launch_safety --expected-child-zone us-east5-b` (`submission/east5_launch_safety.log`).

To submit (from a subshell that sources `~/.zshrc.secrets` with `set -a`, output piped through the key
redaction), run `submission/launch_command.sh`, then record the parent, runs and jobs in Fieldbook as for the
factorial (`exp_01m1vh5s802cj685fbzcx3w475`).
