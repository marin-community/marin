# Fully convex and additive MARINER proposals at 3e18: measured results (collected 2026-09-14 07:10 PDT)

Twelve v6e-8 runs of `launch_delphi_convex_additive_3e18.py` (Iris `/calvinxu/dm-delphi-3e18-convex-additive-v6e8-20260914-r3`,
Fieldbook experiment `exp_01m2fny7wgdnzpbbrwpv29wey7`), collected with
`collect_delphi_3e18_validation_results_20260906.py --launch convex_additive`; proposals and pre-registered predictions in
`../delphi_convex_additive_proposals_3e18_20260914/`. Paired against MARINER's seed-matched runs (seed 0 from the frozen
validation, seeds 1 and 2 from the fairness repeats) by `summarize_delphi_comparator_proposals_20260909.py`.

| candidate | measured (seeds 0/1/2) | mean ± SD | paired Δ vs MARINER ± SE | own prediction | MARINER's prediction |
|---|---|---:|---:|---:|---:|
| `cmp_u_cvx_cap06` | 0.9902 / 0.9920 / 0.9920 | 0.9914 ± 0.0011 | +0.0089 ± 0.0003 | 0.9885 | 0.9846 |
| `cmp_u_add_cap08` | 0.9918 / 0.9917 / 0.9926 | 0.9920 ± 0.0005 | +0.0096 ± 0.0007 | 0.9445 | 0.9842 |
| `cmp_t9_cvx_cap08` | 1.0652 / 1.0656 / 1.0668 | 1.0659 ± 0.0009 | -0.0019 ± 0.0027 | 1.0640 | 1.0641 |
| `cmp_t9_add_cap08` | 1.0709 / 1.0705 / 1.0705 | 1.0706 ± 0.0002 | +0.0028 ± 0.0030 | 1.0051 | 1.0656 |

MARINER: Uncheatable 0.9814 / 0.9834 / 0.9825, suite 1.0682 / 1.0624 / 1.0727. Both ablations cost about 0.01 BPB on
Uncheatable, twice the quadratic and spline proposals, and are within seed noise of MARINER on the suite. These rows are
Table 3 and `tab:a-comparator-proposals` of the paper; the additive row replaces the earlier calibration-protocol run.
