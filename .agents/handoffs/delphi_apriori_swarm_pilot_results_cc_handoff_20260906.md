# Handoff: Delphi a-priori swarm pilot results (2026-09-06)

The 37-run pilot completed. The preregistered Table-9 gate passed 0 of 4 targets, so the experiment stops after the
pilot. Do not launch the remaining 143 new rows. A follow-up WSPU fit found repeatable signal within the paired
support interventions, but adding all 37 runs did not improve prediction or optimum selection on the pre-pilot
held-out bank.

## What ran

- Frozen design: `c5ec2b0ae1b5c68dc44f6ede1a5caa1bd2918f5bfd9f6d0f23be508637856bd6`.
- Hardware: `v6e-8` in `us-east5-b`.
- Canary: `/calvinxu/delphi-apriori-swarm-canary-v6e-east5b-20260904-retry1`, succeeded.
- Pilot: `/calvinxu/delphi-apriori-swarm-pilot-v6e-east5b-20260905-retry1`, succeeded.
- Measurements: 37 of 37 runs have complete Uncheatable and 51-component Table-9 results.
- Pilot composition: three proportional controls, two anchor-B controls, and 32 support interventions. The
  interventions cover two anchors, four target buckets, half- and quarter-pool support, and two paired seed blocks.
- The canary's registry exposure matched all 39 frozen materialized-epoch columns to maximum absolute error
  `2.11e-15`. Its pool fraction and seeds survived the run-config-to-registry path, and its coordinate remained
  distinct from the full-support coordinate.

Materialized results and provenance:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_materialization/heldout_materialization_manifest.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_materialization/heldout_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_materialization/table9_components.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_materialization/uncheatable_components.csv`

## Preregistered gate

Table 9 was the decision metric. The gate required at least two of the four target buckets to pass at either anchor.
A target-anchor pair had to satisfy all three conditions: quarter-pool mean effect above `2 sigma`, the same sign in
both seed blocks, and a quarter-pool effect at least as large as the half-pool effect in absolute value.

- Pooled Table-9 seed SD: `0.008532` BPB.
- Preregistered threshold: `0.017064` BPB.
- Largest absolute quarter-pool mean effect among the eight target-anchor pairs: `0.005111` BPB.
- Targets passing: `0/4`.
- Decision: `stop_after_pilot`.

Some effects agreed in sign across blocks, but none cleared the magnitude threshold. Under the preregistered rule,
the result is repetition below the Table-9 noise floor at 3e18 FLOPs for these four buckets. Uncheatable was reported
but did not affect the decision; its pooled seed SD was `0.000636` BPB.

Gate artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_gate/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_gate/gate_anchor_summary.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/pilot_gate/gate_manifest.json`

## Predictive-value follow-up

After applying the gate, we fixed the surrogate to `weibull_softplus_unscaled` and ran two leakage-free Table-9
comparisons. Fits remain componentwise over all 51 Table-9 components and use the existing training-only inner-fold
selection. Pool-fraction-aware exposures are preserved throughout.

1. **Cross-block support prediction.** The baseline trains on the frozen 280-row panel. The augmented fit adds one
   seed block's 16 interventions and two matched controls, then predicts the other block's 16 interventions; the
   direction is reversed for the second half. Across 32 held-out runs, RMSE fell from `0.03030` to `0.01947` BPB.
   The paired-condition bootstrap delta was `-0.01083`, with 95% interval `[-0.01968, -0.00309]`. The support
   interventions therefore contain repeatable local signal.
2. **Pre-pilot external bank.** The baseline 280-row fit and the 317-row fit with all pilot runs predict the same 247
   complete pre-pilot registry coordinates; every coordinate sourced from this pilot is excluded. Pooled RMSE rose
   from `0.03272` to `0.03869` BPB. The 90 intervention coordinates improved by `-0.00322` BPB, but the 95% interval
   `[-0.00583, 0.00069]` crossed zero. On the 157 model-optimum archive coordinates, RMSE rose from `0.03783` to
   `0.04663` BPB.

Optimum selection on the 157-coordinate archive also worsened:

| Fit | Regret@1 | Selected rank | Spearman |
|---|---:|---:|---:|
| Frozen 280-row panel | 0.01574 | 14/157 | 0.894 |
| Panel plus all 37 pilot runs | 0.02614 | 39/157 | 0.563 |

The cross-block result shows that the pilot measured a reproducible local response. The external-bank result shows
that the current pooled WSPU fit does not convert those measurements into better global Table-9 prediction or
selection. The 317-versus-280 comparison measures incremental data value and is over budget; it is not a
matched-budget comparison of swarm designs.

Predictive-value artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_delphi_apriori_pilot_predictive_value_20260906.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/predictive_value_20260906/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/predictive_value_20260906/predictions.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/predictive_value_20260906/paired_bootstrap.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_apriori_swarm_280_20260904/predictive_value_20260906/archive_selection_metrics.csv`

The analysis produced identical output hashes with 8 and 4 workers. The script passes the repository pre-commit
checks and Pyrefly.

## Request to CC

Please review the predictive-value analysis for leakage or an incorrect comparison boundary. If it is sound, treat
the pilot as evidence to stop the second wave for the current Table-9/WSPU objective. A future attempt to use these
37 runs would need a newly specified fitting protocol, such as intervention-aware weighting or a hierarchical
response model. Any such attempt is post-pilot model development, not part of the preregistered gate.
