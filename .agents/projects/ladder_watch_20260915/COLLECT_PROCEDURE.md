# Collecting a landed rung — the handoff's two commands are only step 1

**Decision (main session, 2026-09-15):** run the full four-step sequence below. The audit is the gate
the paper's table rests on; a bare collector output puts the paper session back at the same step.
Report the audited rows and the final permanent checkpoint step.

`reference_outputs/` is gitignored (`.git/info/exclude:28`), so `measured_results.csv` has **no git
backup**. Copies of the current enriched files are in `csv_backup_20260915/`.

The live `reference_outputs/delphi_*/measured_results.csv` is byte-identical to
`frozen_scaling_update_20260913/<ladder>/audited_measured_results.csv`. It carries three columns the
collector alone cannot produce: `uncheatable_ao3_english_bpb`, `uncheatable_frozen_weighted_bpb`,
`uncheatable_bpb_schema_version`. Running only the two collector commands rewrites the file in the
base schema and silently drops all three (verified 2026-09-15 17:12 PDT, then restored from backup).

Full sequence when a rung's Table-9 eval succeeds:

```bash
cd /Users/calvinxu/Projects/Work/Marin/marin
R=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs
cp $R/delphi_frozen_procedure_scaling_v6e_20260908/measured_results.csv /tmp/mar.bak.csv
cp $R/delphi_matched_olmix_scaling_v6e_20260910/measured_results.csv /tmp/olm.bak.csv

# 1. collectors (base schema)
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/collect_delphi_frozen_procedure_scaling_20260908.py
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/collect_delphi_frozen_procedure_scaling_20260908.py --ladder matched_olmix

# 2. feed the audit its input
cp $R/delphi_frozen_procedure_scaling_v6e_20260908/measured_results.csv $R/frozen_scaling_update_20260913/mariner/measured_results.csv
cp $R/delphi_matched_olmix_scaling_v6e_20260910/measured_results.csv $R/frozen_scaling_update_20260913/matched_olmix/measured_results.csv

# 3. audit — adds the three columns and writes sha256 receipts + metric_audit.json
uv run $R/frozen_scaling_update_20260913/audit_measured.py

# 4. copy the audited result back to the live path
cp $R/frozen_scaling_update_20260913/mariner/audited_measured_results.csv $R/delphi_frozen_procedure_scaling_v6e_20260908/measured_results.csv
cp $R/frozen_scaling_update_20260913/matched_olmix/audited_measured_results.csv $R/delphi_matched_olmix_scaling_v6e_20260910/measured_results.csv
```

`audit_measured.py` asserts the final step matches `launch_dry_run/run_manifest.json`'s
`expected_checkpoint_step`, that the checkpoint metadata has `is_temporary == false`, that the Table-9
result has 51 finite components whose mean matches `table9_macro_bpb`, and that the HF export
`<run_root>/hf/step-<step>/config.json` exists. It therefore only passes for a genuinely landed rung,
including its HF export — so failures here are informative, not noise.

## Step 5 (paper session, added 2026-09-16)

The compute-equivalence script reads `frozen_scaling_update_20260913/figure/plotted_points.csv`, not the live CSVs, so after step 4 the paper session rebuilds that figure directory from the audited files before refitting:

```bash
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/plot_scaling_curves_placeholder_20260905.py \
  --output-dir $R/frozen_scaling_update_20260913/figure \
  --ladder-results $R/frozen_scaling_update_20260913/mariner/audited_measured_results.csv \
  --matched-olmix-results $R/frozen_scaling_update_20260913/matched_olmix/audited_measured_results.csv
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/compute_equivalence_20260915.py
```

The babysitting session stops after step 4 and reports; it does not run this step.

Figure 6 note (2026-09-17): the manuscript's Figure 6 is Codex's fitted-curve version, built by `revision_notes/20260916_figure6_scaling_curve_preview/build_preview.py` (copy it into a new dated revision folder, update its point-count assertion, run with `PYTHONPATH=<marin> uv run --script build_preview.py`, then copy the three `figure6_scaling_curve_preview.*` outputs onto `figures/r1_scaling_curves_placeholder.*`). Do not run `plot_scaling_curves_placeholder_20260905.py --drive-dir figures/` any more; it overwrites the fitted figure.
