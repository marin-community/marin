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
